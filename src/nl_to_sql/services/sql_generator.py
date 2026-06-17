"""SQL generator — builds prompts and calls the LLM to produce SQL."""
import re
import structlog

from nl_to_sql.core.exceptions import RateLimitError, SQLGenerationError
from nl_to_sql.core.interfaces.i_llm_provider import ILLMProvider
from nl_to_sql.core.models.sql_result import GeneratedSQL, LLMResponse, ValidationResult
from nl_to_sql.services.feedback_learner import FeedbackLearner
from nl_to_sql.infrastructure.observability.tracing import trace_function, set_span_attribute

logger = structlog.get_logger(__name__)

# Regex to strip markdown SQL fences: ```sql ... ``` or ``` ... ```
_SQL_FENCE_RE = re.compile(r"```(?:sql)?\s*([\s\S]*?)```", re.IGNORECASE)


_SYSTEM_PROMPT_TEMPLATE = """You are an expert {dialect} SQL generator.

Your job is to convert a natural language question into a valid, efficient, and syntactically correct {dialect} SQL query.

---

### 🚫 STRICT RULES:
1. Output ONLY the SQL query — no explanations, no markdown, no comments, no extra text.
2. Only generate read-only queries (SELECT statements).
   Never generate INSERT, UPDATE, DELETE, DROP, ALTER, CREATE, or TRUNCATE.
3. Use ONLY the tables and columns provided in the schema context.
4. Ensure all referenced tables and columns exist in the schema.
5. If the question cannot be answered using the schema, return exactly:
   -- CANNOT_ANSWER

---

### 🧠 DIALECT AWARENESS:
- Follow {dialect} syntax strictly.
- Use dialect-specific features correctly (e.g., LIMIT / TOP / FETCH, date functions, string functions).
- Ensure the query is fully compatible with {dialect}.

---

### 📊 COLUMN SELECTION:
- Do NOT use SELECT * unless the user explicitly asks for:
  "all columns", "all data", "everything", "full details".
- Otherwise, always select only the required columns.
- Always qualify column names with table names when ambiguity exists.

---

### 🔗 JOIN LOGIC:
- Use JOINs only when necessary.
- Infer relationships using logical keys (e.g., id, user_id, product_id, policy_id).
- Always include correct JOIN conditions.
- Use clear, short table aliases (e.g., c, o, p).
- Avoid unnecessary joins.

---

### 📈 AGGREGATION RULES:
- Use GROUP BY when aggregation functions (SUM, COUNT, AVG, etc.) are used.
- Include all non-aggregated columns in GROUP BY.
- Use meaningful aliases for computed columns.

---

### 🧾 FILTERING & SORTING:
- Use WHERE for filtering conditions.
- Use ORDER BY when sorting is implied (e.g., "top", "highest", "latest").
- Apply LIMIT / TOP appropriately based on {dialect}.

---

### ⚠️ QUERY QUALITY:
- Prefer simple, efficient, and readable queries.
- Avoid redundant columns, joins, and conditions.
- Avoid ambiguous column references.
- Handle NULL values properly when relevant.

---

### 🎨 SQL STYLE PREFERENCES:
{sql_style_instructions}

---

### 🎯 CUSTOM USER INSTRUCTIONS:
{custom_instructions_section}

---

### 🔁 ERROR HANDLING & RETRY:
- If a previous query and validation error are provided:
  - Fix ONLY the issue mentioned in the error.
  - Do NOT modify correct parts of the query.
  - Preserve the original intent.

- Common fixes include:
  - Invalid table or column names
  - Missing or incorrect JOIN conditions
  - Syntax errors
  - Incorrect function usage
  - Missing quotes for string literals

---

### 📦 SCHEMA CONTEXT:
{schema_context}

---

### 📚 FEW-SHOT EXAMPLES:
{few_shot_examples}

---

### ❓ USER QUESTION:
{question}

---

### ✅ OUTPUT:
Return ONLY a valid JSON object with the following structure (no markdown formatting, no comments, just raw JSON):
{
  "sql": "The generated SQL query",
  "follow_up_questions": ["Question 1", "Question 2", "Question 3"],
  "suggested_chart": {
    "type": "bar | line | pie | none",
    "x_axis": "column_name",
    "y_axis": "column_name"
  }
}
If the data cannot be graphed (e.g., single row, generic select *), set "type" to "none".
"""

_FEW_SHOT_EXAMPLES = """EXAMPLES:

Q: How many customers are there?
A: {
  "sql": "SELECT COUNT(*) AS customer_count FROM customers;",
  "follow_up_questions": ["What is the breakdown of customers by region?", "How many new customers joined this month?", "Who are the top 5 customers by order count?"],
  "suggested_chart": {"type": "none", "x_axis": "", "y_axis": ""}
}

Q: List the top 3 products by price.
A: {
  "sql": "SELECT product_id, name, price FROM products ORDER BY price DESC LIMIT 3;",
  "follow_up_questions": ["What is the average price of all products?", "Which category has the most expensive products?", "Show me the cheapest products."],
  "suggested_chart": {"type": "bar", "x_axis": "name", "y_axis": "price"}
}

Q: What is the total revenue per category?
A: {
  "sql": "SELECT p.category, SUM(oi.quantity * oi.unit_price) AS total_revenue FROM order_items oi JOIN products p ON oi.product_id = p.product_id GROUP BY p.category ORDER BY total_revenue DESC;",
  "follow_up_questions": ["Which category sold the highest quantity of items?", "What is the revenue trend over the last 6 months?", "Who are the top buyers in the most profitable category?"],
  "suggested_chart": {"type": "pie", "x_axis": "category", "y_axis": "total_revenue"}
}
"""

def _build_style_instructions(style_hints: dict | None) -> str:
    """Translate user SQL style preferences into prompt instructions."""
    if not style_hints:
        return "Follow standard SQL formatting conventions."
    parts: list[str] = []
    cte = style_hints.get("cte_pref")
    if cte == "cte":
        parts.append("- Prefer CTEs (WITH … AS (…)) over inline subqueries whenever possible.")
    elif cte == "subquery":
        parts.append("- Prefer inline subqueries over CTEs; avoid WITH clauses.")
    kw = style_hints.get("keyword_case")
    if kw == "upper":
        parts.append("- Write ALL SQL keywords in UPPERCASE (SELECT, FROM, WHERE, JOIN, etc.).")
    elif kw == "lower":
        parts.append("- Write all SQL keywords in lowercase (select, from, where, join, etc.).")
    alias = style_hints.get("alias_style")
    if alias == "implicit":
        parts.append("- Use implicit aliases — omit the AS keyword: write `table_name t` not `table_name AS t`.")
    elif alias == "as":
        parts.append("- Always use the explicit AS keyword for aliases: `table_name AS t`.")
    indent = style_hints.get("indent")
    if indent and isinstance(indent, int) and indent > 0:
        parts.append(f"- Indent each SQL clause continuation with {indent} spaces.")
    max_rows = style_hints.get("max_result_rows")
    if max_rows and isinstance(max_rows, int) and max_rows > 0:
        parts.append(
            f"- Always append LIMIT {max_rows} to every SELECT query unless the question "
            f"explicitly asks for all rows or a LIMIT clause is already present."
        )
    return "\n".join(parts) if parts else "Follow standard SQL formatting conventions."


class SQLGeneratorService:
    """Invokes the LLM to generate SQL from a natural language question.

    SOLID:
      S — Only responsible for prompt construction and LLM invocation.
      D — Depends on ILLMProvider abstraction.
    """

    def __init__(
        self,
        llm_provider: ILLMProvider,
        dialect: str = "postgresql",
        temperature: float = 0.0,
        max_tokens: int = 1024,
        feedback_learner: FeedbackLearner | None = None,
    ) -> None:
        self._llm = llm_provider
        self._dialect = dialect
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._feedback_learner = feedback_learner

    @trace_function("llm.generate")
    async def generate(
        self,
        question: str,
        schema_context: str,
        dialect_override: str | None = None,
        error_feedback: str | None = None,
        style_hints: dict | None = None,
        model_override: str | None = None,
        custom_instructions: str | None = None,
    ) -> GeneratedSQL:
        """Generate SQL for the given question.

        Args:
            question: The natural language question.
            schema_context: Concatenated schema chunks from the retriever.
            dialect_override: Optional per-request SQL dialect.
            error_feedback: On retry attempts, include previous validation
                            errors so the LLM can self-correct.

        Returns:
            GeneratedSQL with the raw and cleaned SQL, plus token counts.

        Raises:
            SQLGenerationError: If the LLM call fails.
        """
        dialect = dialect_override or self._dialect
        log = logger.bind(dialect=dialect, question=question[:80])

        set_span_attribute("gen_ai.prompt", question)
        set_span_attribute("gen_ai.dialect", dialect)

        # Get learned patterns to avoid (Layer 4: Feedback Learning)
        learning_patterns = ""
        if self._feedback_learner:
            # Extract table names from schema_context to filter relevant patterns
            import re
            tables = re.findall(r'Table[:\s]+(\w+)', schema_context, re.IGNORECASE)
            learning_patterns = self._feedback_learner.get_learning_prompt(tables)

        # Apply token budget to custom instructions before injection
        effective_instructions: str | None = None
        if custom_instructions:
            from nl_to_sql.services.prompt_budget import PromptBudget, INSTRUCTIONS_CHAR_CAP
            budget = PromptBudget(max_completion_tokens=self._max_tokens)
            assembled = budget.assemble(
                system_preamble=_SYSTEM_PROMPT_TEMPLATE,
                user_query=question,
                custom_instructions=custom_instructions,
                schema_chunks=[schema_context],
            )
            effective_instructions = assembled["custom_instructions"]

        instructions_section = (
            effective_instructions
            if effective_instructions
            else "No custom instructions set."
        )

        # Using .replace instead of .format to avoid KeyError if schema_context
        # or few_shot_examples contain curly braces (common in SQL/JSON).
        system_prompt = (
            _SYSTEM_PROMPT_TEMPLATE
            .replace("{dialect}", dialect.upper())
            .replace("{schema_context}", schema_context + learning_patterns)
            .replace("{few_shot_examples}", _FEW_SHOT_EXAMPLES)
            .replace("{question}", question)
            .replace("{sql_style_instructions}", _build_style_instructions(style_hints))
            .replace("{custom_instructions_section}", instructions_section)
        )

        user_prompt = f"Question: {question}"
        if error_feedback:
            user_prompt += (
                f"\n\nPrevious attempt was invalid:\n{error_feedback}\n"
                "Please fix the SQL and try again."
            )

        log.debug("Calling LLM for SQL generation")
        try:
            response: LLMResponse = await self._llm.complete(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=self._temperature,
                max_tokens=self._max_tokens,
                response_format={"type": "json_object"},
                model_override=model_override,
            )
            # Set GenAI semantic conventions for observability and evaluations
            set_span_attribute("gen_ai.system", "groq")
            set_span_attribute("gen_ai.response.model", self._llm._model if hasattr(self._llm, "_model") else "groq")
            set_span_attribute("gen_ai.request.temperature", self._temperature)
            set_span_attribute("gen_ai.usage.input_tokens", response.prompt_tokens)
            set_span_attribute("gen_ai.usage.output_tokens", response.completion_tokens)
            set_span_attribute("gen_ai.usage.total_tokens", response.total_tokens)
        except RateLimitError:
            # Re-raise rate limit errors to preserve the correct error type
            raise
        except Exception as exc:
            raise SQLGenerationError(
                f"LLM call failed: {exc}", detail=str(exc)
            ) from exc

        import json
        try:
            payload = json.loads(response.content)
            raw_sql = payload.get("sql", "")
            follow_up_questions = payload.get("follow_up_questions", [])
            suggested_chart = payload.get("suggested_chart", None)
        except json.JSONDecodeError:
            log.warning("Failed to parse JSON from LLM, falling back to raw text")
            raw_sql = response.content
            follow_up_questions = []
            suggested_chart = None

        cleaned_sql = self._clean_sql(raw_sql)
        used_tables = self._extract_tables(cleaned_sql)
        log.info("SQL generated", tokens=response.total_tokens, used_tables=used_tables)

        return GeneratedSQL(
            raw_sql=raw_sql,
            cleaned_sql=cleaned_sql,
            dialect=dialect,
            validation=ValidationResult(is_valid=True),  # placeholder
            tokens_used=response.total_tokens,
            used_tables=used_tables,
            suggested_chart=suggested_chart,
            follow_up_questions=follow_up_questions,
        )

    @staticmethod
    def _clean_sql(raw: str) -> str:
        """Strip markdown fences and leading/trailing whitespace from SQL."""
        match = _SQL_FENCE_RE.search(raw)
        if match:
            return match.group(1).strip()
        return raw.strip()

    @staticmethod
    def _extract_tables(sql: str) -> list[str]:
        """Extract table names from SQL query.

        Parses FROM and JOIN clauses to identify which tables are actually used.

        Args:
            sql: The SQL query string.

        Returns:
            List of unique table names found in the query.
        """
        # Normalize SQL to uppercase for keyword matching
        sql_upper = sql.upper()
        tables = set()

        # Pattern 1: FROM table_name or FROM table_name alias
        # Matches: FROM customers, FROM customers c, FROM customers AS c
        from_pattern = re.compile(
            r'\bFROM\s+([a-zA-Z_][a-zA-Z0-9_]*)',
            re.IGNORECASE
        )

        # Pattern 2: JOIN table_name or JOIN table_name alias
        # Matches: JOIN orders, JOIN orders o, JOIN orders AS o, LEFT JOIN orders, etc.
        join_pattern = re.compile(
            r'\bJOIN\s+([a-zA-Z_][a-zA-Z0-9_]*)',
            re.IGNORECASE
        )

        # Extract tables from FROM clauses
        for match in from_pattern.finditer(sql):
            table_name = match.group(1).lower()
            # Filter out SQL keywords that might be mistakenly matched
            if table_name not in ('select', 'where', 'group', 'order', 'having', 'limit', 'offset'):
                tables.add(table_name)

        # Extract tables from JOIN clauses
        for match in join_pattern.finditer(sql):
            table_name = match.group(1).lower()
            tables.add(table_name)

        return sorted(list(tables))
