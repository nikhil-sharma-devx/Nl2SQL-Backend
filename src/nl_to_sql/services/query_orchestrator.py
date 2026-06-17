"""Query orchestrator — the main use-case: NL question → SQL response."""
import asyncio
import hashlib
import json
import time

import structlog

from nl_to_sql.core.exceptions import DatabaseExecutionError, RateLimitError, SQLGenerationError
from nl_to_sql.core.interfaces.i_cache import ICache
from nl_to_sql.core.interfaces.i_sql_validator import ISQLValidator
from nl_to_sql.core.models.query import QueryRequest, QueryResponse
from nl_to_sql.services.query_classifier import QueryClassifier
from nl_to_sql.services.query_expander import QueryExpander
from nl_to_sql.services.query_history import QueryHistoryService
from nl_to_sql.services.query_rewriter import QueryRewriter
from nl_to_sql.services.schema_retriever import SchemaRetriever
from nl_to_sql.services.sql_generator import SQLGeneratorService
from nl_to_sql.services.sql_validator import SQLValidatorService
from nl_to_sql.services.sql_column_validator import SQLColumnValidator
from nl_to_sql.rag.retrieval.table_selector import TableSelectorService
from nl_to_sql.infrastructure.observability.tracing import trace_function, set_span_attribute

logger = structlog.get_logger(__name__)


class QueryOrchestrator:
    """Coordinates the full NL-to-SQL RAG pipeline.

    Pipeline steps:
      1. Check cache for identical (question, dialect) pair.
      2. Retrieve relevant schema chunks via SchemaRetriever.
      3. Generate SQL via SQLGeneratorService.
      4. Validate SQL via ISQLValidator.
      5. Self-correct: if invalid, feed errors back to generator (up to max_retries).
      6. Optionally execute against the target DB.
      7. Cache the result and return.

    SOLID:
      S — Orchestrates; delegates all domain logic to specialised services.
      D — Depends on abstractions injected via constructor.
    """

    def __init__(
        self,
        retriever: SchemaRetriever,
        generator: SQLGeneratorService,
        validator: ISQLValidator,
        cache: ICache,
        max_retries: int = 3,
        db_client: object | None = None,  # AsyncDatabaseClient | None
        query_history: QueryHistoryService | None = None,
        query_classifier: QueryClassifier | None = None,
        session_service: object | None = None,  # ChatSessionService | None
        training_data_service: object | None = None,  # TrainingDataService | None
        table_selector: TableSelectorService | None = None,
        fk_extractor: object | None = None,  # FKRelationshipExtractor | None
        column_validator: SQLColumnValidator | None = None,
    ) -> None:
        self._retriever = retriever
        self._generator = generator
        self._validator = validator
        self._cache = cache
        self._max_retries = max_retries
        self._db_client = db_client
        self._query_history = query_history
        self._query_classifier = query_classifier
        self._session_service = session_service
        self._training_data_service = training_data_service
        self._table_selector = table_selector
        self._fk_extractor = fk_extractor
        self._column_validator = column_validator

    @trace_function("pipeline.run")
    async def run(self, request: QueryRequest, style_hints: dict | None = None, model_override: str | None = None, custom_instructions: str | None = None) -> QueryResponse:
        """Execute the full pipeline for a single query request.

        Args:
            request: Validated QueryRequest from the API layer.

        Returns:
            QueryResponse with SQL, validation info, and optional results.
        """
        start_time = time.time()
        log = logger.bind(question=request.question[:80])
        dialect = request.dialect or self._generator._dialect

        set_span_attribute("pipeline.question", request.question)
        set_span_attribute("pipeline.dialect", dialect)
        set_span_attribute("pipeline.execute", request.execute)
        set_span_attribute("pipeline.session_id", request.session_id)

        # Log execution request
        log.info(
            "Processing query request",
            execute=request.execute,
            session_id=request.session_id,
            dialect=dialect,
        )

        # ── Step 1: Cache lookup ──────────────────────────────────────────────
        # Only use cache when execute=False. When execute=True, we need to run the query.
        cached = None
        if hasattr(self._cache, "get_semantic"):
            cached = await self._cache.get_semantic(request.question)
            
        if not cached:
            cache_key = self._make_cache_key(request.question, dialect, self.PROMPT_VERSION)
            cached = await self._cache.get(cache_key)
            
        if cached and not request.execute:
            log.info("Cache hit — returning cached SQL")
            response_data = dict(cached)
            response_data["cached"] = True
            response = QueryResponse(**response_data)

            # Save cached response to chat session if session_id provided
            log.info(
                "Attempting to save cached response to chat session",
                session_service_available=self._session_service is not None,
                session_id=request.session_id,
            )
            if self._session_service is not None and request.session_id:
                try:
                    log.info("Saving cached message to chat session", session_id=request.session_id)
                    await self._session_service.add_message(
                        session_id=request.session_id,
                        question=request.question,
                        response=response,
                    )
                    log.info("Cached message saved successfully to chat session", session_id=request.session_id)
                except Exception as sess_exc:
                    log.warning("Failed to save cached response to chat session — skipping", error=str(sess_exc))

            return response
        elif cached and request.execute:
            log.info("Cache hit but execute=True — will execute the cached SQL")
            # Use cached SQL and execute it directly
            cached_sql = cached.get("sql", "")
            cached_tables = cached.get("retrieved_tables", [])
            cached_validation = cached.get("is_valid", False)

            # Execute the cached SQL
            execution_result: list[dict] | None = None
            execution_error: str | None = None
            if cached_validation and self._db_client:
                try:
                    log.info("Executing cached SQL", sql=cached_sql[:100])
                    execution_result = await self._db_client.execute_sql(cached_sql)  # type: ignore[union-attr]
                    log.info("Cached SQL execution successful", rows=len(execution_result) if execution_result else 0)
                except DatabaseExecutionError as exc:
                    log.warning("Cached SQL execution failed", error=str(exc))
                    execution_error = str(exc)
                except Exception as exc:
                    log.error("Unexpected error during cached SQL execution", error=str(exc))
                    execution_error = f"Unexpected execution error: {str(exc)}"
            elif request.execute:
                log.warning(
                    "Execution requested but skipped",
                    is_valid=cached_validation,
                    has_db_client=self._db_client is not None,
                )

            # Build response with execution results
            response = QueryResponse(
                question=request.question,
                sql=cached_sql,
                dialect=dialect,
                is_valid=cached_validation,
                validation_errors=cached.get("validation_errors", []),
                retrieved_tables=cached_tables,
                used_tables=cached.get("used_tables", []),
                execution_result=execution_result,
                execution_error=execution_error,
                tokens_used=cached.get("tokens_used", 0),
                cached=True,
            )

            # Save to chat session if session_id provided
            if self._session_service is not None and request.session_id:
                try:
                    log.info("Saving cached response with execution to chat session", session_id=request.session_id)
                    await self._session_service.add_message(
                        session_id=request.session_id,
                        question=request.question,
                        response=response,
                    )
                    log.info("Cached response with execution saved successfully", session_id=request.session_id)
                except Exception as sess_exc:
                    log.warning("Failed to save cached response to chat session — skipping", error=str(sess_exc))

            return response

        # ── Step 2: Query classification (greeting / off-topic detection) ───────
        intent_type = "database_query"  # Default intent
        if self._query_classifier is not None:
            classification = self._query_classifier.classify(request.question)
            if classification in ("greeting", "off_topic"):
                message = self._query_classifier.get_response_message(classification)
                log.info(f"Query classified as {classification} — returning early")
                response_time_ms = int((time.time() - start_time) * 1000)
                response = QueryResponse(
                    question=request.question,
                    sql="",
                    dialect=dialect,
                    is_valid=False,
                    validation_errors=[],
                    retrieved_tables=[],
                    used_tables=[],
                    execution_result=None,
                    tokens_used=0,
                    cached=False,
                    message=message,
                    intent_type=classification,
                    query_complexity=0,
                    prompt_version="v1.0",
                    retrieval_method="none",
                    response_time_ms=response_time_ms,
                )

                # Save to chat session if session_id provided
                log.info(
                    "Attempting to save greeting/off-topic message to chat session",
                    session_service_available=self._session_service is not None,
                    session_id=request.session_id,
                )
                if self._session_service is not None and request.session_id:
                    try:
                        await self._session_service.add_message(
                            session_id=request.session_id,
                            question=request.question,
                            response=response,
                        )
                        log.info("Greeting message saved successfully", session_id=request.session_id)
                    except Exception as sess_exc:
                        log.warning("Failed to save greeting to chat session — skipping", error=str(sess_exc))

                return response

        # ── Step 3: Schema retrieval — two-phase grounding ─────────────────────
        # Phase A: vector similarity search → candidate tables (coarse)
        log.info("Phase A: Retrieving candidate schema chunks via vector search")
        candidate_chunks = await self._retriever.retrieve(request.question)
        candidate_tables = list({c.table_name for c in candidate_chunks})
        log.info("Candidate tables from vector search", tables=candidate_tables)

        if self._table_selector is not None:
            # Phase B: LLM picks tables from the known ingested list (no hallucination)
            log.info("Phase B: Running LLM table selector")
            all_known_tables = await self._retriever.get_all_table_names()
            selected_tables = await self._table_selector.select_tables(
                question=request.question,
                available_tables=all_known_tables,
                fallback_tables=candidate_tables,
            )
            log.info("Tables selected by LLM", selected=selected_tables)

            # Phase C: fetch exact column definitions for selected tables
            log.info("Phase C: Fetching exact schema chunks for selected tables")
            grounded_chunks = await self._retriever.get_schema_for_tables(selected_tables)

            # Fallback: if grounding returned nothing, use candidate chunks
            final_chunks = grounded_chunks if grounded_chunks else candidate_chunks

            # Phase D: FK-Aware Expansion (NEW - Layer 1)
            if self._fk_extractor is not None and final_chunks:
                log.info("Phase D: Expanding schema via FK relationships")
                try:
                    final_chunks = await self._fk_extractor.expand_tables(
                        final_chunks,
                        max_expansion=3,
                    )
                except Exception as fk_exc:
                    log.warning("FK expansion failed, using grounded chunks", error=str(fk_exc))

            retrieved_tables = list({c.table_name for c in final_chunks})
            log.info(
                "Schema grounding complete",
                grounded=bool(grounded_chunks),
                final_tables=retrieved_tables,
                fk_expansion_applied=self._fk_extractor is not None,
            )
        else:
            # No table selector configured — use candidate chunks directly
            final_chunks = candidate_chunks
            retrieved_tables = candidate_tables

        schema_context = self._retriever.build_schema_context(final_chunks)

        # ── Steps 4-6: Generate + Validate with self-correction loop ──────────
        error_feedback: str | None = None
        generated_sql = None
        rate_limit_retry_count = 0
        max_rate_limit_retries = 2
        execution_result = None
        execution_error = None

        for attempt in range(1, self._max_retries + 1):
            try:
                log.info("Generating SQL", attempt=attempt)
                generated_sql = await self._generator.generate(
                    question=request.question,
                    schema_context=schema_context,
                    dialect_override=request.dialect,
                    error_feedback=error_feedback,
                    style_hints=style_hints,
                    model_override=model_override,
                    custom_instructions=custom_instructions,
                )
                generated_sql.attempt = attempt

                validation = self._validator.validate(generated_sql.cleaned_sql)
                generated_sql.validation = validation

                # Layer 2: Column-level validation (NEW)
                column_errors: list[str] = []
                if self._column_validator is not None:
                    # Extract schema from chunks for validation
                    schema_dict: dict[str, list[str]] = {}
                    for chunk in final_chunks:
                        # Parse column names from chunk content
                        # Format: "- column_name (type)"
                        import re
                        columns = re.findall(r'- (\w+)\s*\(', chunk.content)
                        schema_dict[chunk.table_name] = columns

                    column_errors = self._column_validator.validate(
                        generated_sql.cleaned_sql,
                        schema_dict,
                    )

                    if column_errors:
                        log.warning(
                            "Column validation failed",
                            errors=column_errors,
                        )

                # Combine validation errors
                all_errors = validation.errors + column_errors

                if validation.is_valid and not column_errors:
                    if request.execute and self._db_client:
                        try:
                            log.info("Executing generated SQL (Agentic validation)", sql=generated_sql.cleaned_sql[:100])
                            execution_result = await self._db_client.execute_sql(generated_sql.cleaned_sql)
                            log.info("SQL execution successful", rows=len(execution_result) if execution_result else 0)
                            break
                        except DatabaseExecutionError as exc:
                            execution_error = str(exc)
                            log.warning("Agentic SQL execution failed — retrying", error=execution_error)
                            all_errors.append(f"Database execution error: {execution_error}")
                            validation.is_valid = False
                    else:
                        log.info("SQL passed all validation (No execution requested)", attempt=attempt)
                        break


                log.warning(
                    "SQL failed validation — retrying",
                    attempt=attempt,
                    errors=all_errors,
                )
                error_feedback = "\n".join(all_errors)

            except RateLimitError as rate_exc:
                rate_limit_retry_count += 1
                if rate_limit_retry_count > max_rate_limit_retries:
                    log.warning(
                        "Rate limit exceeded — max retries reached",
                        retries=rate_limit_retry_count,
                        error=str(rate_exc),
                    )
                    raise

                # Wait and retry with exponential backoff
                wait_time = rate_exc.retry_after or (30 * rate_limit_retry_count)
                log.warning(
                    "Rate limit hit — waiting before retry",
                    retry_count=rate_limit_retry_count,
                    wait_seconds=wait_time,
                    max_retries=max_rate_limit_retries,
                )
                await asyncio.sleep(wait_time)
                # Decrement attempt to not count rate limit retries against validation retries
                attempt -= 1
                continue

        if generated_sql is None:
            raise SQLGenerationError("Orchestrator failed to produce SQL.")

        final_sql = generated_sql.validation.normalised_sql or generated_sql.cleaned_sql

        # ── Step 7: Optional execution ────────────────────────────────────────
        # Execution is now handled inside the validation loop.
        # If it failed on the final attempt, execution_error will be populated.
        # If execution was skipped (execute=False), both result and error will be None.
        if request.execute and not self._db_client:
            log.warning("Execution requested but skipped — no db_client available")
        elif request.execute and not generated_sql.validation.is_valid:
            log.warning("Execution failed or skipped due to invalid SQL")

        # ── Step 8: Build response and cache ──────────────────────────────────
        response_time_ms = int((time.time() - start_time) * 1000)

        # Determine query complexity based on SQL characteristics
        query_complexity = self._estimate_complexity(final_sql)

        response = QueryResponse(
            question=request.question,
            sql=final_sql,
            dialect=dialect,
            is_valid=generated_sql.validation.is_valid,
            validation_errors=generated_sql.validation.errors,
            retrieved_tables=retrieved_tables,
            used_tables=generated_sql.used_tables,
            execution_result=execution_result,
            execution_error=execution_error,
            tokens_used=generated_sql.tokens_used,
            cached=False,
            intent_type=intent_type,
            query_complexity=query_complexity,
            prompt_version="v1.0",
            retrieval_method="vector",
            response_time_ms=response_time_ms,
            suggested_chart=generated_sql.suggested_chart,
            follow_up_questions=generated_sql.follow_up_questions,
        )

        # Save to chat session if session_id provided (save regardless of validation status)
        log.info(
            "Attempting to save to chat session",
            session_service_available=self._session_service is not None,
            session_id=request.session_id,
        )
        if self._session_service is not None and request.session_id:
            try:
                log.info("Saving message to chat session", session_id=request.session_id)
                await self._session_service.add_message(
                    session_id=request.session_id,
                    question=request.question,
                    response=response,
                )
                log.info("Message saved successfully to chat session", session_id=request.session_id)
            except Exception as sess_exc:
                log.warning("Failed to save to chat session — skipping", error=str(sess_exc))

        if generated_sql.validation.is_valid:
            try:
                # Update exact cache
                cache_key = self._make_cache_key(request.question, dialect, self.PROMPT_VERSION)
                await self._cache.set(cache_key, response.model_dump())
                
                # Update semantic cache
                if hasattr(self._cache, "set_semantic"):
                    await self._cache.set_semantic(request.question, response.model_dump())
            except Exception as cache_exc:
                log.warning("Failed to cache response — skipping", error=str(cache_exc))

            # Collect training data for self-learning — non-critical
            if self._training_data_service is not None:
                try:
                    # Build schema context from retrieved tables
                    schema_context = f"Tables used: {', '.join(retrieved_tables)}" if retrieved_tables else ""

                    await self._training_data_service.collect_training_data(
                        question=request.question,
                        sql=final_sql,
                        retrieved_tables=retrieved_tables,
                        schema_context=schema_context,
                        intent_type=intent_type,
                        success_score=1.0 if not execution_error else 0.8,
                    )
                    log.debug("Training data collected successfully")
                except Exception as train_exc:
                    log.warning("Failed to collect training data — skipping", error=str(train_exc))

        return response

    # Bump this string whenever the prompt template changes to auto-invalidate old cache entries.
    PROMPT_VERSION = "v1.0"

    @staticmethod
    def _make_cache_key(question: str, dialect: str, prompt_version: str = "v1.0") -> str:
        """Deterministic cache key from question + dialect + prompt version.

        Including prompt_version means changing PROMPT_VERSION automatically
        invalidates all existing cache entries so stale SQL is never returned.
        """
        raw = json.dumps({"q": question.strip().lower(), "d": dialect.lower(), "pv": prompt_version})
        return f"nl2sql:{hashlib.sha256(raw.encode()).hexdigest()}"

    @staticmethod
    def _estimate_complexity(sql: str) -> int:
        """Estimate query complexity on a 1-10 scale based on SQL characteristics.

        Args:
            sql: The SQL query string.

        Returns:
            Complexity score from 1 (simple) to 10 (complex).
        """
        sql_upper = sql.upper()
        complexity = 1

        # Basic SELECT is complexity 1
        # Add complexity for various SQL features
        if "JOIN" in sql_upper:
            complexity += 2
        if "GROUP BY" in sql_upper:
            complexity += 1
        if "HAVING" in sql_upper:
            complexity += 1
        if "ORDER BY" in sql_upper:
            complexity += 1
        if "SUBQUERY" in sql_upper or sql_upper.count("SELECT") > 1:
            complexity += 2
        if "UNION" in sql_upper:
            complexity += 1
        if "CASE" in sql_upper:
            complexity += 1
        if "WINDOW" in sql_upper or "OVER(" in sql_upper:
            complexity += 2

        # Cap at 10
        return min(complexity, 10)

    async def run_stream(self, request: QueryRequest, style_hints: dict | None = None, model_override: str | None = None, custom_instructions: str | None = None):
        """Streaming version of run() — yields chunks as they're generated.

        Yields:
            Dict chunks with status and partial SQL generation.
        """
        import time

        start_time = time.time()
        log = logger.bind(question=request.question[:80])
        dialect = request.dialect or self._generator._dialect

        try:
            # Yield initial status
            yield {"status": "started", "stage": "initializing"}

            # Check cache first
            cached = None
            if hasattr(self._cache, "get_semantic"):
                cached = await self._cache.get_semantic(request.question)
            
            if not cached:
                cache_key = self._make_cache_key(request.question, dialect, self.PROMPT_VERSION)
                cached = await self._cache.get(cache_key)
                
            if cached:
                # Mark the payload itself as cached (the stored copy has cached=False),
                # and persist the cached answer to the chat session so it shows in history.
                response_data = dict(cached)
                response_data["cached"] = True
                if self._session_service is not None and request.session_id:
                    try:
                        cached_response = QueryResponse(**response_data)
                        await self._session_service.add_message(
                            session_id=request.session_id,
                            question=request.question,
                            response=cached_response,
                        )
                    except Exception as sess_exc:
                        log.warning("Failed to save cached response to chat session in stream", error=str(sess_exc))
                yield {"status": "complete", "cached": True, "data": response_data}
                return

            # ── Query classification (greeting / off-topic detection) ───────
            intent_type = "database_query"
            if self._query_classifier is not None:
                classification = self._query_classifier.classify(request.question)
                if classification in ("greeting", "off_topic"):
                    message = self._query_classifier.get_response_message(classification)
                    response_time_ms = int((time.time() - start_time) * 1000)
                    response = QueryResponse(
                        question=request.question,
                        sql="",
                        dialect=dialect,
                        is_valid=False,
                        validation_errors=[],
                        retrieved_tables=[],
                        used_tables=[],
                        execution_result=None,
                        tokens_used=0,
                        cached=False,
                        message=message,
                        intent_type=classification,
                        query_complexity=0,
                        prompt_version="v1.0",
                        retrieval_method="none",
                        response_time_ms=response_time_ms,
                    )
                    
                    if self._session_service is not None and request.session_id:
                        try:
                            await self._session_service.add_message(
                                session_id=request.session_id,
                                question=request.question,
                                response=response,
                            )
                        except Exception as sess_exc:
                            log.warning("Failed to save greeting to chat session", error=str(sess_exc))

                    yield {
                        "status": "complete",
                        "cached": False,
                        "data": response.model_dump(),
                        "response_time_ms": response_time_ms,
                    }
                    return

            # Schema retrieval — two-phase grounding
            yield {"status": "progress", "stage": "retrieving_schema"}
            candidate_chunks = await self._retriever.retrieve(request.question)
            candidate_tables = list({c.table_name for c in candidate_chunks})

            if self._table_selector is not None:
                all_known_tables = await self._retriever.get_all_table_names()
                selected_tables = await self._table_selector.select_tables(
                    question=request.question,
                    available_tables=all_known_tables,
                    fallback_tables=candidate_tables,
                )
                grounded_chunks = await self._retriever.get_schema_for_tables(selected_tables)
                final_chunks = grounded_chunks if grounded_chunks else candidate_chunks

                if self._fk_extractor is not None and final_chunks:
                    try:
                        final_chunks = await self._fk_extractor.expand_tables(
                            final_chunks,
                            max_expansion=3,
                        )
                    except Exception as fk_exc:
                        log.warning("FK expansion failed", error=str(fk_exc))

                retrieved_tables = list({c.table_name for c in final_chunks})
            else:
                final_chunks = candidate_chunks
                retrieved_tables = candidate_tables

            schema_context = self._retriever.build_schema_context(final_chunks)

            yield {
                "status": "progress",
                "stage": "schema_retrieved",
                "tables": retrieved_tables,
            }

            # Generate SQL with streaming and retry loop
            error_feedback: str | None = None
            generated_sql = None
            rate_limit_retry_count = 0
            max_rate_limit_retries = 2
            execution_result = None
            execution_error = None
            
            for attempt in range(1, self._max_retries + 1):
                try:
                    yield {"status": "progress", "stage": "generating_sql"}
                    generated_sql = await self._generator.generate(
                        question=request.question,
                        schema_context=schema_context,
                        dialect_override=request.dialect,
                        error_feedback=error_feedback,
                        style_hints=style_hints,
                        model_override=model_override,
                        custom_instructions=custom_instructions,
                    )
                    generated_sql.attempt = attempt

                    yield {
                        "status": "progress",
                        "stage": "sql_generated",
                        "sql": generated_sql.cleaned_sql,
                    }

                    # Validate
                    yield {"status": "progress", "stage": "validating_sql"}
                    validation = self._validator.validate(generated_sql.cleaned_sql)
                    generated_sql.validation = validation

                    # Column-level validation
                    column_errors: list[str] = []
                    if self._column_validator is not None:
                        schema_dict: dict[str, list[str]] = {}
                        for chunk in final_chunks:
                            import re
                            columns = re.findall(r'- (\w+)\s*\(', chunk.content)
                            schema_dict[chunk.table_name] = columns
                        column_errors = self._column_validator.validate(generated_sql.cleaned_sql, schema_dict)

                    all_errors = validation.errors + column_errors
                    
                    if validation.is_valid and not column_errors:
                        if request.execute and self._db_client:
                            yield {"status": "progress", "stage": "executing_sql"}
                            try:
                                log.info("Executing generated SQL in stream mode (Agentic)")
                                execution_result = await self._db_client.execute_sql(generated_sql.cleaned_sql)
                                break
                            except DatabaseExecutionError as exc:
                                execution_error = str(exc)
                                all_errors.append(f"Database execution error: {execution_error}")
                                validation.is_valid = False
                            except Exception as exc:
                                execution_error = f"Unexpected execution error: {str(exc)}"
                                all_errors.append(execution_error)
                                validation.is_valid = False
                        else:
                            break

                    error_feedback = "\n".join(all_errors)
                except RateLimitError as rate_exc:
                    rate_limit_retry_count += 1
                    if rate_limit_retry_count > max_rate_limit_retries:
                        raise
                    wait_time = rate_exc.retry_after or (30 * rate_limit_retry_count)
                    await asyncio.sleep(wait_time)
                    attempt -= 1
                    continue

            if generated_sql is None:
                raise SQLGenerationError("Orchestrator failed to produce SQL.")

            final_sql = generated_sql.validation.normalised_sql or generated_sql.cleaned_sql

            # Optional Execution handled in loop
            if request.execute and not self._db_client:
                log.warning("Execution requested but skipped in stream — no db_client")

            # Build response
            response_time_ms = int((time.time() - start_time) * 1000)
            query_complexity = self._estimate_complexity(final_sql)
            
            response = QueryResponse(
                question=request.question,
                sql=final_sql,
                dialect=dialect,
                is_valid=generated_sql.validation.is_valid,
                validation_errors=generated_sql.validation.errors,
                retrieved_tables=retrieved_tables,
                used_tables=generated_sql.used_tables,
                execution_result=execution_result,
                execution_error=execution_error,
                tokens_used=generated_sql.tokens_used,
                cached=False,
                intent_type=intent_type,
                query_complexity=query_complexity,
                prompt_version="v1.0",
                retrieval_method="vector",
                response_time_ms=response_time_ms,
                suggested_chart=generated_sql.suggested_chart,
                follow_up_questions=generated_sql.follow_up_questions,
            )

            # Save to chat session if session_id provided
            if self._session_service is not None and request.session_id:
                try:
                    await self._session_service.add_message(
                        session_id=request.session_id,
                        question=request.question,
                        response=response,
                    )
                except Exception as sess_exc:
                    log.warning("Failed to save to chat session in stream", error=str(sess_exc))

            # Cache if valid
            if validation.is_valid:
                try:
                    cache_key = self._make_cache_key(request.question, dialect, self.PROMPT_VERSION)
                    await self._cache.set(cache_key, response.model_dump())
                    if hasattr(self._cache, "set_semantic"):
                        await self._cache.set_semantic(request.question, response.model_dump())
                except Exception as cache_exc:
                    log.warning("Failed to cache response", error=str(cache_exc))

            yield {
                "status": "complete",
                "cached": False,
                "data": response.model_dump(),
                "response_time_ms": response_time_ms,
            }

        except Exception as exc:
            log.error("Streaming query failed", error=str(exc))
            yield {"status": "error", "error": str(exc), "type": type(exc).__name__}
