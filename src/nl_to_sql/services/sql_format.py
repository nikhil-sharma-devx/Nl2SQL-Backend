"""Deterministic SQL formatting pass using sqlglot.

Applied AFTER generation to enforce user SQL style preferences.
Never alters SQL semantics — formatting only.

Usage:
    from nl_to_sql.services.sql_format import format_sql
    formatted = format_sql(sql, dialect="postgres", keyword_case="upper", indent=2)
"""
from __future__ import annotations

import structlog

logger = structlog.get_logger(__name__)


def format_sql(
    sql: str,
    *,
    dialect: str | None = None,
    keyword_case: str = "upper",
    indent: int = 2,
    alias_style: str = "as",
) -> str:
    """Parse SQL with sqlglot and reformat per style preferences.

    Falls back to the original SQL string on any parse error so generation
    is never blocked by a formatting failure.
    """
    try:
        import sqlglot

        parsed = sqlglot.parse_one(sql, dialect=dialect or "")
        formatted = parsed.sql(
            dialect=dialect or "",
            pretty=True,
            indent=indent,
        )
        formatted = _apply_keyword_case(formatted, upper=(keyword_case == "upper"))
        formatted = _apply_alias_style(formatted, alias_style)
        return formatted
    except Exception as exc:
        logger.warning("sql_format: formatting failed, returning original", error=str(exc))
        return sql


def _apply_alias_style(sql: str, alias_style: str) -> str:
    """Remove the AS keyword from alias positions when alias_style == 'implicit'.

    Tracks parenthesis depth so that CAST(x AS type) is never touched — AS
    inside parentheses is always a type-cast, not a column/table alias.
    """
    if alias_style != "implicit":
        return sql

    result: list[str] = []
    i = 0
    depth = 0
    n = len(sql)

    while i < n:
        c = sql[i]
        if c == '(':
            depth += 1
            result.append(c)
            i += 1
        elif c == ')':
            depth -= 1
            result.append(c)
            i += 1
        elif depth == 0 and sql[i: i + 4].upper() == ' AS ':
            # Only strip AS when what follows looks like an alias identifier
            after = sql[i + 4:]
            if after and (after[0].isalpha() or after[0] in '_"$`['):
                result.append(' ')   # keep the leading space, drop " AS "
                i += 4
            else:
                result.append(c)
                i += 1
        else:
            result.append(c)
            i += 1

    return ''.join(result)


def _apply_keyword_case(sql: str, *, upper: bool) -> str:
    """Case SQL keywords using sqlglot's tokenizer, preserving string literals."""
    try:
        import sqlglot
        from sqlglot.tokens import TokenType

        _KEYWORD_TYPES = frozenset({
            TokenType.SELECT, TokenType.FROM, TokenType.WHERE,
            TokenType.JOIN, TokenType.LEFT, TokenType.RIGHT, TokenType.INNER,
            TokenType.OUTER, TokenType.FULL, TokenType.CROSS, TokenType.ON,
            TokenType.AND, TokenType.OR, TokenType.NOT, TokenType.IN, TokenType.IS,
            TokenType.NULL, TokenType.AS, TokenType.GROUP, TokenType.BY,
            TokenType.ORDER, TokenType.HAVING, TokenType.LIMIT, TokenType.OFFSET,
            TokenType.WITH, TokenType.UNION, TokenType.INTERSECT, TokenType.EXCEPT,
            TokenType.DISTINCT, TokenType.ALL, TokenType.CASE, TokenType.WHEN,
            TokenType.THEN, TokenType.ELSE, TokenType.END, TokenType.BETWEEN,
            TokenType.LIKE, TokenType.EXISTS, TokenType.INSERT, TokenType.INTO,
            TokenType.VALUES, TokenType.UPDATE, TokenType.SET, TokenType.DELETE,
            TokenType.ASC, TokenType.DESC,
        })

        tokens = sqlglot.tokenize(sql)
        parts: list[str] = []
        prev_end = 0

        for token in tokens:
            start = token.start
            end = token.end + 1
            parts.append(sql[prev_end:start])
            text = sql[start:end]
            if token.token_type in _KEYWORD_TYPES:
                text = text.upper() if upper else text.lower()
            parts.append(text)
            prev_end = end

        parts.append(sql[prev_end:])
        return "".join(parts)

    except Exception:
        return sql.upper() if upper else sql.lower()
