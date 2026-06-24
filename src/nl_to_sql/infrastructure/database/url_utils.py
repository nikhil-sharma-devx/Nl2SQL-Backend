"""Database URL helpers.

Ensures URLs handed to ``create_async_engine`` use an async-capable driver.
Also parses ADO.NET / key-value connection strings into standard URIs and
URL-encodes special characters in passwords.
"""
from __future__ import annotations

import re
from urllib.parse import quote

# Driver tokens that are already async-capable -> leave the URL as-is.
_ASYNC_DRIVERS = ("+asyncpg", "+aiomysql", "+asyncmy")

# Keys recognised in ADO.NET / libpq key=value connection strings.
_KV_ALIASES: dict[str, str] = {
    "host":                 "host",
    "server":               "host",
    "data source":          "host",
    "port":                 "port",
    "database":             "database",
    "initial catalog":      "database",
    "dbname":               "database",
    "user":                 "user",
    "user id":              "user",
    "uid":                  "user",
    "username":             "user",
    "password":             "password",
    "pwd":                  "password",
}


def _parse_kv_connection_string(raw: str) -> str | None:
    """Parse a key=value connection string (ADO.NET / libpq style) into a URI.

    Returns a postgresql+asyncpg:// URI, or None if the input doesn't look
    like a key=value connection string.
    """
    # Must contain at least one 'Key=Value;' segment and no '://'
    if "://" in raw or "=" not in raw:
        return None

    parts: dict[str, str] = {}
    for segment in raw.split(";"):
        segment = segment.strip()
        if not segment:
            continue
        if "=" not in segment:
            return None  # not a valid kv format
        key, _, value = segment.partition("=")
        canonical = _KV_ALIASES.get(key.strip().lower())
        if canonical:
            parts[canonical] = value.strip()

    if not parts.get("host"):
        return None

    user = parts.get("user", "")
    password = parts.get("password", "")
    host = parts.get("host", "")
    port = parts.get("port", "5432")
    database = parts.get("database", "")

    # URL-encode user and password so special chars (@, $, #, %, etc.) are safe
    user_enc = quote(user, safe="")
    pass_enc = quote(password, safe="")

    userinfo = f"{user_enc}:{pass_enc}" if user_enc else ""
    hostport = f"{host}:{port}" if port else host
    authority = f"{userinfo}@{hostport}" if userinfo else hostport

    return f"postgresql+asyncpg://{authority}/{database}"


def _encode_password_in_uri(url: str) -> str:
    """Re-encode the password component of a URI if it contains unescaped special chars.

    Characters like @, $, #, %, space are valid in passwords but break URI
    parsing when unescaped. This function detects and fixes them.
    """
    # Find the scheme+authority part: scheme://[user:pass@]host/...
    match = re.match(
        r"^((?:postgresql|postgres|mysql|sqlite)(?:\+\w+)?://)"  # scheme
        r"([^:@/]+)"                                              # user (no colon, @, /)
        r"(?::([^@]*))?@"                                         # :password (optional)
        r"(.+)$",                                                  # rest
        url,
        re.IGNORECASE,
    )
    if not match:
        return url

    scheme, user, password, rest = match.groups()
    if password is None:
        return url

    # If the password already contains %XX sequences it's already encoded — leave it.
    if re.search(r"%[0-9A-Fa-f]{2}", password):
        return url

    encoded_password = quote(password, safe="")
    return f"{scheme}{quote(user, safe='')}:{encoded_password}@{rest}"


def to_async_database_url(url: str) -> str:
    """Normalise any supported connection string to an asyncpg/aiomysql URI.

    Accepted input formats:
      - Standard URI:              postgresql://user:pass@host:5432/db
      - asyncpg URI (pass-through): postgresql+asyncpg://...
      - postgres:// shorthand:     postgres://user:pass@host/db
      - ADO.NET key-value:         Host=...;Port=...;Database=...;Username=...;Password=...
    """
    if not url:
        return url

    url = url.strip()

    # ── Step 1: convert ADO.NET / key-value format ────────────────────────────
    kv_uri = _parse_kv_connection_string(url)
    if kv_uri is not None:
        url = kv_uri

    lowered = url.lower()

    # ── Step 2: normalise scheme to async driver ──────────────────────────────
    if lowered.startswith(("postgresql+psycopg2://", "postgresql://", "postgres://")):
        normalized = "postgresql+asyncpg://" + url.split("://", 1)[1]
    elif lowered.startswith(("mysql+pymysql://", "mysql://")):
        normalized = "mysql+aiomysql://" + url.split("://", 1)[1]
    elif any(driver in lowered for driver in _ASYNC_DRIVERS):
        normalized = url
    else:
        normalized = url

    # ── Step 3: encode special characters in the password ────────────────────
    normalized = _encode_password_in_uri(normalized)

    # ── Step 4: asyncpg-specific query-param cleanups ─────────────────────────
    if "asyncpg" in normalized:
        normalized = re.sub(r"([?&])channel_binding=[^&]*", r"\1", normalized)
        normalized = normalized.rstrip("?&")
        if "sslmode=" in normalized:
            normalized = normalized.replace("sslmode=", "ssl=")

    return normalized
