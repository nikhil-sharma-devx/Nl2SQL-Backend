"""AuthThrottleService — brute-force protection for credential endpoints.

Why token buckets on top of SlowAPI's per-IP limits: per-IP limits alone are
defeated by IP rotation (botnets, proxies), and an attacker could cycle
``resend-otp`` → 5 guesses → ``resend-otp`` to brute-force a 6-digit OTP. So
every sensitive action is also bucketed **per account** (keyed on the
submitted email, existing or not — no enumeration signal) and failures are
additionally bucketed **per IP** to catch credential stuffing across many
accounts from one source.

Policies (capacity = burst; refill = seconds to regain one token):

==================  ==========  ========  =====================================
Bucket              capacity    refill    Effect
==================  ==========  ========  =====================================
login fail / acct   5           180 s     5 wrong passwords, then 1 per 3 min
login fail / IP     20          90 s      20 failures, then ~40/h per IP
otp send cooldown   1           60 s      at most one email per minute per acct
otp send / acct     3           600 s     3 emails, then 6/h per account
otp send / IP       10          360 s     10 emails, then 10/h per IP
otp verify / acct   10          120 s     10 code guesses, then 30/h per acct
==================  ==========  ========  =====================================

Successful login resets the account's failure bucket (not the IP bucket).
Keys hash the email so no PII lands in Redis.
"""

import hashlib

import structlog

from nl_to_sql.core.exceptions import AuthRateLimitError
from nl_to_sql.core.interfaces.i_rate_limiter import BucketPolicy, IRateLimiter, RateDecision

logger = structlog.get_logger(__name__)

LOGIN_FAIL_ACCOUNT = BucketPolicy("login_fail_acct", capacity=5, refill_seconds=180)
LOGIN_FAIL_IP = BucketPolicy("login_fail_ip", capacity=20, refill_seconds=90)
OTP_SEND_COOLDOWN = BucketPolicy("otp_send_cooldown", capacity=1, refill_seconds=60)
OTP_SEND_ACCOUNT = BucketPolicy("otp_send_acct", capacity=3, refill_seconds=600)
OTP_SEND_IP = BucketPolicy("otp_send_ip", capacity=10, refill_seconds=360)
OTP_VERIFY_ACCOUNT = BucketPolicy("otp_verify_acct", capacity=10, refill_seconds=120)

_ALLOWED = RateDecision(allowed=True, remaining=0.0, retry_after=0)


def _email_id(email: str) -> str:
    return hashlib.sha256(email.strip().lower().encode()).hexdigest()[:32]


def _key(policy: BucketPolicy, subject: str) -> str:
    return f"auth:{policy.name}:{subject}"


class AuthThrottleService:
    """Applies the auth bucket policies; raises AuthRateLimitError when exhausted.

    SOLID: S — owns only auth throttling policy; D — depends on IRateLimiter.
    """

    def __init__(self, rate_limiter: IRateLimiter, enabled: bool = True) -> None:
        self._limiter = rate_limiter
        self._enabled = enabled

    # ── helpers ──────────────────────────────────────────────────────────────

    async def _consume(self, policy: BucketPolicy, subject: str, cost: int = 1) -> RateDecision:
        if not self._enabled:
            return _ALLOWED
        return await self._limiter.consume(_key(policy, subject), policy, cost)

    @staticmethod
    def _deny(decision: RateDecision, message: str, **log_kw: str) -> AuthRateLimitError:
        logger.warning("Auth throttle tripped", retry_after=decision.retry_after, **log_kw)
        return AuthRateLimitError(message, retry_after=decision.retry_after)

    # ── login ────────────────────────────────────────────────────────────────

    async def check_login_allowed(self, email: str, ip: str) -> None:
        """Peek (no token spent) — refuse before even checking the password."""
        acct = await self._consume(LOGIN_FAIL_ACCOUNT, _email_id(email), cost=0)
        if not acct.allowed:
            raise self._deny(
                acct,
                "Too many failed login attempts for this account. Try again later.",
                bucket=LOGIN_FAIL_ACCOUNT.name,
            )
        by_ip = await self._consume(LOGIN_FAIL_IP, ip, cost=0)
        if not by_ip.allowed:
            raise self._deny(
                by_ip,
                "Too many failed login attempts. Try again later.",
                bucket=LOGIN_FAIL_IP.name,
                ip=ip,
            )

    async def record_login_failure(self, email: str, ip: str) -> None:
        await self._consume(LOGIN_FAIL_ACCOUNT, _email_id(email))
        await self._consume(LOGIN_FAIL_IP, ip)

    async def record_login_success(self, email: str) -> None:
        if self._enabled:
            await self._limiter.reset(_key(LOGIN_FAIL_ACCOUNT, _email_id(email)))

    # ── OTP / reset-link emails ──────────────────────────────────────────────

    async def check_otp_send_ip(self, ip: str) -> None:
        """Per-IP send cap — raises (safe: reveals nothing about any account)."""
        decision = await self._consume(OTP_SEND_IP, ip)
        if not decision.allowed:
            raise self._deny(
                decision,
                "Too many verification emails requested. Try again later.",
                bucket=OTP_SEND_IP.name,
                ip=ip,
            )

    async def try_otp_send_account(self, email: str) -> RateDecision:
        """Per-account cooldown + cap. Returns the decision instead of raising,
        so enumeration-safe callers (forgot-password) can silently skip."""
        subject = _email_id(email)
        cooldown = await self._consume(OTP_SEND_COOLDOWN, subject)
        if not cooldown.allowed:
            return cooldown
        return await self._consume(OTP_SEND_ACCOUNT, subject)

    async def enforce_otp_send(self, email: str, ip: str) -> None:
        """IP cap + account cap, raising on either (register / resend-otp)."""
        await self.check_otp_send_ip(ip)
        decision = await self.try_otp_send_account(email)
        if not decision.allowed:
            raise self._deny(
                decision,
                "Please wait before requesting another code.",
                bucket="otp_send_acct",
            )

    # ── OTP verification (verify-otp, reset-password) ────────────────────────

    async def enforce_otp_verify(self, email: str) -> None:
        decision = await self._consume(OTP_VERIFY_ACCOUNT, _email_id(email))
        if not decision.allowed:
            raise self._deny(
                decision,
                "Too many verification attempts. Try again later.",
                bucket=OTP_VERIFY_ACCOUNT.name,
            )

    async def health_check(self) -> bool:
        return await self._limiter.health_check()
