"""Unit tests for the token-bucket store and AuthThrottleService policies."""

import pytest

from nl_to_sql.core.exceptions import AuthRateLimitError
from nl_to_sql.core.interfaces.i_rate_limiter import BucketPolicy
from nl_to_sql.infrastructure.rate_limit.in_memory_bucket import InMemoryTokenBucket
from nl_to_sql.services.auth_throttle_service import (
    LOGIN_FAIL_ACCOUNT,
    LOGIN_FAIL_IP,
    OTP_SEND_ACCOUNT,
    OTP_SEND_COOLDOWN,
    OTP_VERIFY_ACCOUNT,
    AuthThrottleService,
)


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def bucket(clock: FakeClock) -> InMemoryTokenBucket:
    return InMemoryTokenBucket(clock=clock)


@pytest.fixture
def throttle(bucket: InMemoryTokenBucket) -> AuthThrottleService:
    return AuthThrottleService(rate_limiter=bucket, enabled=True)


POLICY = BucketPolicy("t", capacity=3, refill_seconds=10)


class TestInMemoryTokenBucket:
    async def test_burst_then_deny_with_retry_after(self, bucket) -> None:
        for _ in range(3):
            assert (await bucket.consume("k", POLICY)).allowed
        denied = await bucket.consume("k", POLICY)
        assert not denied.allowed
        assert denied.retry_after == 10

    async def test_refills_over_time(self, bucket, clock) -> None:
        for _ in range(3):
            await bucket.consume("k", POLICY)
        clock.now += 10
        assert (await bucket.consume("k", POLICY)).allowed
        assert not (await bucket.consume("k", POLICY)).allowed

    async def test_peek_does_not_spend(self, bucket) -> None:
        for _ in range(5):
            assert (await bucket.consume("k", POLICY, cost=0)).allowed
        for _ in range(3):
            assert (await bucket.consume("k", POLICY)).allowed

    async def test_reset_refills(self, bucket) -> None:
        for _ in range(3):
            await bucket.consume("k", POLICY)
        await bucket.reset("k")
        assert (await bucket.consume("k", POLICY)).allowed

    async def test_keys_isolated(self, bucket) -> None:
        for _ in range(3):
            await bucket.consume("a", POLICY)
        assert (await bucket.consume("b", POLICY)).allowed


class TestLoginThrottle:
    async def test_account_locked_after_failures_regardless_of_ip(self, throttle) -> None:
        for i in range(LOGIN_FAIL_ACCOUNT.capacity):
            await throttle.check_login_allowed("Victim@x.com", f"10.0.0.{i}")
            await throttle.record_login_failure("victim@x.com", f"10.0.0.{i}")
        with pytest.raises(AuthRateLimitError) as ei:
            await throttle.check_login_allowed("victim@x.com", "10.9.9.9")
        assert ei.value.retry_after == LOGIN_FAIL_ACCOUNT.refill_seconds

    async def test_unlocks_after_refill(self, throttle, clock) -> None:
        for _ in range(LOGIN_FAIL_ACCOUNT.capacity):
            await throttle.record_login_failure("v@x.com", "1.1.1.1")
        clock.now += LOGIN_FAIL_ACCOUNT.refill_seconds
        await throttle.check_login_allowed("v@x.com", "1.1.1.1")

    async def test_success_resets_account_bucket(self, throttle) -> None:
        for _ in range(LOGIN_FAIL_ACCOUNT.capacity - 1):
            await throttle.record_login_failure("v@x.com", "1.1.1.1")
        await throttle.record_login_success("v@x.com")
        for _ in range(LOGIN_FAIL_ACCOUNT.capacity - 1):
            await throttle.record_login_failure("v@x.com", "1.1.1.1")
        await throttle.check_login_allowed("v@x.com", "1.1.1.1")

    async def test_ip_blocked_when_spraying_many_accounts(self, throttle) -> None:
        for i in range(LOGIN_FAIL_IP.capacity):
            await throttle.record_login_failure(f"user{i}@x.com", "6.6.6.6")
        with pytest.raises(AuthRateLimitError):
            await throttle.check_login_allowed("fresh@x.com", "6.6.6.6")
        await throttle.check_login_allowed("fresh@x.com", "7.7.7.7")


class TestOtpThrottle:
    async def test_cooldown_between_sends(self, throttle, clock) -> None:
        await throttle.enforce_otp_send("a@x.com", "1.1.1.1")
        with pytest.raises(AuthRateLimitError):
            await throttle.enforce_otp_send("a@x.com", "2.2.2.2")
        clock.now += OTP_SEND_COOLDOWN.refill_seconds
        await throttle.enforce_otp_send("a@x.com", "2.2.2.2")

    async def test_account_send_cap(self, throttle, clock) -> None:
        for _ in range(OTP_SEND_ACCOUNT.capacity):
            assert (await throttle.try_otp_send_account("a@x.com")).allowed
            clock.now += OTP_SEND_COOLDOWN.refill_seconds
        assert not (await throttle.try_otp_send_account("a@x.com")).allowed

    async def test_verify_budget_survives_resend(self, throttle) -> None:
        for _ in range(OTP_VERIFY_ACCOUNT.capacity):
            await throttle.enforce_otp_verify("a@x.com")
        with pytest.raises(AuthRateLimitError):
            await throttle.enforce_otp_verify("a@x.com")


async def test_disabled_flag_is_consumed(bucket) -> None:
    throttle = AuthThrottleService(rate_limiter=bucket, enabled=False)
    for _ in range(50):
        await throttle.record_login_failure("a@x.com", "1.1.1.1")
        await throttle.enforce_otp_verify("a@x.com")
    await throttle.check_login_allowed("a@x.com", "1.1.1.1")
