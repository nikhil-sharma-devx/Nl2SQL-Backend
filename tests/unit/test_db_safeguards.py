from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from nl_to_sql.infrastructure.database.sqlalchemy_client import AsyncDatabaseClient


@pytest.mark.asyncio
async def test_switch_engine_disposes_old_pool() -> None:
    """Verify switch_engine on AsyncDatabaseClient disposes the old engine connection pool."""
    initial_url = "postgresql://user:pass@localhost:5432/first_db"

    patch_target = "nl_to_sql.infrastructure.database.sqlalchemy_client.create_async_engine"
    with patch(patch_target) as mock_create_engine:
        mock_old_engine = MagicMock()
        mock_old_engine.dispose = AsyncMock()

        mock_new_engine = MagicMock()
        mock_new_engine.dispose = AsyncMock()

        # Side effect returns old engine first, then new engine on subsequent calls
        mock_create_engine.side_effect = [mock_old_engine, mock_new_engine]

        client = AsyncDatabaseClient(database_url=initial_url)
        assert client._engine == mock_old_engine

        # Swap to a new database
        new_url = "postgresql://user2:pass2@localhost:5432/second_db"
        await client.switch_engine(new_url=new_url)

        # Assert engine is updated to the new mock engine
        assert client._engine == mock_new_engine

        # Verify old engine connection pool was disposed to prevent connection leaks
        mock_old_engine.dispose.assert_awaited_once()
        mock_new_engine.dispose.assert_not_called()
