"""Chat session service — manages chat sessions and messages."""
from datetime import date, datetime
from decimal import Decimal
from typing import Any
from uuid import uuid4

import sqlalchemy
import structlog
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import selectinload

from nl_to_sql.core.models.query import QueryResponse
from nl_to_sql.infrastructure.database.models import ChatMessage, ChatSession
from nl_to_sql.infrastructure.database.url_utils import to_async_database_url

logger = structlog.get_logger(__name__)


def make_json_serializable(obj: Any) -> Any:
    """Convert an object to be JSON serializable.

    Handles Decimal, datetime, date, and other non-serializable types.
    """
    if isinstance(obj, Decimal):
        return float(obj)
    elif isinstance(obj, (datetime, date)):
        return obj.isoformat()
    elif isinstance(obj, dict):
        return {key: make_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [make_json_serializable(item) for item in obj]
    return obj


class SessionInfo:
    """Lightweight session info for listing."""
    def __init__(self, id: str, title: str, created_at: datetime, updated_at: datetime, message_count: int) -> None:
        self.id = id
        self.title = title
        self.created_at = created_at
        self.updated_at = updated_at
        self.message_count = message_count


class ChatSessionService:
    """Manages chat sessions and their messages.

    Provides CRUD operations for chat sessions and message storage.
    """

    def __init__(self, database_url: str) -> None:
        """Initialize the session service with database connection.

        Args:
            database_url: SQLAlchemy async database URL for session storage.
        """
        database_url = to_async_database_url(database_url)
        self._engine = create_async_engine(
            database_url,
            echo=False,
            pool_pre_ping=True,
            pool_size=2,
            max_overflow=1,
            pool_timeout=30,
            pool_recycle=300,
            connect_args={"prepared_statement_cache_size": 0},
        )
        self._session_factory = async_sessionmaker(
            bind=self._engine,
            expire_on_commit=False,
            class_=AsyncSession,
        )
        self._logger = logger.bind(service="ChatSession")

    async def initialize(self) -> None:
        """Create the chat session and message tables if they don't exist."""
        # Schema is initialized once globally via query_history.initialize() on startup
        self._logger.info("Chat session database initialized (schema checked globally)")

    async def create_session(self, title: str = "New Chat", user_id: str | None = None) -> ChatSession:
        """Create a new chat session.

        Args:
            title: Optional title for the session.
            user_id: Optional user ID to associate with the session.

        Returns:
            Created ChatSession object.
        """
        session_id = str(uuid4())
        now = datetime.utcnow()

        async with self._session_factory() as session:
            chat_session = ChatSession(
                id=session_id,
                user_id=user_id,
                title=title,
                created_at=now,
                updated_at=now,
            )
            session.add(chat_session)
            await session.commit()
            await session.refresh(chat_session)

        self._logger.info("Chat session created", session_id=session_id, title=title, user_id=user_id)
        return chat_session

    async def get_session(self, session_id: str) -> ChatSession | None:
        """Get a chat session by ID with all its messages.

        Args:
            session_id: The session UUID.

        Returns:
            ChatSession with messages or None if not found.
        """
        async with self._session_factory() as session:
            result = await session.execute(
                select(ChatSession)
                .options(selectinload(ChatSession.messages))
                .where(ChatSession.id == session_id)
            )
            return result.scalar_one_or_none()

    async def list_sessions(self, limit: int = 50, offset: int = 0, user_id: str | None = None) -> list[SessionInfo]:
        """List chat sessions ordered by most recently updated.

        Args:
            limit: Maximum number of sessions to return.
            offset: Number of sessions to skip.
            user_id: If provided, only return sessions belonging to this user.

        Returns:
            List of SessionInfo objects.
        """
        async with self._session_factory() as session:
            # Correlated subquery: counts messages in one DB round-trip instead of N+1
            msg_count = (
                select(sqlalchemy.func.count(ChatMessage.id))
                .where(ChatMessage.session_id == ChatSession.id)
                .correlate(ChatSession)
                .scalar_subquery()
            )
            query = (
                select(
                    ChatSession.id,
                    ChatSession.title,
                    ChatSession.created_at,
                    ChatSession.updated_at,
                    msg_count.label("message_count"),
                )
                .order_by(ChatSession.updated_at.desc())
                .limit(limit)
                .offset(offset)
            )
            if user_id is not None:
                query = query.where(ChatSession.user_id == user_id)

            result = await session.execute(query)
            return [
                SessionInfo(
                    id=row.id,
                    title=row.title,
                    created_at=row.created_at,
                    updated_at=row.updated_at,
                    message_count=row.message_count,
                )
                for row in result.all()
            ]

    async def count_sessions(self, user_id: str | None = None) -> int:
        """Return total count of sessions, optionally filtered by user."""
        async with self._session_factory() as session:
            query = select(sqlalchemy.func.count()).select_from(ChatSession)
            if user_id is not None:
                query = query.where(ChatSession.user_id == user_id)
            result = await session.execute(query)
            return result.scalar() or 0

    async def list_all_messages(self, limit: int = 50, offset: int = 0) -> list[ChatMessage]:
        """Return all chat messages paginated, newest first (global across all sessions)."""
        async with self._session_factory() as session:
            result = await session.execute(
                select(ChatMessage)
                .order_by(ChatMessage.timestamp.desc())
                .limit(limit)
                .offset(offset)
            )
            return list(result.scalars().all())

    async def count_all_messages(self) -> int:
        """Return total count of all chat messages."""
        async with self._session_factory() as session:
            result = await session.execute(
                select(sqlalchemy.func.count()).select_from(ChatMessage)
            )
            return result.scalar() or 0

    async def add_message(self, session_id: str, question: str, response: QueryResponse) -> ChatMessage:
        """Add a message to an existing chat session.

        Args:
            session_id: The session UUID.
            question: User's question.
            response: AI's QueryResponse.

        Returns:
            Created ChatMessage object.
        """
        self._logger.info(
            "Adding message to session",
            session_id=session_id,
            question=question[:50],
        )
        async with self._session_factory() as session:
            # Update session's updated_at timestamp
            session_result = await session.execute(
                select(ChatSession).where(ChatSession.id == session_id)
            )
            chat_session = session_result.scalar_one_or_none()
            if not chat_session:
                self._logger.error("Session not found when adding message", session_id=session_id)
                raise ValueError(f"Session {session_id} not found")

            self._logger.info("Session found, updating and adding message", session_id=session_id)
            chat_session.updated_at = datetime.utcnow()

            # Update title from first question if it's still default
            if chat_session.title == "New Chat":
                chat_session.title = question[:50] + ("..." if len(question) > 50 else "")

            # Create message
            message = ChatMessage(
                session_id=session_id,
                question=question,
                sql=response.sql,
                dialect=response.dialect,
                is_valid=response.is_valid,
                validation_errors=response.validation_errors,
                retrieved_tables=response.retrieved_tables,
                execution_result=make_json_serializable(response.execution_result),
                execution_error=response.execution_error,
                tokens_used=response.tokens_used,
                cached=response.cached,
                message=response.message,
                # Premium response fields
                used_tables=list(response.used_tables) if response.used_tables else [],
                suggested_chart=make_json_serializable(response.suggested_chart) if response.suggested_chart else None,
                follow_up_questions=list(response.follow_up_questions) if response.follow_up_questions else [],
                # Analytics fields
                intent_type=getattr(response, 'intent_type', None),
                query_complexity=getattr(response, 'query_complexity', None),
                prompt_version=getattr(response, 'prompt_version', None),
                retrieval_method=getattr(response, 'retrieval_method', None),
                response_time_ms=getattr(response, 'response_time_ms', None),
            )
            session.add(message)
            await session.commit()
            await session.refresh(message)

        self._logger.info(
            "Message added to session successfully",
            session_id=session_id,
            message_id=message.id,
        )
        return message

    async def delete_session(self, session_id: str) -> None:
        """Delete a chat session and all its messages.

        Args:
            session_id: The session UUID to delete.
        """
        async with self._session_factory() as session:
            await session.execute(
                delete(ChatSession).where(ChatSession.id == session_id)
            )
            await session.commit()

        self._logger.info("Chat session deleted", session_id=session_id)

    async def delete_all_sessions(self, user_id: str | None = None) -> None:
        """Delete chat sessions (optionally scoped to a user).

        Args:
            user_id: If provided, only delete sessions belonging to this user.
                     If None, deletes ALL sessions (admin use).
        """
        async with self._session_factory() as session:
            if user_id is not None:
                # Fetch session IDs for this user, then delete their messages first
                result = await session.execute(
                    select(ChatSession.id).where(ChatSession.user_id == user_id)
                )
                session_ids = [row[0] for row in result.all()]
                if session_ids:
                    await session.execute(
                        delete(ChatMessage).where(ChatMessage.session_id.in_(session_ids))
                    )
                    await session.execute(
                        delete(ChatSession).where(ChatSession.id.in_(session_ids))
                    )
            else:
                await session.execute(delete(ChatMessage))
                await session.execute(delete(ChatSession))
            await session.commit()

        self._logger.info("Chat sessions deleted", user_id=user_id)

    async def dispose(self) -> None:
        """Close all database connections."""
        await self._engine.dispose()
        self._logger.info("Chat session database connections closed")
