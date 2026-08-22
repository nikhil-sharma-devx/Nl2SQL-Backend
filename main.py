"""Application entry point — starts the Uvicorn server."""
import uvicorn

from nl_to_sql.config.settings import get_settings


def main() -> None:
    """Run the FastAPI application with Uvicorn."""
    settings = get_settings()
    uvicorn.run(
        "nl_to_sql.api.app:create_app",
        factory=True,
        host=settings.app_host,
        port=settings.app_port,
        reload=settings.app_env == "development",
        log_level=settings.app_log_level.lower(),
        workers=1 if settings.app_env == "development" else 4,
        log_config=None,
    )


if __name__ == "__main__":
    main()
