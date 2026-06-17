"""Prompt Manager — Manages prompt versions for A/B testing."""
import hashlib
import random
import time
from dataclasses import dataclass, field
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


@dataclass
class PromptTemplate:
    """A versioned prompt template."""

    version: str
    name: str
    template: str
    is_active: bool = True
    success_count: int = 0
    failure_count: int = 0
    created_at: float = field(default_factory=time.time)


class PromptManager:
    """Manages multiple prompt templates with A/B testing support.

    Features:
    - Store multiple prompt versions
    - A/B testing: randomly assign versions
    - Track success/failure rates per version
    - Analytics: compare version performance

    SOLID:
      S — Only handles prompt versioning and testing
      O — Can be extended with different selection strategies
    """

    def __init__(self, ab_testing_enabled: bool = True) -> None:
        self._templates: dict[str, PromptTemplate] = {}
        self._ab_testing_enabled = ab_testing_enabled
        self._logger = logger.bind(component="PromptManager")

    def register_template(
        self,
        version: str,
        name: str,
        template: str,
        is_active: bool = True,
    ) -> None:
        """Register a new prompt template version.

        Args:
            version: Unique version identifier (e.g., "v1", "v2").
            name: Human-readable name.
            template: The prompt template string.
            is_active: Whether this version should be used in A/B testing.
        """
        self._templates[version] = PromptTemplate(
            version=version,
            name=name,
            template=template,
            is_active=is_active,
        )
        self._logger.info(
            "Prompt template registered",
            version=version,
            name=name,
        )

    def select_template(self) -> PromptTemplate:
        """Select a prompt template for use.

        If A/B testing is enabled, randomly selects from active templates.
        Otherwise, selects the latest active template.

        Returns:
            Selected PromptTemplate.
        """
        active_templates = [t for t in self._templates.values() if t.is_active]

        if not active_templates:
            raise ValueError("No active prompt templates available")

        if self._ab_testing_enabled and len(active_templates) > 1:
            # Random selection for A/B testing
            selected = random.choice(active_templates)
            self._logger.debug(
                "A/B test: selected template",
                version=selected.version,
                name=selected.name,
            )
        else:
            # Select latest (highest version number)
            selected = max(active_templates, key=lambda t: t.version)
            self._logger.debug(
                "Selected latest template",
                version=selected.version,
                name=selected.name,
            )

        return selected

    def record_success(self, version: str) -> None:
        """Record a successful query using this prompt version.

        Args:
            version: The prompt version that was used.
        """
        if version in self._templates:
            self._templates[version].success_count += 1

    def record_failure(self, version: str) -> None:
        """Record a failed query using this prompt version.

        Args:
            version: The prompt version that was used.
        """
        if version in self._templates:
            self._templates[version].failure_count += 1

    def get_performance_stats(self) -> dict[str, Any]:
        """Get performance statistics for all prompt versions.

        Returns:
            Dictionary with performance data per version.
        """
        stats = {}
        for version, template in self._templates.items():
            total = template.success_count + template.failure_count
            success_rate = (
                template.success_count / total if total > 0 else 0.0
            )
            stats[version] = {
                "name": template.name,
                "is_active": template.is_active,
                "success_count": template.success_count,
                "failure_count": template.failure_count,
                "total_uses": total,
                "success_rate": round(success_rate, 3),
            }

        return stats

    def deactivate_version(self, version: str) -> None:
        """Deactivate a prompt version (stop using in A/B testing).

        Args:
            version: The version to deactivate.
        """
        if version in self._templates:
            self._templates[version].is_active = False
            self._logger.info("Prompt version deactivated", version=version)

    def activate_version(self, version: str) -> None:
        """Activate a prompt version.

        Args:
            version: The version to activate.
        """
        if version in self._templates:
            self._templates[version].is_active = True
            self._logger.info("Prompt version activated", version=version)

    def get_template(self, version: str) -> PromptTemplate:
        """Get a specific prompt template by version.

        Args:
            version: The version identifier.

        Returns:
            The PromptTemplate.

        Raises:
            KeyError: If version not found.
        """
        return self._templates[version]

    def list_versions(self) -> list[str]:
        """List all registered prompt versions.

        Returns:
            List of version strings.
        """
        return list(self._templates.keys())
