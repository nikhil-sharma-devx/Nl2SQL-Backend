"""IEmbedder — Abstract interface for text embedding providers."""
from abc import ABC, abstractmethod


class IEmbedder(ABC):
    """Contract for embedding text into dense vector representations.

    SOLID: Interface Segregation — focused solely on embedding concerns.
    """

    @abstractmethod
    async def embed(self, text: str) -> list[float]:
        """Embed a single string into a float vector.

        Args:
            text: The text to embed.

        Returns:
            A list of floats representing the dense embedding.
        """
        ...

    @abstractmethod
    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple strings in a single API call (more efficient).

        Args:
            texts: A list of strings to embed.

        Returns:
            A list of float vectors, one per input text.
        """
        ...

    @property
    @abstractmethod
    def dimensions(self) -> int:
        """Return the dimensionality of the embedding vectors."""
        ...
