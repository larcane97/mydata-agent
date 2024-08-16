from abc import ABC, abstractmethod

from langchain_core.embeddings import Embeddings


class EmbeddingDispatcher(ABC):

    @classmethod
    @abstractmethod
    def get_embedding(cls, *args, **kwargs) -> Embeddings:
        raise NotImplementedError
