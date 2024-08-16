from abc import ABC, abstractmethod
from typing import List, Union

from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever


class RetrieverDispatcher(ABC):
    @abstractmethod
    def get_retriever(self, docs: Union[BaseLoader, List[Document]]) -> BaseRetriever:
        raise NotImplementedError
