from abc import abstractmethod, ABC

from langchain_core.embeddings import Embeddings
from langchain_core.tools import Tool
from langchain_text_splitters import TextSplitter


class ToolDispatcher(ABC):
    @classmethod
    @abstractmethod
    def get_tool(cls, /, **kwargs) -> Tool:
        raise NotImplementedError
