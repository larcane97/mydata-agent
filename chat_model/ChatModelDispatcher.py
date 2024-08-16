from abc import ABC, abstractmethod

from langchain_core.language_models import BaseChatModel


class ChatModelDispatcher(ABC):
    @classmethod
    @abstractmethod
    def get_chat_model(cls, *args, **kwargs) -> BaseChatModel:
        raise NotImplementedError

