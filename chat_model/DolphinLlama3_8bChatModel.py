from langchain_community.chat_models import ChatOllama
from langchain_core.language_models import BaseChatModel

from chat_model.ChatModelDispatcher import ChatModelDispatcher


class DolphinLlama3_8bChatModel(ChatModelDispatcher):
    __model_name = "dolphin-llama3:8b"

    __model: BaseChatModel = None

    @classmethod
    def get_chat_model(cls, *args, **kwargs) -> BaseChatModel:
        if cls.__model is not None:
            return cls.__model

        cls.__model = cls._get_chat_model()
        return cls.__model

    @classmethod
    def _get_chat_model(cls) -> BaseChatModel:
        return ChatOllama(model=cls.__model_name)
