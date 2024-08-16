from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI

from chat_model.ChatModelDispatcher import ChatModelDispatcher


class XionicKoLlama3_70bChatModel(ChatModelDispatcher):
    __base_url = "http://sionic.chat:8001/v1"
    __api_key = "934c4bbc-c384-4bea-af82-1450d7f8128d"
    __model_name = "xionic-ko-llama-3-70b"
    __temperature = 0.1

    __model: BaseChatModel = None

    @classmethod
    def get_chat_model(cls, *args, **kwargs) -> BaseChatModel:
        if cls.__model is not None:
            return cls.__model

        cls.__model = cls._get_chat_model()

        return cls.__model

    @classmethod
    def _get_chat_model(cls) -> BaseChatModel:
        return ChatOpenAI(
            base_url=cls.__base_url,
            api_key=cls.__api_key,
            model=cls.__model_name,
            temperature=cls.__temperature)
