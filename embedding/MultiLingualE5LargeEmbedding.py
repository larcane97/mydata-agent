from langchain.embeddings import CacheBackedEmbeddings
from langchain.storage import LocalFileStore
from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings

from embedding.EmbeddingDispatcher import EmbeddingDispatcher


class MultiLingualE5LargeEmbedding(EmbeddingDispatcher):
    __model_name = "intfloat/multilingual-e5-large-instruct"
    __embedding_model: Embeddings = None

    @classmethod
    def get_embedding(cls, *args, **kwargs) -> Embeddings:
        if cls.__embedding_model is not None:
            return cls.__embedding_model
        embedding = HuggingFaceEmbeddings(model_name=cls.__model_name)
        store = LocalFileStore("./cache/")

        cls.__embedding_model = CacheBackedEmbeddings.from_bytes_store(
            underlying_embeddings=embedding,
            document_embedding_cache=store,
            namespace=cls.__model_name
        )

        return cls.__embedding_model
