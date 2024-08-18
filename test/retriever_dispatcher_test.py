from config.Config import Config
from retriever.BM25RetrieverDispatcher import BM25RetrieverDispatcher
from langchain_core.documents import Document

from retriever.FaissRetrieverDispatcher import FaissRetrieverDispatcher


def test_bm25_retriver_dispatcher():
    test_docs = [
        Document('“어제의 나보다 더 성장하기 위해 노력하는 개발자” 임문경입니다.'),
        Document('학부 시절부터 여러 분야에 관심이 많아 웹, 앱, AI 등 여러 분야를 공부했고, 다양한 프로젝트를 진행하였습니다.'),
        Document('그러다 전체적인 인프라나 파이프라이닝에 흥미를 더욱 흥미를 느끼게 되었고, 현재는 AI 모델에 적절한 피처를 전달하고 관리하는 MLOps 엔지니어로 활동하고 있습니다.'),
        Document('하지만 여전히 다양한 분야에 관심이 있고, 추후에는 데이터 엔지니어링과 DevOps 등 더욱 다양한 분야를 배워보려 합니다 :)'),
    ]

    retriever_dispatcher = BM25RetrieverDispatcher()
    retriever = retriever_dispatcher.get_retriever(test_docs, k=1)
    relevant_doc = retriever.invoke('“어제의 나보다 더 성장하기 위해 노력하는 개발자” 임문경입니다.')[0]

    assert relevant_doc.page_content == test_docs[0].page_content


def test_faiss_retriver_dispatcher():
    test_docs = [
        Document('“어제의 나보다 더 성장하기 위해 노력하는 개발자” 임문경입니다.'),
        Document('학부 시절부터 여러 분야에 관심이 많아 웹, 앱, AI 등 여러 분야를 공부했고, 다양한 프로젝트를 진행하였습니다.'),
        Document('그러다 전체적인 인프라나 파이프라이닝에 흥미를 더욱 흥미를 느끼게 되었고, 현재는 AI 모델에 적절한 피처를 전달하고 관리하는 MLOps 엔지니어로 활동하고 있습니다.'),
        Document('하지만 여전히 다양한 분야에 관심이 있고, 추후에는 데이터 엔지니어링과 DevOps 등 더욱 다양한 분야를 배워보려 합니다 :)'),
    ]

    retriever_dispatcher = FaissRetrieverDispatcher(embedding_model=Config.embedding_model())
    retriever = retriever_dispatcher.get_retriever(test_docs, search_kwargs={"k": 1})
    relevant_doc = retriever.invoke('“어제의 나보다 더 성장하기 위해 노력하는 개발자” 임문경입니다.')[0]

    assert relevant_doc.page_content == test_docs[0].page_content
