import numpy as np

from embedding.MultiLingualE5LargeEmbedding import MultiLingualE5LargeEmbedding


def test_multi_lingual_e5_large_embedding():
    test_docs = [
        '“어제의 나보다 더 성장하기 위해 노력하는 개발자” 임문경입니다.',
        '학부 시절부터 여러 분야에 관심이 많아 웹, 앱, AI 등 여러 분야를 공부했고, 다양한 프로젝트를 진행하였습니다.',
        '그러다 전체적인 인프라나 파이프라이닝에 흥미를 더욱 흥미를 느끼게 되었고, 현재는 AI 모델에 적절한 피처를 전달하고 관리하는 MLOps 엔지니어로 활동하고 있습니다.',
        '하지만 여전히 다양한 분야에 관심이 있고, 추후에는 데이터 엔지니어링과 DevOps 등 더욱 다양한 분야를 배워보려 합니다 :)',
    ]

    embedding_model = MultiLingualE5LargeEmbedding.get_embedding()

    embed_docs = np.array(embedding_model.embed_documents(test_docs))

    query = "개발자 임문경입니다."
    embed_query = np.array(embedding_model.embed_query(query))

    t = embed_docs @ embed_query.T
    retrieved_doc = test_docs[t.argmax()]

    assert retrieved_doc == test_docs[0]
