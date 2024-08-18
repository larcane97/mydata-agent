from config.Config import Config


def test_embedding_singleton():
    embedding_model1 = Config.embedding_model()
    embedding_model2 = Config.embedding_model()

    assert id(embedding_model1) == id(embedding_model2)


def test_chat_model_singleton():
    chat_model1 = Config.chat_model()
    chat_model2 = Config.chat_model()

    assert id(chat_model1) == id(chat_model2)
