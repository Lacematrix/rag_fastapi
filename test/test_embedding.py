import os
import sys
sys.path.append("..")
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.services.embedding_service import EmbeddingService
embeddings = EmbeddingService(
                model="qwen3-embedding:0.6b",
                openai_api_base='http://127.0.0.1:11434/v1/',#说明一下，openai会自己拼接成http://127.0.0.1:11434/v1/embeddings
                api_key="ollama",
                chunk_size=512,
                )
output = embeddings.embed_query("你好")  
from app.services.vector_store import MilvusVectorStoreService
from app.core.config import Settings
connection_args = {"uri": './milvus.db'}
connection_args["db_name"] = "my_database_1"
connection_args["token"] = ""
connection_args['alias'] = 'default'

store = MilvusVectorStoreService(
    settings=Settings(),
    embedding_service=embeddings
)


from uuid import uuid4

from langchain_core.documents import Document

document_1 = Document(
    page_content="I had chocalate chip pancakes and scrambled eggs for breakfast this morning.",
    metadata={"source": "tweet"},
)

document_2 = Document(
    page_content="The weather forecast for tomorrow is cloudy and overcast, with a high of 62 degrees.",
    metadata={"source": "news"},
)

document_3 = Document(
    page_content="Building an exciting new project with LangChain - come check it out!",
    metadata={"source": "tweet"},
)

document_4 = Document(
    page_content="Robbers broke into the city bank and stole $1 million in cash.",
    metadata={"source": "news"},
)

document_5 = Document(
    page_content="Wow! That was an amazing movie. I can't wait to see it again.",
    metadata={"source": "tweet"},
)

document_6 = Document(
    page_content="Is the new iPhone worth the price? Read this review to find out.",
    metadata={"source": "website"},
)

document_7 = Document(
    page_content="The top 10 soccer players in the world right now.",
    metadata={"source": "website"},
)

document_8 = Document(
    page_content="LangGraph is the best framework for building stateful, agentic applications!",
    metadata={"source": "tweet"},
)

document_9 = Document(
    page_content="The stock market is down 500 points today due to fears of a recession.",
    metadata={"source": "news"},
)

document_10 = Document(
    page_content="I have a bad feeling I am going to get deleted :(",
    metadata={"source": "tweet"},
)

documents = [
    document_1,
    document_2,
    document_3,
    document_4,
    document_5,
    document_6,
    document_7,
    document_8,
    document_9,
    document_10,
]
uuids = [str(uuid4()) for _ in range(len(documents))]

store.add_documents(store.settings.default_collection, docs=documents, ids=uuids)

results = store.retrieve(
    collection_name=store.settings.default_collection,
    query=
    "LangChain provides abstractions to make working with LLMs easy",
    top_k=2,
    # param=...  # Search params for the index type
)
for res in results:
    print(f"* {res.text} [{res.metadata}]")