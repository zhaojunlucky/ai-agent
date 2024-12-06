import os
import time

from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.core import SimpleDirectoryReader, StorageContext
from llama_index.core import VectorStoreIndex
from llama_index.core import Settings
from llama_index.vector_stores.qdrant import QdrantVectorStore
import qdrant_client
import logging
from langchain_text_splitters import (
    Language,
    RecursiveCharacterTextSplitter,
)

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Create a console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
console_handler.setFormatter(formatter)

# Add handlers to the logger
LOGGER.addHandler(console_handler)

LOGGER.info("start")
Settings.llm = Ollama(model="codellama:13b", request_timeout=120.0)

Settings.embed_model = OllamaEmbedding(
    base_url='http://127.0.0.1:11434',
    model_name='codellama:13b',
)

data = '/Users/jun/magicworldz/github/golib'
documents = []
for root, dirs, files in os.walk(data):
    for file in files:
        if file.endswith('.go') and not file.endswith('_test.go'):
            path = os.path.join(root, file)
            LOGGER.info(path)
            go_splitter = RecursiveCharacterTextSplitter.from_language(
                language=Language.GO, chunk_size=50, chunk_overlap=0
            )
            with open(path, 'r', encoding='utf-8') as f:
                go_docs = go_splitter.create_documents([f.read()])
                documents.extend(go_docs)

client = qdrant_client.QdrantClient(
    path="/Users/jun/Downloads/64217-ai-agents-20240606/qdrant_go_data"
)
vector_store = QdrantVectorStore(client=client, collection_name="go")
storage_context = StorageContext.from_defaults(vector_store=vector_store)

LOGGER.info("load vector store")

index = VectorStoreIndex.from_documents(documents, storage_context=storage_context, show_progress=True)
agent = index.as_query_engine()

LOGGER.info("start to query")
start = time.time()
response = agent.query("get object as slice")
print(response)

end = time.time()

LOGGER.info("query time: {}".format(end-start))
