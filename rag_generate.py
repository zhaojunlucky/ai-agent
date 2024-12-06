import time

from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.core import SimpleDirectoryReader, StorageContext
from llama_index.core import VectorStoreIndex
from llama_index.core import Settings
from llama_index.vector_stores.qdrant import QdrantVectorStore
import qdrant_client
import logging
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
Settings.llm = Ollama(model="mixtral:latest", request_timeout=120.0)

Settings.embed_model = OllamaEmbedding(
    base_url='http://127.0.0.1:11434',
    model_name='mixtral:latest',
)

# data = '/Users/jun/Downloads/64217-ai-agents-20240606/03-frameworks/docs'
data = '/Users/jun/Downloads/ai/al'

client = qdrant_client.QdrantClient(
    path="/Users/jun/Downloads/64217-ai-agents-20240606/qdrant_data"
)
vector_store = QdrantVectorStore(client=client, collection_name="rag")
storage_context = StorageContext.from_defaults(vector_store=vector_store)
documents = SimpleDirectoryReader(data).load_data()

LOGGER.info("load vector store")

index = VectorStoreIndex.from_documents(documents, storage_context=storage_context, show_progress=True)
agent = index.as_query_engine()

# response = agent.query("花语秘境的员工有几种角色？")
# print(response)
#
# print("====\n")

response = agent.query("write quick sort")
print(response)

print("====\n")

LOGGER.info("start to query")
start = time.time()
response = agent.query("what is Misusing init functions")
print(response)

end = time.time()

LOGGER.info("query time: {}".format(end-start))
