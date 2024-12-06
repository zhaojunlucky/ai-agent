import logging
import time

import qdrant_client
from llama_index.core import Settings
from llama_index.core import StorageContext
from llama_index.core import VectorStoreIndex
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core.agent import ReActAgent
from llama_index.core.tools import QueryEngineTool
from llama_index.core.tools import ToolMetadata

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
Settings.llm = Ollama(model="mixtral:latest", request_timeout=480.0)

Settings.embed_model = OllamaEmbedding(
    base_url='http://127.0.0.1:11434',
    model_name='mixtral:latest',
)

client = qdrant_client.QdrantClient(
    path="/Users/jun/Downloads/64217-ai-agents-20240606/qdrant_data"
)
LOGGER.info("init vector store")
vector_store = QdrantVectorStore(client=client, collection_name="rag")
storage_context = StorageContext.from_defaults(vector_store=vector_store)

LOGGER.info("load vector store")

index = VectorStoreIndex.from_vector_store(vector_store=vector_store)

query_agent_tools = [
    QueryEngineTool(
        query_engine=index.as_query_engine(similarity_top_k=3),
        metadata=ToolMetadata(
            name="rag",
            description="rag",
        )
    )
]
agent = ReActAgent.from_tools(query_agent_tools, llm=Settings.llm, verbose=True)

# response = agent.chat("花语秘境的员工有几种角色？")
# print(response)
#
# print("====\n")

response = agent.chat("write quick sort")
print(response)

print("====\n")

LOGGER.info("start to query")
start = time.time()
response = agent.chat("what is Misusing init functions")
print(response)

end = time.time()

LOGGER.info("query time: {}".format(end-start))
