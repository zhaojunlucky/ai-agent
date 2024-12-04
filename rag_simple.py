from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.core import SimpleDirectoryReader
from llama_index.core import VectorStoreIndex
from llama_index.core import Settings
from llama_index.vector_stores.qdrant import QdrantVectorStore

Settings.llm = Ollama(model="mixtral:latest", request_timeout=120.0)

Settings.embed_model = OllamaEmbedding(
    base_url='http://127.0.0.1:11434',
    model_name='mixtral:latest',
)

data = '/Users/jun/Downloads/64217-ai-agents-20240606/03-frameworks/docs'

documents = SimpleDirectoryReader(data).load_data()

index = VectorStoreIndex.from_documents(documents)

agent = index.as_query_engine()

response = agent.query("花语秘境的员工有几种角色？")
print(response)

print("====\n")

response = agent.query("go common mistakes")
print(response)
