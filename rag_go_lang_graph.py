import os
import time

import chromadb
import httpx
from httpx import Timeout
from langchain import hub
from langchain.chains import RetrievalQA
from langchain_core.output_parsers import StrOutputParser
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_neo4j import Neo4jVector, Neo4jGraph, GraphCypherQAChain
from langchain_openai import ChatOpenAI
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from langchain.prompts  import PromptTemplate
from llama_index.core import Settings, ChatPromptTemplate
from llama_index.vector_stores.chroma import ChromaVectorStore
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, OllamaLLM
import logging
from langchain_text_splitters import (
    Language,
    RecursiveCharacterTextSplitter,
)
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain


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
Settings.llm = Ollama(model="mistral", request_timeout=60000.0)

Settings.embed_model = OllamaEmbedding(
    base_url='http://127.0.0.1:11434',
    model_name='mistral',
)
llm = ChatOpenAI(model='mistral', api_key='ollama', base_url='http://127.0.0.1:11434/v1', temperature=0)

llm_transformer = LLMGraphTransformer(llm=llm)
go_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.GO
)

data = '/Users/jun/magicworldz/github/golib'

LOGGER.info("load code")
documents = []
os.environ['HTTPX_TIMEOUT'] = '0'
os.environ["NEO4J_URI"] = "bolt://10.53.1.66:7687"
os.environ["NEO4J_USERNAME"] = "neo4j"
os.environ["NEO4J_PASSWORD"] = "Password123!"
embed_func = OllamaEmbeddings(model="mistral", client_kwargs={'timeout': Timeout(None, connect=5.0)})
graph = Neo4jGraph()


def generate():
    for root, dirs, files in os.walk(data):
        for file in files:
            if file.endswith('.go') and not file.endswith('_test.go'):
                path = os.path.join(root, file)
                LOGGER.info(path)
                if path.endswith('encoding_detect.go'): continue
                with open(path, 'r', encoding='utf-8') as f:
                    go_docs = go_splitter.create_documents([f.read()], metadatas=[{"source": path}])
                    # documents.extend(go_docs)
                    try:
                        graph.add_graph_documents(llm_transformer.convert_to_graph_documents(go_docs),
                                                  baseEntityLabel=True,
                                                  include_source=True
                                                  )
                    except Exception as e:
                        LOGGER.error("failed to add documents", exc_info=e)
                        continue

def query_rag():
    CYPHER_GENERATION_TEMPLATE = """Task: Generate Cypher statement to query a graph database.
    ...
    """
    CYPHER_GENERATION_PROMPT = PromptTemplate(
        input_variables=["schema", "question"], template=CYPHER_GENERATION_TEMPLATE
    )

    CYPHER_QA_TEMPLATE = """You are an assistant that helps to form nice and human understandable answers.
    ...
    """
    CYPHER_QA_PROMPT = PromptTemplate(
        input_variables=["context", "question"], template=CYPHER_QA_TEMPLATE
    )

    graph_chain = GraphCypherQAChain.from_llm(
        llm,
        graph=graph,
        cypher_prompt=CYPHER_GENERATION_PROMPT,
        qa_prompt=CYPHER_QA_PROMPT,
        verbose=True,
        allow_dangerous_requests=True,
    )


    query = 'get object as slice'

    ctx_query = 'in the golang, pass a any type variable, convert it as a slice'

    results = graph_chain.invoke({'query': query})

    print(results)


if __name__ == '__main__':
    generate()
    query_rag()
