import os
import time

import chromadb
import httpx
from httpx import Timeout
from langchain import hub
from langchain.chains import RetrievalQA
from langchain_core.output_parsers import StrOutputParser
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
from ollama import embed

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
Settings.llm = Ollama(model="codellama:13b", request_timeout=60000.0)

Settings.embed_model = OllamaEmbedding(
    base_url='http://127.0.0.1:11434',
    model_name='codellama:13b',
)

go_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.GO, chunk_size=50, chunk_overlap=0
)

data = '/Users/jun/magicworldz/github/golib'

LOGGER.info("load code")
documents = []
os.environ['HTTPX_TIMEOUT'] = '0'
chroma_path = '/Users/jun/Downloads/64217-ai-agents-20240606/chroma_data'
embed_func = OllamaEmbeddings(model="codellama:13b", client_kwargs={'timeout': Timeout(None, connect=5.0)})
# db = Chroma.from_texts(['golang'], embed_func,
#                            persist_directory=chroma_path)
# for root, dirs, files in os.walk(data):
#     for file in files:
#         if file.endswith('.go') and not file.endswith('_test.go'):
#             path = os.path.join(root, file)
#             LOGGER.info(path)
#             with open(path, 'r', encoding='utf-8') as f:
#                 go_docs = go_splitter.create_documents([f.read()], metadatas=[{"source": path}])
#                 # documents.extend(go_docs)
#                 try:
#                     db.add_documents(go_docs)
#                 except Exception as e:
#                     LOGGER.error("failed to add documents", exc_info=e)
#                     continue

db = Chroma(embedding_function=embed_func, persist_directory=chroma_path)


LOGGER.info("created Chroma db")

# query_text = "get object as slice"
# results = db.similarity_search(query_text)
#
# for result in results:
#     print(result.page_content)

# qa_chain = RetrievalQA.from_llm(
#     llm=OllamaLLM(model="codellama:13b"), retriever=db.as_retriever(), return_source_documents=True,
# )

system_prompt = (
    "Use the given context to answer the question. "
    "If you don't know the answer, say you don't know. "
    "Use three sentence maximum and keep the answer concise. "
    "Context: {context}"
)
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("user", "{input}"),
    ]
)

retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
question_answer_chain = create_stuff_documents_chain(OllamaLLM(model="codellama:13b"), retrieval_qa_chat_prompt)
chain = create_retrieval_chain(db.as_retriever(), question_answer_chain)


results = chain.invoke({'input':"Encrypt With Public key"})

print(results)
# LOGGER.info("create chroma vector store")
# vector_store = ChromaVectorStore(db)
# index = VectorStoreIndex.from_vector_store(vector_store, embed_model=OllamaEmbeddings(model="codellama:13b"))
#
# LOGGER.info("start to query")


#
# agent = index.as_query_engine()
# response = agent.query("get object as slice")
# print(response)
