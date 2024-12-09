import json
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

model = "codellama:13b"
LOGGER.info("start")
Settings.llm = Ollama(model=model, request_timeout=60000.0)

Settings.embed_model = OllamaEmbedding(
    base_url='http://127.0.0.1:11434',
    model_name=model,
)

go_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON, chunk_size=50, chunk_overlap=0
)

data = '/Users/jun/magicworldz/github/scripts'

LOGGER.info("load code")
documents = []
os.environ['HTTPX_TIMEOUT'] = '0'
chroma_path = '/Users/jun/Downloads/64217-ai-agents-20240606/chroma_python_data'
embed_func = OllamaEmbeddings(model=model, client_kwargs={'timeout': Timeout(None, connect=5.0)})
db = Chroma.from_texts(['python'], embed_func,
                           persist_directory=chroma_path)
for root, dirs, files in os.walk(data):
    for file in files:
        if file.endswith('.py'):
            path = os.path.join(root, file)
            if '/venv/' in path: continue
            LOGGER.info(path)

            with open(path, 'r', encoding='utf-8') as f:
                py_docs = go_splitter.create_documents([f.read()], metadatas=[{"source": path}])
                if not py_docs: continue
                try:
                    db.add_documents(py_docs)
                except Exception as e:
                    LOGGER.error("failed to add documents", exc_info=e)
                    continue

# db = Chroma(embedding_function=embed_func, persist_directory=chroma_path)

LOGGER.info("created Chroma db")

retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
question_answer_chain = create_stuff_documents_chain(OllamaLLM(model=model), retrieval_qa_chat_prompt)
chain = create_retrieval_chain(db.as_retriever(), question_answer_chain)


results = chain.invoke({'input':"use the above context to get nnr rules"})

print(results)

