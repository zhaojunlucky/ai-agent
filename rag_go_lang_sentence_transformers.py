import logging
import os

from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM
from langchain_text_splitters import (
    Language,
    RecursiveCharacterTextSplitter,
)
from llama_index.core import Settings
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from langchain.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)
# from langchain_huggingface import HuggingFaceEmbeddings
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
Settings.llm = Ollama(model="codellama:34b", request_timeout=60000.0)

Settings.embed_model = OllamaEmbedding(
    base_url='http://127.0.0.1:11434',
    model_name='codellama:34b',
)

go_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.GO,
    chunk_size=1000,
    chunk_overlap=200
)

data = '/Users/jun/magicworldz/github/golib'

LOGGER.info("load code")
documents = []
os.environ['HTTPX_TIMEOUT'] = '0'
chroma_path = '/Users/jun/Downloads/64217-ai-agents-20240606/chroma_data_sentence_transformers'
# embed_func = OllamaEmbeddings(model="codellama:34b", client_kwargs={'timeout': Timeout(None, connect=5.0)})
embed_func = SentenceTransformerEmbeddings(model_name='all-MiniLM-L6-v2')


def generate():
    db = Chroma.from_texts(['golang'], embed_func,
                               persist_directory=chroma_path)

    for root, dirs, files in os.walk(data):
        for file in files:
            if file.endswith('.go') and not file.endswith('_test.go'):
                path = os.path.join(root, file)
                LOGGER.info(path)
                with open(path, 'r', encoding='utf-8') as f:
                    go_docs = go_splitter.create_documents([f.read()], metadatas=[{"source": path}])
                    # documents.extend(go_docs)
                    try:
                        db.add_documents(go_docs)
                    except Exception as e:
                        LOGGER.error("failed to add documents", exc_info=e)
                        continue

def query_rag():
    db = Chroma(embedding_function=embed_func, persist_directory=chroma_path)


    LOGGER.info("created Chroma db")

    # retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    # retrieval_qa_chat_prompt = hub.pull("rlm/rag-prompt")
    retrieval_qa_chat_prompt = ChatPromptTemplate.from_template("""
        Answer the following golang question based only on the provided context. 
        Think step by step before providing a detailed answer. 
        <context>
        {context}
        </context>
        Question: {input}""")
    # Create document chain
    chain = create_stuff_documents_chain(OllamaLLM(model="codellama:34b"), retrieval_qa_chat_prompt)
    chain = create_retrieval_chain(db.as_retriever(), chain)

    query = 'get object as slice'
    context = db.similarity_search(query, k=3)

    ctx_query = f'get object with type any as slice'

    results = chain.invoke({'context': context, 'question': ctx_query, 'input': ctx_query})

    print(results['answer'])


if __name__ == '__main__':
    generate()
    query_rag()
