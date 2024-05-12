from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter


# ----------------------------------------------------------------------------------------------------------------------
# 0. Choosing the LLM
# ----------------------------------------------------------------------------------------------------------------------
model = ChatOllama(model='mistral')

# ----------------------------------------------------------------------------------------------------------------------
# 1. Splitting into chunks data from URLs
# ----------------------------------------------------------------------------------------------------------------------
urls = [
    "https://spectacularnwt.com/attractions/ibyuk-pingo/",
    "https://ollama.com/",
    ]

docs = [WebBaseLoader(url).load() for url in urls]
docs_list = [item for sublist in docs for item in sublist]

# #Character Chunking - fast and usually good enough
# text_splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=7500, chunk_overlap=100)
# docs_split = text_splitter.split_documents(docs_list)

# Semantic Chunking - for most articles, fast enough, I believe the quality justifies time consumption
hf_embeddings = HuggingFaceEmbeddings()
text_splitter = SemanticChunker(hf_embeddings, breakpoint_threshold_type="percentile")
docs_split = text_splitter.split_documents(docs_list)

# ----------------------------------------------------------------------------------------------------------------------
# 2. Converting documents to Embeddings and storing them
# ----------------------------------------------------------------------------------------------------------------------
vectorstore = Chroma.from_documents(
    documents=docs_split,
    collection_name="rag_demo",
    embedding=OllamaEmbeddings(model='nomic-embed-text')
)
retriever = vectorstore.as_retriever()

# ----------------------------------------------------------------------------------------------------------------------
# 3. Asking LLM without RAG
# ----------------------------------------------------------------------------------------------------------------------
without_rag_template = '{question}'
without_rag_prompt = ChatPromptTemplate.from_template(without_rag_template)
without_rag_chain = without_rag_prompt | model | StrOutputParser()
print("Answer without RAG:\n")
print(without_rag_chain.invoke({'question': 'What is Ibyuk?'}))

# ----------------------------------------------------------------------------------------------------------------------
# 4. Asking LLM with RAG
# ----------------------------------------------------------------------------------------------------------------------
with_rag_template = """Answer the question based only on the following context:
{context}
Question: {question}
"""
with_rag_prompt = ChatPromptTemplate.from_template(with_rag_template)
with_rag_chain = ({'context': retriever, 'question': RunnablePassthrough()} |
                  with_rag_prompt | model | StrOutputParser())
print("Answer with RAG:\n")
print(with_rag_chain.invoke('What is Ibyuk?'))





