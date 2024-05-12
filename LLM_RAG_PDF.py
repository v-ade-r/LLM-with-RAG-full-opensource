from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker


# ----------------------------------------------------------------------------------------------------------------------
# 0. Choosing the LLM
# ----------------------------------------------------------------------------------------------------------------------
model = ChatOllama(model='mistral')

# ----------------------------------------------------------------------------------------------------------------------
# 1. Splitting into chunks data from PDF
# ----------------------------------------------------------------------------------------------------------------------
loader = PyPDFLoader("Polish_energy_security_in_the_oil_sector.pdf")
# # Default load_and_split - Recursive splitter but with unknown parameters
# docs_split = loader.load_and_split()

# # Recursive Splitter - 1000/200 good for finding detailed info, but then almost as slow as Semantic.
# # For more general questions, larger chunk_size needed
# text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
# docs_split = loader.load_and_split(text_splitter)

# Semantic Chunking - painfully slow (it took more than 12min for 100 pages PDF), but usually very accurate
docs_split = loader.load()
hf_embeddings = HuggingFaceEmbeddings()
text_splitter = SemanticChunker(hf_embeddings, breakpoint_threshold_type="standard_deviation") # "percentile", "interquartile"
docs_split = text_splitter.split_documents(docs_split)

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
print(without_rag_chain.invoke({'question': 'How much crude oil does Saudi Arabia consume per year?'}))

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
print(with_rag_chain.invoke('How much crude oil does Saudi Arabia consume per year?'))




