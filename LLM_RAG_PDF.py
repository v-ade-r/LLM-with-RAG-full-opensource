from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.embeddings.ollama import OllamaEmbeddings


# ----------------------------------------------------------------------------------------------------------------------
# 0. Choosing the LLM
# ----------------------------------------------------------------------------------------------------------------------
model = ChatOllama(model='mistral')

# ----------------------------------------------------------------------------------------------------------------------
# 1. Splitting into chunks data from PDF
# ----------------------------------------------------------------------------------------------------------------------
loader = PyPDFLoader("Polish_energy_security_in_the_oil_sector.pdf")
docs_split = loader.load_and_split()

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




