from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
import gradio as gr

# ----------------------------------------------------------------------------------------------------------------------
# 1. Everything in one function ready for Gradio
# ----------------------------------------------------------------------------------------------------------------------
def RAG_test(pdf=None, urls=None, question="Tell me something interesting"):
    model = ChatOllama(model='mistral')
    docs_split = None

    if pdf:
        loader = PyPDFLoader(pdf)
        # # Default load_and_split - Recursive splitter but with unknown parameters
        # docs_split = loader.load_and_split()
        
        # # Recursive Splitter - 1000/200 good for finding detailed info, but then almost as slow as Semantic.
        # # For more general questions, larger chunk_size needed
        # text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        # docs_split = loader.load_and_split(text_splitter)
        
        # Semantic Chunking - painfully slow (it took more than 12min for 100 pages PDF), but usually very accurate
        # Depending on the type of question, a specific breakpoint_threshold_type performs best.
        docs_split = loader.load()
        hf_embeddings = HuggingFaceEmbeddings()
        text_splitter = SemanticChunker(hf_embeddings, breakpoint_threshold_type="percentile") # "standard_deviation", "interquartile" 
        docs_split = text_splitter.split_documents(docs_split)

    elif urls:
        url_list = urls.split('\n')
        docs = [WebBaseLoader(url).load() for url in url_list]
        docs_list = [item for sublist in docs for item in sublist]
        
        # # Character Chunking - fast and usually good enough
        # text_splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=7500, chunk_overlap=100)
        # docs_split = text_splitter.split_documents(docs_list)

        # Semantic Chunking - for most articles, fast enough, I believe the quality justifies time consumption
        hf_embeddings = HuggingFaceEmbeddings()
        text_splitter = SemanticChunker(hf_embeddings, breakpoint_threshold_type="percentile") # "standard_deviation", "interquartile" 
        docs_split = text_splitter.split_documents(docs_list)

    vectorstore = Chroma.from_documents(
        documents=docs_split,
        collection_name='rag_gradio',
        embedding=OllamaEmbeddings(model='nomic-embed-text')
    )
    retriever = vectorstore.as_retriever()

    after_rag_template = """Answer the question based only on the following context:
    {context}
    Question: {question}
    """
    after_rag_prompt = ChatPromptTemplate.from_template(after_rag_template)
    after_rag_chain = ({'context': retriever, 'question': RunnablePassthrough()} |
                       after_rag_prompt | model | StrOutputParser())

    return after_rag_chain.invoke(question)

# ----------------------------------------------------------------------------------------------------------------------
# 2. Gradio deployment
# ----------------------------------------------------------------------------------------------------------------------
gr.Interface(
    fn=RAG_test,
    inputs=[gr.File(label='Drop pdf file here:'),
            gr.Textbox(label='Enter here each URL in a separate line (shift+enter to enter another URL ;) ):'),
            gr.Textbox(label='Enter your question here:'),
            ],
    outputs=gr.Textbox(label='Mistral has spoken:'),
    title='LLM with RAG and Ollama, demo',
    description="Enter URLs or upload PDF file and ask a question to query the documents."
).launch()
#).launch(share=True)




