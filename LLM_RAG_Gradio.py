from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings.ollama import OllamaEmbeddings
import gradio as gr

# ----------------------------------------------------------------------------------------------------------------------
# 1. Everything in one function ready for Gradio
# ----------------------------------------------------------------------------------------------------------------------
def RAG_test(pdf=None, urls=None, question="Tell me something interesting"):
    model = ChatOllama(model='mistral')
    docs_split = None

    if pdf:
        loader = PyPDFLoader(pdf)
        docs_split = loader.load_and_split()

    elif urls:
        url_list = urls.split('\n')
        docs = [WebBaseLoader(url).load() for url in url_list]
        docs_list = [item for sublist in docs for item in sublist]
        text_splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=7500, chunk_overlap=100)
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




