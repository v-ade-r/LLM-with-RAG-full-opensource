
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings.ollama import OllamaEmbeddings
from flask import Flask, request, render_template
import os


# ----------------------------------------------------------------------------------------------------------------------
# 0. Settings
# ----------------------------------------------------------------------------------------------------------------------
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'PDFs'


@app.route('/')
def home():
    return render_template('index3.html')

# ----------------------------------------------------------------------------------------------------------------------
# 1. Everything in one function ready for Flask
# ----------------------------------------------------------------------------------------------------------------------
@app.route('/RAG_test', methods=['POST'])
def RAG_test():
    pdf = None
    urls = None
    question = "Tell me something interesting"
    model = ChatOllama(model='mistral')
    docs_split = None

    urls = request.form['urls']
    pdf = request.files['pdf']
    question = request.form['question']

    if pdf:
        pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], pdf.filename)
        pdf.save(pdf_path)
        loader = PyPDFLoader(pdf_path)
        docs_split = loader.load_and_split()

    elif urls:
        url_list = urls.split('\n')
        docs = [WebBaseLoader(url).load() for url in url_list]
        docs_list = [item for sublist in docs for item in sublist]
        text_splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=7500, chunk_overlap=100)
        docs_split = text_splitter.split_documents(docs_list)

    vectorstore = Chroma.from_documents(
        documents=docs_split,
        collection_name='rag_flask',
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
    answer = after_rag_chain.invoke(question)

    if pdf:
        os.remove(pdf_path)

    return render_template('index3.html', prediction_text=f"The answer for the question '{question}' "
                                                          f" according to {model} is as follows: <br><br> {answer} ")


if __name__=="__main__":
    app.run(debug=True)






