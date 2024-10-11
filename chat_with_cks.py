from flask import Flask, request, render_template
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.document_loaders import BSHTMLLoader, DirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
import markdown

app = Flask(__name__)

def load_documents(directory: str):
    loader = DirectoryLoader(directory, glob="**/*.html", loader_cls=BSHTMLLoader)
    return loader.load()

def create_embeddings(documents):
    embeddings = OllamaEmbeddings(model="mxbai-embed-large:latest")
    text_splitter = RecursiveCharacterTextSplitter()
    split_docs = text_splitter.split_documents(documents)
    return FAISS.from_documents(split_docs, embeddings)

def setup_retrieval_chain(vector):
    llm = Ollama(model="gemma2:27b")
    prompt = ChatPromptTemplate.from_template("""Answer the following question based only on the provided context:

<context>
{context}
</context>

Question: {input}""")
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = vector.as_retriever()
    return create_retrieval_chain(retriever, document_chain)

def handle_chat(question: str, retrieval_chain):
    response = retrieval_chain.invoke({"input": question})
    return response["answer"]

@app.route('/', methods=['GET', 'POST'])
def chat():
    if request.method == 'POST':
        question = request.form['question']
        response = handle_chat(question, retrieval_chain)
        html_content = markdown.markdown(response)
        return render_template('chat.html', question=question, response=html_content)
    return render_template('chat.html')

if __name__ == '__main__':
    # Initialize the LLM and load documents

    documents = load_documents("./CKS")
    vector = create_embeddings(documents)
    retrieval_chain = setup_retrieval_chain(vector)

    app.run(debug=True, port=5001)