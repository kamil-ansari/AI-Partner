import os

from flask import Flask, request, jsonify
from flask import Flask, render_template
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.document_loaders import TextLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain import OpenAI
from langchain.chat_models import ChatOpenAI
from flask_cors import CORS
from dotenv import load_dotenv
from pathlib import Path

import os


from langchain import OpenAI
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import TextLoader
from langchain.embeddings import OpenAIEmbeddings
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from pathlib import Path


load_dotenv(Path("./.env"))
app = Flask(__name__)

CORS(app)
# Set API key
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Initialize the model
#chatgpt = ChatOpenAI(model_name='gpt-3')

@app.route('/chat', methods=['POST'])
def chat():
    query = request.json['userText']


    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

    llm_gpt = ChatOpenAI(
    model_name='gpt-3.5-turbo',
    temperature=0.9,
    max_tokens=500
)

    vectordb = Chroma(persist_directory = 'vectorStore', collection_name = "my_collection",embedding_function = OpenAIEmbeddings())
    retriever = vectordb.as_retriever()
    #llm=OpenAI(temperature=0.1)
    qa = RetrievalQA.from_chain_type(llm=llm_gpt,chain_type="stuff",retriever = retriever)
    response = qa.run(query)

    return jsonify({'botResponse': response})


@app.route('/load_data', methods=['GET'])
def load_data():
    load_dotenv(Path("./.env"))
    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, 'context.txt') 
    loader = TextLoader(file_path,'utf-8')
    embeddings = OpenAIEmbeddings()

    index = VectorstoreIndexCreator(
        # split the documents into chunks
        text_splitter=CharacterTextSplitter(chunk_size=1000, chunk_overlap=0,),
        # select which embeddings we want to use
        embedding=embeddings,
        # use Chroma as the vectorestore to index and search embeddings
        vectorstore_cls=Chroma,
        vectorstore_kwargs={"persist_directory": "vectorStore", "collection_name":"my_collection"}
    ).from_loaders([loader])

    return jsonify({'status': 'data loaded'})


@app.route('/')
def home():
    with open('./templates/index.html', 'r', encoding='utf-8') as html_file:
            return html_file.read()
    
@app.route('/story')
def homeStory():
    return render_template('index2.html')

if __name__ == "__main__":
    app.run()
