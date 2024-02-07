from flask import Flask, request, jsonify,render_template
from langchain_community.document_loaders import YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from dotenv import find_dotenv, load_dotenv
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
import textwrap

app = Flask(__name__)
load_dotenv(find_dotenv())
embeddings = OpenAIEmbeddings()

@app.route('/')
def welcome():
    return render_template('index.html') 

@app.route('/api/process_query', methods=['POST'])
def process_query():
    data = request.get_json()

    video_url = data.get('video_url')
    query = data.get('query')

    if not video_url or not query:
        return jsonify({'error': 'Missing video_url or query parameter'}), 400

    db = create_db_from_youtube_video_url(video_url)
    response, docs = get_response_from_query(db, query)
    formatted_response = textwrap.fill(response, width=50)
    serializable_docs = [{'page_content': d.page_content} for d in docs]
    
    return jsonify({'response': formatted_response, 'docs': serializable_docs})

def create_db_from_youtube_video_url(video_url):
    loader = YoutubeLoader.from_youtube_url(video_url)
    transcript = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=100)
    docs = text_splitter.split_documents(transcript)

    db = FAISS.from_documents(docs, embeddings)
    return db

def get_response_from_query(db, query, k=4):
    docs = db.similarity_search(query, k=k)
    docs_page_content = " ".join([d.page_content for d in docs])

    chat = ChatOpenAI(model_name="gpt-3.5-turbo-16k", temperature=0.2)

    template = """
        You are a helpful assistant that can answer questions about YouTube videos 
        based on the video's transcript: {docs}
        
        Only use the factual information from the transcript to answer the question.
        
        If you feel like you don't have enough information to answer the question, say "I don't know".
        """

    system_message_prompt = SystemMessagePromptTemplate.from_template(template)

    human_template = "Answer the following question: {question}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])

    chain = LLMChain(llm=chat, prompt=chat_prompt)

    response = chain.run(question=query, docs=docs_page_content)
    response = response.replace("\n", "")
    return response, docs
