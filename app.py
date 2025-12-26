from flask import Flask, request, jsonify,render_template
from functools import lru_cache
from langchain_community.document_loaders import YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from langchain.schema import Document
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
_TEXT_SPLITTER = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=100)
_CHAT = ChatOpenAI(model_name="gpt-3.5-turbo-16k", temperature=0.2)

def get_text_splitter():
    return _TEXT_SPLITTER

def get_chat():
    return _CHAT

def normalize_video_url(video_url: str) -> str:
    cleaned = (video_url or "").strip()
    if not cleaned:
        raise ValueError("video_url must be provided.")
    return cleaned

def get_transcript_pages(video_url):
    normalized_url = normalize_video_url(video_url)
    return _get_transcript_pages(normalized_url)

@lru_cache(maxsize=8)
def _get_transcript_pages(normalized_url: str):
    loader = YoutubeLoader.from_youtube_url(normalized_url)
    try:
        transcript = loader.load()
    except Exception:
        _get_transcript_pages.cache_clear()
        _get_split_chunks.cache_clear()
        raise
    if not transcript:
        raise ValueError("Transcript could not be retrieved for the provided URL.")
    return tuple(doc.page_content for doc in transcript)

def get_split_chunks(video_url):
    normalized_url = normalize_video_url(video_url)
    return _get_split_chunks(normalized_url)

@lru_cache(maxsize=8)
def _get_split_chunks(normalized_url: str):
    transcript_pages = _get_transcript_pages(normalized_url)
    splitter = get_text_splitter()
    split_texts = []
    for page in transcript_pages:
        split_texts.extend(splitter.split_text(page))
    return tuple(split_texts)

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
    split_chunks = get_split_chunks(video_url)
    docs = [Document(page_content=chunk) for chunk in split_chunks]

    db = FAISS.from_documents(docs, embeddings)
    return db

def get_response_from_query(db, query, k=4):
    docs = db.similarity_search(query, k=k)
    docs_page_content = " ".join([d.page_content for d in docs])

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

    chain = LLMChain(llm=get_chat(), prompt=chat_prompt)

    response = chain.run(question=query, docs=docs_page_content)
    response = response.replace("\n", "")
    return response, docs
