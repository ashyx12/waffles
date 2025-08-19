import os
import logging
import sys
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
log = logging.getLogger(__name__)

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
DB_FAISS_PATH = os.path.join(BASE_DIR, "db")

app = FastAPI()

qa_chain = None
try:
    log.info("Initializing the QA Chain...")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)

    if not os.path.exists(DB_FAISS_PATH):
        raise FileNotFoundError(f"Database not found at {DB_FAISS_PATH}. Ensure the 'db' folder is deployed.")
        
    db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
    log.info("FAISS database loaded successfully.")

    prompt_template = """See if the user is conversating with you or asking a question regarding
    Embedded Systems, if the user is conversating with you just talk to the user. If the user is asking
    a question regarding Embedded Systems then
    Use the following pieces of context to answer the question at the end. 
    Elaborate the answer so that it is easy to understand.
    Context: {context}
    Question: {question}
    Answer:
    """
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=api_key)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type='stuff',
        retriever=db.as_retriever(search_kwargs={'k': 3}),
        return_source_documents=True,
        chain_type_kwargs={'prompt': prompt}
    )
    log.info("QA Chain initialized successfully!")

except Exception as e:
    log.error(f"Failed during initialization: {e}", exc_info=True)

class Query(BaseModel):
    question: str

@app.post("/ask")
async def ask(query: Query):
    if not qa_chain:
        raise HTTPException(status_code=500, detail="Server not initialized properly.")
    
    try:
        log.info(f"Received query: {query.question}")
        res = qa_chain.invoke({'query': query.question})
        answer = res.get("result", "Sorry, I could not find an answer.")
        sources = [os.path.basename(doc.metadata.get('source', 'Unknown')) for doc in res.get("source_documents", [])]
        return {"answer": answer, "sources": list(set(sources))}
    
    except Exception as e:
        log.error(f"Error processing query: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")