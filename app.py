import os
import logging
import sys
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

# --- Basic Logging Setup ---
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
log = logging.getLogger(__name__)

# --- Load Environment Variables ---
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    # On Render, we use environment variables directly, so this error is for local testing
    log.warning("GOOGLE_API_KEY not found in .env file. Hoping it's set in the deployment environment.")

# --- Define Paths ---
DB_FAISS_PATH = 'db'

# --- Initialize FastAPI App ---
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

# --- Initialize LangChain QA Chain ---
qa_chain = None
try:
    log.info("Initializing the QA Chain...")
    
    # Load the embeddings model
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    
    # Load the FAISS database
    if not os.path.exists(DB_FAISS_PATH):
        raise FileNotFoundError(f"FAISS database not found at path: {DB_FAISS_PATH}. Please make sure the 'db' folder is uploaded.")
    
    db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
    log.info("FAISS database loaded successfully.")

    # Create the prompt template
    prompt_template = """Use the following pieces of context to answer the question at the end. 
    Elaborate the answer so that it is easy to understand.
    Context: {context}
    Question: {question}
    Answer:
    """
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    # Initialize the LLM
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=api_key)

    # Create the QA Chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type='stuff',
        retriever=db.as_retriever(search_kwargs={'k': 3}), # Using 3 sources for better context
        return_source_documents=True,
        chain_type_kwargs={'prompt': prompt}
    )
    log.info("✅ QA Chain initialized successfully!")

except Exception as e:
    log.error(f"❌ Failed during initialization: {e}", exc_info=True)


# --- API Endpoints ---
class Query(BaseModel):
    question: str

@app.get("/")
async def read_index():
    """Serves the frontend HTML file."""
    return FileResponse('static/index.html')

@app.post("/ask")
async def ask(query: Query):
    """Handles user questions and returns answers from the QA chain."""
    if not qa_chain:
        raise HTTPException(status_code=500, detail="Server not initialized properly. Check the logs for errors.")
    
    try:
        log.info(f"Received query: {query.question}")
        res = qa_chain.invoke({'query': query.question})
        answer = res.get("result", "Sorry, I could not find an answer.")
        source_docs = res.get("source_documents", [])
        sources = [os.path.basename(doc.metadata.get('source', 'Unknown')) for doc in source_docs]
        return {"answer": answer, "sources": list(set(sources))} # Use set to remove duplicate sources
    
    except Exception as e:
        log.error(f"Error processing query: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")