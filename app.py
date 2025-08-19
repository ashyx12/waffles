from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
import os
import logging, sys, traceback
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# Logging setup
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

load_dotenv()

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

DB_FAISS_PATH = 'db'

# --- Initialize FAISS and API safely ---
embeddings = None
db = None
qa_chain = None

try:
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not set in environment!")

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)

    prompt_template = """Use the following pieces of context to answer the question at the end. 
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
        retriever=db.as_retriever(search_kwargs={'k': 2}),
        return_source_documents=True,
        chain_type_kwargs={'prompt': prompt}
    )
except Exception as init_err:
    logging.error("Failed during initialization: %s", traceback.format_exc())

class Query(BaseModel):
    question: str

@app.get("/")
async def read_index():
    return FileResponse('static/index.html')

@app.post("/ask")
async def ask(query: Query):
    if qa_chain is None:
        raise HTTPException(
            status_code=500,
            detail="Server not initialized properly. Check FAISS DB path and GOOGLE_API_KEY."
        )
    try:
        res = qa_chain.invoke({'query': query.question})
        answer = res["result"]
        source_docs = res.get("source_documents", [])
        sources = [os.path.basename(doc.metadata.get('source', 'Unknown')) for doc in source_docs]
        return {"answer": answer, "sources": sources}
    except Exception as e:
        logging.error("Error in /ask: %s", traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")
