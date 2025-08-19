# app.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

load_dotenv()

# Initialize the FastAPI app
app = FastAPI()

# Mount the 'static' directory to serve the frontend
app.mount("/static", StaticFiles(directory="static"), name="static")

# --- Configuration ---
DB_FAISS_PATH = 'db'

# --- Load Pre-built Vector DB ---
embeddings = HuggingFaceEmbeddings(
    model_name='sentence-transformers/all-MiniLM-L6-v2',
    model_kwargs={'device': 'cpu'}
)
db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)

# --- Set up LLM and QA Chain ---
prompt_template = """Use the following pieces of context to answer the question at the end. Elaborate the answer so that it is easy to understand.
Context: {context}
Question: {question}
Answer:
"""
prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=os.environ["GOOGLE_API_KEY"])

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type='stuff',
    retriever=db.as_retriever(search_kwargs={'k': 2}),
    return_source_documents=True,
    chain_type_kwargs={'prompt': prompt}
)

# --- API Endpoints ---
class Query(BaseModel):
    question: str

@app.get("/")
async def read_index():
    return FileResponse('static/index.html')

@app.post("/ask")
def ask(query: Query):
    try:
        res = qa_chain.invoke({'query': query.question})
        answer = res["result"]
        # Note: Source document metadata might be minimal if not stored during indexing
        source_docs = res.get("source_documents", [])
        
        sources = []
        for doc in source_docs:
            sources.append(os.path.basename(doc.metadata.get('source', 'Unknown')))

        return {"answer": answer, "sources": sources}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))