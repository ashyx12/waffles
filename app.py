from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from dotenv import load_dotenv

load_dotenv()

# Initialize the FastAPI app
app = FastAPI()

# --- Configuration ---
DB_FAISS_PATH = 'db'
MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'
LLM_MODEL = "gemini-pro"

# --- Load Embeddings and Vector DB ---
embeddings = HuggingFaceEmbeddings(model_name=MODEL_NAME, model_kwargs={'device': 'cpu'})
db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)

# --- Set up LLM and QA Chain ---
prompt_template = """Use the following pieces of context to answer the question at the end. Elaborate the answer so that it is easy to understand.
Context: {context}
Question: {question}
Answer:
"""
prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

llm = ChatGoogleGenerativeAI(model=LLM_MODEL, google_api_key=os.environ["GOOGLE_API_KEY"])

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
def read_root():
    return {"message": "Welcome to the RAG API!"}


@app.post("/ask")
def ask(query: Query):
    try:
        res = qa_chain.invoke({'query': query.question})
        answer = res["result"]
        source_docs = res["source_documents"]
        
        sources = []
        if source_docs:
            for doc in source_docs:
                sources.append(os.path.basename(doc.metadata.get('source', 'Unknown')))

        return {"answer": answer, "sources": sources}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))