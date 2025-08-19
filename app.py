from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

load_dotenv()

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

DB_FAISS_PATH = 'db'

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)

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
        source_docs = res.get("source_documents", [])
        sources = [os.path.basename(doc.metadata.get('source', 'Unknown')) for doc in source_docs]
        return {"answer": answer, "sources": sources}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
