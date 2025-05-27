import os
import requests
import fitz
import openai
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import asyncio
import uvicorn
import time
import boto3
from botocore.exceptions import ClientError
from pinecone import Pinecone, ServerlessSpec

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")
app = FastAPI()

s3 = boto3.client('s3', region_name='eu-central-1')
bucket_name = "vehicle-chatbot-docs"

class QuestionRequest(BaseModel):
    question: str

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["chat_id"],
)

pc = Pinecone(api_key=pinecone_api_key)
index_name = "pdf-embeddings"
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=1536,
        metric='cosine',
        spec=ServerlessSpec(cloud='aws', region='us-east-1')
    )
    while not pc.describe_index(index_name).status['ready']:
        time.sleep(1)

index = pc.Index(index_name)

if not os.path.exists('data'):
    os.makedirs('data')

def extract_text_from_pdf_s3(bucket_name: str, file_name: str) -> str:
    response = s3.get_object(Bucket=bucket_name, Key=file_name)
    pdf_data = response['Body'].read()
    doc = fitz.open(stream=pdf_data, filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def get_embedding(text: str, model="text-embedding-ada-002"):
    response = openai.embeddings.create(model=model, input=text)
    return response.data[0].embedding

def store_embeddings_in_pinecone(file_name: str, text: str):
    chunk_size = 300
    chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
    embeddings = [get_embedding(chunk) for chunk in chunks]
    ids = [f"{file_name}_{i}" for i in range(len(chunks))]
    vectors = [
        {"id": ids[i], "values": embeddings[i], "metadata": {"text": chunks[i]}}
        for i in range(len(chunks))
    ]
    index.upsert(vectors=vectors)

@app.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    try:
        if not file.filename.endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Only PDF files are allowed.")
        s3.upload_fileobj(file.file, bucket_name, file.filename)
        text = extract_text_from_pdf_s3(bucket_name, file.filename)
        store_embeddings_in_pinecone(file.filename, text)
        return {"status": "success", "detail": f"File {file.filename} processed and embeddings stored in Pinecone."}
    except ClientError as e:
        raise HTTPException(status_code=500, detail=f"S3 upload error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

conversation_history = []

def get_vehicle_info(question: str) -> str:
    url = "https://api.openai.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    messages = [{"role": "system", "content": "You are a personal German vehicle information assistant. Answer questions about German vehicles, including specifications, history, and models."}] + conversation_history + [{"role": "user", "content": question}]
    data = {"model": "gpt-4", "messages": messages}
    response = requests.post(url, headers=headers, json=data)
    if response.status_code == 200:
        return response.json()['choices'][0]['message']['content']
    else:
        raise HTTPException(status_code=response.status_code, detail=response.text)

@app.post("/question")
async def handle_question(question_request: QuestionRequest):
    async def stream_answer_with_context():
        user_message = {"role": "user", "content": question_request.question}
        question_embedding = get_embedding(question_request.question)
        result = index.query(vector=question_embedding, top_k=3, include_metadata=True)
        context = ""
        for match in result['matches']:
            if 'metadata' in match and 'text' in match['metadata']:
                context += match['metadata']['text'] + " "
        if not context:
            context = "No relevant context found in the knowledge base."
        full_prompt = f"You have access to the following context from a knowledge base:\n{context}\nNow answer the user's question: {question_request.question}"
        conversation_history.append(user_message)
        conversation_history.append({"role": "system", "content": context})
        answer = await asyncio.to_thread(get_vehicle_info, full_prompt)
        assistant_message = {"role": "assistant", "content": answer}
        conversation_history.append(assistant_message)
        chunks = answer.split('. ')
        for i, chunk in enumerate(chunks):
            yield chunk + (". " if i < len(chunks) - 1 else "")
            await asyncio.sleep(0.5)
    return StreamingResponse(stream_answer_with_context(), media_type="text/plain")

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
