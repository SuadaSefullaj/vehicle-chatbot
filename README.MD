# 🚗 Vehicle Chatbot API
This is a FastAPI-based backend application that enables users to upload PDF documents related to German vehicles, stores their content as vector embeddings in Pinecone, and uses OpenAI's GPT-4 model to answer vehicle-related questions by retrieving contextual information from the PDFs.

## 🚀 Features
Upload and process PDF documents via an API

Store extracted text embeddings in Pinecone

Retrieve relevant context using vector similarity search

Stream GPT-4 generated responses to user questions with contextual knowledge

S3 integration for document storage

OpenAI Chat API integration

CORS enabled for frontend interaction

## 🛠️ Technologies Used
FastAPI

OpenAI GPT-4

Pinecone Vector DB

Amazon S3 (Boto3)

PyMuPDF

Uvicorn

dotenv

## 📦 Installation
Clone the Repository:

```bash
git clone https://github.com/SuadaSefullaj/vehicle-chatbot.git
cd vehicle-chatbot-api
```
Create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```
Install dependencies:

```bash
pip install -r requirements.txt
```
Create a .env file with the following variables:

```bash
OPENAI_API_KEY=your_openai_api_key
PINECONE_API_KEY=your_pinecone_api_key
```
