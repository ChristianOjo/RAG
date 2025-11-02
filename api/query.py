# D:\mini-rag-chatbot\api\query.py

from http.server import BaseHTTPRequestHandler
import json
import os
from typing import Dict, Any

# Import the RAGChatbot class from the file we will create next
from src.rag_core import RAGChatbot

# Initialize the chatbot globally for performance
try:
    GROQ_API_KEY = os.environ.get("GROQ_API_KEY" )
    if not GROQ_API_KEY:
        # This will raise an error if the key is not set in Vercel
        raise ValueError("GROQ_API_KEY environment variable not set.")
        
    # NOTE: The model name 'llama-3.2-3b-preview' is taken from your code.
    RAG_BOT = RAGChatbot(
        vectorstore_dir="vectorstore",
        groq_api_key=GROQ_API_KEY,
        model="llama-3.2-3b-preview",
        top_k=4,
        temperature=0.1
    )
except Exception as e:
    print(f"Error initializing RAGChatbot: {e}")
    RAG_BOT = None


def handler(request: BaseHTTPRequestHandler) -> Dict[str, Any]:
    """
    The entry point for the Vercel Serverless Function.
    """
    headers = {
        'Content-Type': 'application/json',
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Methods': 'POST, OPTIONS',
        'Access-Control-Allow-Headers': 'Content-Type'
    }

    if request.method == 'OPTIONS':
        return {'statusCode': 200, 'headers': headers, 'body': ''}

    if request.method != 'POST':
        return {'statusCode': 405, 'headers': headers, 'body': json.dumps({'error': 'Method Not Allowed'})}

    # Parse the request body
    try:
        body = json.loads(request.body)
        user_query = body.get('question')
        if not user_query:
            raise ValueError("Missing 'question' in request body.")
    except Exception as e:
        return {'statusCode': 400, 'headers': headers, 'body': json.dumps({'error': f'Invalid request body: {str(e)}'})}

    # Run the RAG pipeline
    try:
        if not RAG_BOT:
            raise Exception("RAG Chatbot not initialized.")
            
        response_data = RAG_BOT.answer(user_query, return_sources=True)
        
        return {
            'statusCode': 200,
            'headers': headers,
            'body': json.dumps(response_data)
        }
    except Exception as e:
        print(f"Unhandled RAG error: {e}")
        return {
            'statusCode': 500,
            'headers': headers,
            'body': json.dumps({'error': f'Internal Server Error: {str(e)}'})
        }
