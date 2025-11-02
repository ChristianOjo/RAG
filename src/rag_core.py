# D:\mini-rag-chatbot\src\rag_core.py
# This file contains the RAGChatbot class, adapted from your chatbot_groq.py

from typing import List, Dict, Any
from groq import Groq
# NOTE: This assumes 'retrieval.py' is in the 'src' folder
from retrieval import Retriever 
import os

class RAGChatbot:
    def __init__(
        self,
        vectorstore_dir: str = "vectorstore",
        groq_api_key: str = None,
        model: str = "llama-3.2-3b-preview",
        top_k: int = 4,
        temperature: float = 0.1
    ):
        self.model = model
        self.temperature = temperature
        self.top_k = top_k
        
        # Initialize Groq client
        if not groq_api_key:
            # In Vercel, the key will be passed via the environment variable
            groq_api_key = os.getenv("GROQ_API_KEY")
            if not groq_api_key:
                raise ValueError("Please provide a Groq API key or set GROQ_API_KEY env var")
        
        self.client = Groq(api_key=groq_api_key)
        
        # Initialize retriever
        print("Initializing retriever...")
        # NOTE: Assuming Retriever is correctly defined in retrieval.py
        self.retriever = Retriever(vectorstore_dir=vectorstore_dir, top_k=top_k)
        
        print(f"✓ RAG Chatbot ready with {self.model}")
    
    def _create_prompt(self, query: str, context: str) -> str:
        """Create RAG prompt with context"""
        prompt = f"""You are a helpful research assistant. Answer the question based on the provided context from research papers.

Context from research papers:
{context}

Question: {query}

Instructions:
- Answer based ONLY on the information in the context above
- If the context doesn't contain enough information to answer, say so
- Be concise but comprehensive
- Cite specific details from the context when relevant

Answer:"""
        
        return prompt
    
    def answer(self, query: str, return_sources: bool = True) -> Dict[str, Any]:
        """
        Answer a question using RAG
        """
        print(f"\nProcessing query: {query}")
        
        # Step 1: Retrieve relevant documents
        print("Retrieving relevant documents...")
        results = self.retriever.search(query, k=self.top_k)
        
        if not results:
            return {
                'answer': "I couldn't find any relevant information in the documents.",
                'sources': [],
                'metadata': {'retrieved_chunks': 0}
            }
        
        # Step 2: Create context
        context = "\n\n".join([
            f"[Source {i+1}]: {r['content']}" 
            for i, r in enumerate(results)
        ])
        
        # Step 3: Generate answer with Groq
        print(f"Generating answer with {self.model}...")
        prompt = self._create_prompt(query, context)
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=512
            )
            
            answer = response.choices[0].message.content.strip()
            
        except Exception as e:
            answer = f"Error generating answer: {str(e)}"
            print(f"✗ Generation error: {str(e)}")
        
        # Step 4: Format response
        response_data = {
            'answer': answer,
            'sources': results if return_sources else [],
            'metadata': {
                'retrieved_chunks': len(results),
                'model': self.model,
                'query': query
            }
        }
        
        return response_data
