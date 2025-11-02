"""
RAG Chatbot using Groq API (Free Llama 3.2)
"""

from typing import List, Dict, Any
from groq import Groq
from retrieval import Retriever


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
            raise ValueError("Please provide a Groq API key")
        
        self.client = Groq(api_key=groq_api_key)
        
        # Initialize retriever
        print("Initializing retriever...")
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
        
        Args:
            query: User question
            return_sources: Whether to return source documents
            
        Returns:
            Dict with 'answer', 'sources', and 'metadata'
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
    
    def chat(self):
        """Interactive chat loop"""
        print("\n" + "=" * 50)
        print("RAG Chatbot - Interactive Mode")
        print("=" * 50)
        print("Type 'quit' or 'exit' to end the conversation\n")
        
        while True:
            try:
                query = input("You: ").strip()
                
                if query.lower() in ['quit', 'exit', 'q']:
                    print("Goodbye!")
                    break
                
                if not query:
                    continue
                
                # Get answer
                result = self.answer(query)
                
                # Print answer
                print(f"\nAssistant: {result['answer']}\n")
                
                # Print sources
                if result['sources']:
                    print(f"📚 Sources ({len(result['sources'])} documents):")
                    for i, source in enumerate(result['sources'], 1):
                        print(f"  [{i}] {source['source']} (score: {source['score']:.4f})")
                    print()
                
            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except Exception as e:
                print(f"\n✗ Error: {str(e)}\n")


def main():
    """CLI entry point"""
    import argparse
    import os
    
    parser = argparse.ArgumentParser(description="RAG Chatbot with Groq API")
    parser.add_argument("--vectorstore-dir", default="vectorstore", help="Vector store directory")
    parser.add_argument("--model", default="llama-3.2-3b-preview", help="Groq model name")
    parser.add_argument("--top-k", type=int, default=4, help="Number of chunks to retrieve")
    parser.add_argument("--temperature", type=float, default=0.1, help="Generation temperature")
    parser.add_argument("--query", help="Single query (non-interactive mode)")
    parser.add_argument("--api-key", help="Groq API key (or set GROQ_API_KEY env var)")
    
    args = parser.parse_args()
    
    # Get API key
    api_key = args.api_key or os.getenv("GROQ_API_KEY")
    if not api_key:
        print("Error: Please provide Groq API key via --api-key or GROQ_API_KEY environment variable")
        print("\nGet your free API key at: https://console.groq.com")
        return
    
    try:
        chatbot = RAGChatbot(
            vectorstore_dir=args.vectorstore_dir,
            groq_api_key=api_key,
            model=args.model,
            top_k=args.top_k,
            temperature=args.temperature
        )
        
        if args.query:
            # Single query mode
            result = chatbot.answer(args.query)
            print(f"\nQuestion: {args.query}")
            print(f"\nAnswer: {result['answer']}")
            
            if result['sources']:
                print(f"\nSources:")
                for i, source in enumerate(result['sources'], 1):
                    print(f"  [{i}] {source['source']}")
        else:
            # Interactive mode
            chatbot.chat()
            
    except Exception as e:
        print(f"✗ Error: {str(e)}")
        raise


if __name__ == "__main__":
    main()