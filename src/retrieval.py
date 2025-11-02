"""
Retrieval Module
Handles semantic search and document retrieval from FAISS
"""

from pathlib import Path
from typing import List, Dict, Any
import pickle

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings


class Retriever:
    def __init__(
        self,
        vectorstore_dir: str = "vectorstore",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        top_k: int = 4
    ):
        self.vectorstore_dir = Path(vectorstore_dir)
        self.top_k = top_k
        
        # Load embeddings (same model used during ingestion)
        print(f"Loading embedding model: {embedding_model}")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={'device': 'cpu'}
        )
        
        # Load vector store
        self.vectorstore = self._load_vectorstore()
        self.metadata = self._load_metadata()
        
    def _load_vectorstore(self):
        """Load FAISS vector store from disk"""
        if not self.vectorstore_dir.exists():
            raise FileNotFoundError(
                f"Vector store not found at {self.vectorstore_dir}. "
                "Please run ingest.py first."
            )
        
        print(f"Loading vector store from {self.vectorstore_dir}...")
        vectorstore = FAISS.load_local(
            str(self.vectorstore_dir),
            self.embeddings,
            allow_dangerous_deserialization=True
        )
        
        print(f"✓ Vector store loaded with {vectorstore.index.ntotal} chunks")
        return vectorstore
    
    def _load_metadata(self) -> Dict[str, Any]:
        """Load metadata about the vector store"""
        metadata_path = self.vectorstore_dir / "metadata.pkl"
        
        if metadata_path.exists():
            with open(metadata_path, "rb") as f:
                return pickle.load(f)
        
        return {}
    
    def search(self, query: str, k: int = None) -> List[Dict[str, Any]]:
        """
        Perform semantic search
        
        Args:
            query: Search query
            k: Number of results to return (default: self.top_k)
            
        Returns:
            List of dicts with 'content', 'metadata', and 'score'
        """
        if k is None:
            k = self.top_k
        
        # Perform similarity search with scores
        results = self.vectorstore.similarity_search_with_score(query, k=k)
        
        # Format results
        formatted_results = []
        for doc, score in results:
            formatted_results.append({
                'content': doc.page_content,
                'metadata': doc.metadata,
                'score': float(score),
                'source': doc.metadata.get('source', 'Unknown')
            })
        
        return formatted_results
    
    def get_context(self, query: str, k: int = None) -> str:
        """
        Get concatenated context from top-k results
        
        Args:
            query: Search query
            k: Number of results to return
            
        Returns:
            Concatenated text from top results
        """
        results = self.search(query, k=k)
        
        context_parts = []
        for i, result in enumerate(results, 1):
            context_parts.append(f"[Document {i}]\n{result['content']}")
        
        return "\n\n".join(context_parts)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store"""
        return {
            'total_chunks': self.vectorstore.index.ntotal,
            'chunk_size': self.metadata.get('chunk_size', 'Unknown'),
            'chunk_overlap': self.metadata.get('chunk_overlap', 'Unknown'),
            'embedding_model': 'sentence-transformers/all-MiniLM-L6-v2'
        }


def main():
    """CLI entry point for testing retrieval"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test retrieval system")
    parser.add_argument("query", help="Search query")
    parser.add_argument("--k", type=int, default=4, help="Number of results")
    parser.add_argument("--vectorstore-dir", default="vectorstore", help="Vector store directory")
    
    args = parser.parse_args()
    
    try:
        retriever = Retriever(
            vectorstore_dir=args.vectorstore_dir,
            top_k=args.k
        )
        
        print(f"\nSearching for: '{args.query}'")
        print("=" * 50)
        
        results = retriever.search(args.query)
        
        for i, result in enumerate(results, 1):
            print(f"\n[Result {i}] (Score: {result['score']:.4f})")
            print(f"Source: {result['source']}")
            print(f"Content: {result['content'][:200]}...")
            print("-" * 50)
        
        print(f"\n✓ Found {len(results)} relevant chunks")
        
    except Exception as e:
        print(f"✗ Error: {str(e)}")
        raise


if __name__ == "__main__":
    main()
