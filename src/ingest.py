"""
Document Ingestion Pipeline
Loads PDFs, chunks them, creates embeddings, and stores in FAISS
"""

import os
from pathlib import Path
from typing import List
import pickle

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings


class DocumentIngestor:
    def __init__(
        self,
        data_dir: str = "data",
        vectorstore_dir: str = "vectorstore",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    ):
        self.data_dir = Path(data_dir)
        self.vectorstore_dir = Path(vectorstore_dir)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Initialize embeddings (runs locally, no API needed)
        print(f"Loading embedding model: {embedding_model}")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={'device': 'cpu'}
        )
        
        # Text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
    def load_documents(self) -> List:
        """Load all PDF documents from data directory"""
        print(f"Loading documents from {self.data_dir}...")
        
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")
        
        # Load PDFs
        loader = DirectoryLoader(
            str(self.data_dir),
            glob="**/*.pdf",
            loader_cls=PyPDFLoader,
            show_progress=True
        )
        
        documents = loader.load()
        print(f"Loaded {len(documents)} document pages")
        
        return documents
    
    def chunk_documents(self, documents: List) -> List:
        """Split documents into chunks"""
        print("Chunking documents...")
        chunks = self.text_splitter.split_documents(documents)
        print(f"Created {len(chunks)} chunks")
        
        return chunks
    
    def create_vectorstore(self, chunks: List):
        """Create FAISS vector store from chunks"""
        print("Creating embeddings and vector store...")
        print("This may take a few minutes depending on document size...")
        
        vectorstore = FAISS.from_documents(
            documents=chunks,
            embedding=self.embeddings
        )
        
        return vectorstore
    
    def save_vectorstore(self, vectorstore):
        """Save vector store to disk"""
        self.vectorstore_dir.mkdir(exist_ok=True, parents=True)
        
        print(f"Saving vector store to {self.vectorstore_dir}...")
        vectorstore.save_local(str(self.vectorstore_dir))
        
        # Save metadata
        metadata = {
            'num_chunks': vectorstore.index.ntotal,
            'chunk_size': self.chunk_size,
            'chunk_overlap': self.chunk_overlap
        }
        
        with open(self.vectorstore_dir / "metadata.pkl", "wb") as f:
            pickle.dump(metadata, f)
        
        print(f"✓ Vector store saved with {metadata['num_chunks']} chunks")
    
    def ingest(self):
        """Run full ingestion pipeline"""
        print("=" * 50)
        print("Starting Document Ingestion Pipeline")
        print("=" * 50)
        
        try:
            # Step 1: Load documents
            documents = self.load_documents()
            
            if len(documents) == 0:
                print("⚠ No documents found. Please add PDF files to the data/ directory")
                return
            
            # Step 2: Chunk documents
            chunks = self.chunk_documents(documents)
            
            # Step 3: Create vector store
            vectorstore = self.create_vectorstore(chunks)
            
            # Step 4: Save to disk
            self.save_vectorstore(vectorstore)
            
            print("\n" + "=" * 50)
            print("✓ Ingestion Complete!")
            print("=" * 50)
            
        except Exception as e:
            print(f"\n✗ Error during ingestion: {str(e)}")
            raise


def main():
    """CLI entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Ingest documents into vector store")
    parser.add_argument("--data-dir", default="data", help="Directory containing PDF files")
    parser.add_argument("--vectorstore-dir", default="vectorstore", help="Output directory for vector store")
    parser.add_argument("--chunk-size", type=int, default=1000, help="Chunk size for text splitting")
    parser.add_argument("--chunk-overlap", type=int, default=200, help="Overlap between chunks")
    
    args = parser.parse_args()
    
    ingestor = DocumentIngestor(
        data_dir=args.data_dir,
        vectorstore_dir=args.vectorstore_dir,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap
    )
    
    ingestor.ingest()


if __name__ == "__main__":
    main()
