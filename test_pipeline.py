"""
Test script for the entire RAG pipeline
Run this to verify everything works end-to-end
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from ingest import DocumentIngestor
from retrieval import Retriever
from chatbot import RAGChatbot


def test_ingestion():
    """Test document ingestion"""
    print("\n" + "=" * 60)
    print("TEST 1: Document Ingestion")
    print("=" * 60)
    
    try:
        data_dir = Path("data")
        
        # Check if data directory has PDFs
        pdf_files = list(data_dir.glob("*.pdf"))
        if not pdf_files:
            print("⚠ No PDF files found in data/ directory")
            print("Please add some PDF files to test ingestion")
            return False
        
        print(f"✓ Found {len(pdf_files)} PDF files")
        
        # Run ingestion
        ingestor = DocumentIngestor(
            data_dir="data",
            vectorstore_dir="vectorstore_test",
            chunk_size=1000,
            chunk_overlap=200
        )
        
        ingestor.ingest()
        
        print("✓ Ingestion test passed!")
        return True
        
    except Exception as e:
        print(f"✗ Ingestion test failed: {str(e)}")
        return False


def test_retrieval():
    """Test retrieval system"""
    print("\n" + "=" * 60)
    print("TEST 2: Retrieval System")
    print("=" * 60)
    
    try:
        # Load retriever
        retriever = Retriever(
            vectorstore_dir="vectorstore_test",
            top_k=4
        )
        
        # Test queries
        test_queries = [
            "What is the main topic?",
            "Explain the methodology",
            "What are the results?"
        ]
        
        all_passed = True
        
        for query in test_queries:
            print(f"\nQuery: '{query}'")
            results = retriever.search(query, k=2)
            
            if len(results) > 0:
                print(f"✓ Retrieved {len(results)} chunks")
                print(f"  Best score: {results[0]['score']:.4f}")
            else:
                print("✗ No results returned")
                all_passed = False
        
        if all_passed:
            print("\n✓ Retrieval test passed!")
            return True
        else:
            print("\n✗ Some retrieval tests failed")
            return False
            
    except Exception as e:
        print(f"✗ Retrieval test failed: {str(e)}")
        return False


def test_chatbot():
    """Test RAG chatbot"""
    print("\n" + "=" * 60)
    print("TEST 3: RAG Chatbot")
    print("=" * 60)
    
    try:
        # Initialize chatbot
        chatbot = RAGChatbot(
            vectorstore_dir="vectorstore_test",
            model="llama3.2",
            top_k=4
        )
        
        # Test question
        test_question = "What is this document about?"
        
        print(f"\nAsking: '{test_question}'")
        result = chatbot.answer(test_question)
        
        print(f"\n✓ Got answer ({len(result['answer'])} chars)")
        print(f"✓ Retrieved {len(result['sources'])} source chunks")
        
        # Check answer quality
        if len(result['answer']) > 20:
            print(f"\nAnswer preview: {result['answer'][:200]}...")
            print("\n✓ Chatbot test passed!")
            return True
        else:
            print("✗ Answer too short, might be an error")
            return False
            
    except Exception as e:
        print(f"✗ Chatbot test failed: {str(e)}")
        print("\nNote: Make sure Ollama is running and llama3.2 is installed:")
        print("  ollama pull llama3.2")
        return False


def test_failure_cases():
    """Test known failure cases and fixes"""
    print("\n" + "=" * 60)
    print("TEST 4: Failure Cases")
    print("=" * 60)
    
    try:
        chatbot = RAGChatbot(
            vectorstore_dir="vectorstore_test",
            model="llama3.2"
        )
        
        # Test 1: Out-of-context question
        print("\nFailure Case 1: Out-of-context question")
        ooc_question = "What is the capital of France?"
        result = chatbot.answer(ooc_question)
        
        # Check if model admits it doesn't know
        answer_lower = result['answer'].lower()
        if any(phrase in answer_lower for phrase in ['cannot find', 'not in', 'don\'t have', 'no information']):
            print("✓ Model correctly identified out-of-context question")
        else:
            print("⚠ Model may have hallucinated (answered out-of-context)")
        
        # Test 2: Very specific question
        print("\nFailure Case 2: Very specific question")
        specific_question = "What was the exact value of hyperparameter X in experiment Y?"
        result = chatbot.answer(specific_question)
        
        if result['sources']:
            print(f"✓ Retrieved {len(result['sources'])} relevant sources")
        
        print("\n✓ Failure case tests completed!")
        return True
        
    except Exception as e:
        print(f"✗ Failure case tests failed: {str(e)}")
        return False


def cleanup():
    """Clean up test files"""
    import shutil
    
    print("\n" + "=" * 60)
    print("Cleanup")
    print("=" * 60)
    
    test_dir = Path("vectorstore_test")
    if test_dir.exists():
        shutil.rmtree(test_dir)
        print("✓ Cleaned up test files")


def main():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("Mini RAG Chatbot - Pipeline Test Suite")
    print("=" * 60)
    
    results = {
        'Ingestion': False,
        'Retrieval': False,
        'Chatbot': False,
        'Failure Cases': False
    }
    
    # Run tests
    results['Ingestion'] = test_ingestion()
    
    if results['Ingestion']:
        results['Retrieval'] = test_retrieval()
        results['Chatbot'] = test_chatbot()
        results['Failure Cases'] = test_failure_cases()
    
    # Cleanup
    cleanup()
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{test_name:20} {status}")
    
    total_passed = sum(results.values())
    total_tests = len(results)
    
    print(f"\nTotal: {total_passed}/{total_tests} tests passed")
    
    if total_passed == total_tests:
        print("\n🎉 All tests passed! Your RAG system is working perfectly.")
        return 0
    else:
        print("\n⚠ Some tests failed. Check the output above for details.")
        return 1


if __name__ == "__main__":
    exit(main())
