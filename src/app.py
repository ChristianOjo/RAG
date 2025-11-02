"""
Streamlit Web UI for RAG Chatbot (Groq API version)
"""

import streamlit as st
from pathlib import Path
import sys
import os

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from chatbot_groq import RAGChatbot
from retrieval import Retriever


# Page config
st.set_page_config(
    page_title="Mini RAG Chatbot",
    page_icon="🤖",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .stChatMessage {
        padding: 1rem;
        border-radius: 0.5rem;
    }
    .source-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_chatbot(vectorstore_dir: str, model: str, top_k: int, api_key: str):
    """Load chatbot (cached)"""
    return RAGChatbot(
        vectorstore_dir=vectorstore_dir,
        groq_api_key=api_key,
        model=model,
        top_k=top_k
    )


def main():
    # Header
    st.title("🤖 Mini RAG Chatbot")
    st.markdown("*Ask questions about your research papers powered by Groq*")
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # API Key input
        api_key = st.text_input(
            "Groq API Key",
            type="password",
            value=os.getenv("GROQ_API_KEY", ""),
            help="Get your free API key at https://console.groq.com"
        )
        
        if not api_key:
            st.warning("⚠️ Please enter your Groq API key above")
            st.info("Get a free key at: https://console.groq.com")
            st.stop()
        
        vectorstore_dir = st.text_input(
            "Vector Store Directory",
            value="vectorstore",
            help="Directory containing the FAISS index"
        )
        
        model = st.selectbox(
            "LLM Model",
            options=[
                "llama-3.3-70b-versatile",
                "llama-3.1-8b-instant",
                "mixtral-8x7b-32768",
                "gemma2-9b-it"
            ],
            index=0,
            help="Groq model to use for generation"
        )
        
        top_k = st.slider(
            "Number of Retrieved Chunks",
            min_value=1,
            max_value=10,
            value=4,
            help="How many document chunks to retrieve"
        )
        
        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=0.1,
            step=0.1,
            help="Lower = more focused, Higher = more creative"
        )
        
        st.markdown("---")
        
        # Show stats
        if Path(vectorstore_dir).exists():
            try:
                retriever = Retriever(vectorstore_dir=vectorstore_dir)
                stats = retriever.get_stats()
                
                st.subheader("📊 Dataset Stats")
                st.metric("Total Chunks", stats['total_chunks'])
                st.metric("Chunk Size", stats['chunk_size'])
                st.metric("Chunk Overlap", stats['chunk_overlap'])
            except Exception as e:
                st.error(f"Could not load stats: {str(e)}")
        else:
            st.warning("⚠️ Vector store not found. Run `python src/ingest.py` first.")
        
        st.markdown("---")
        st.markdown("### 🚀 Quick Start")
        st.code("python src/ingest.py", language="bash")
        st.markdown("Get API key: [console.groq.com](https://console.groq.com)")
    
    # Main chat interface
    if not Path(vectorstore_dir).exists():
        st.error("❌ Vector store not found. Please run the ingestion script first.")
        st.code("python src/ingest.py --data-dir data --vectorstore-dir vectorstore", language="bash")
        return
    
    # Initialize chatbot
    try:
        with st.spinner("Loading chatbot..."):
            chatbot = load_chatbot(vectorstore_dir, model, top_k, api_key)
            chatbot.temperature = temperature
    except Exception as e:
        st.error(f"❌ Error loading chatbot: {str(e)}")
        st.info("Make sure your Groq API key is valid.")
        return
    
    # Chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Show sources if available
            if message["role"] == "assistant" and "sources" in message:
                with st.expander("📚 View Sources"):
                    for i, source in enumerate(message["sources"], 1):
                        st.markdown(f"""
                        <div class="source-box">
                            <strong>Source {i}</strong> (Relevance: {source['score']:.4f})<br>
                            <small>{source['source']}</small><br><br>
                            {source['content'][:300]}...
                        </div>
                        """, unsafe_allow_html=True)
    
    # Chat input
    if prompt := st.chat_input("Ask a question about your documents..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    result = chatbot.answer(prompt)
                    
                    # Display answer
                    st.markdown(result['answer'])
                    
                    # Store message with sources
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": result['answer'],
                        "sources": result['sources']
                    })
                    
                    # Show sources
                    with st.expander("📚 View Sources"):
                        for i, source in enumerate(result['sources'], 1):
                            st.markdown(f"""
                            <div class="source-box">
                                <strong>Source {i}</strong> (Relevance: {source['score']:.4f})<br>
                                <small>{source['source']}</small><br><br>
                                {source['content'][:300]}...
                            </div>
                            """, unsafe_allow_html=True)
                    
                except Exception as e:
                    error_msg = f"❌ Error: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg
                    })
    
    # Clear chat button
    if st.session_state.messages:
        if st.sidebar.button("🗑️ Clear Chat"):
            st.session_state.messages = []
            st.rerun()


if __name__ == "__main__":
    main()
