#!/bin/bash

# Mini RAG Chatbot Setup Script
# Run this to set up your environment

set -e  # Exit on error

echo "=================================="
echo "Mini RAG Chatbot Setup"
echo "=================================="
echo ""

# Check Python version
echo "[1/6] Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "✓ Found Python $python_version"

# Create virtual environment
echo ""
echo "[2/6] Creating virtual environment..."
if [ -d "venv" ]; then
    echo "⚠ venv already exists, skipping..."
else
    python3 -m venv venv
    echo "✓ Virtual environment created"
fi

# Activate virtual environment
echo ""
echo "[3/6] Activating virtual environment..."
source venv/bin/activate
echo "✓ Virtual environment activated"

# Install dependencies
echo ""
echo "[4/6] Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt
echo "✓ Dependencies installed"

# Create directory structure
echo ""
echo "[5/6] Creating directory structure..."
mkdir -p data
mkdir -p vectorstore
mkdir -p notebooks
echo "✓ Directories created"

# Check Ollama
echo ""
echo "[6/6] Checking Ollama installation..."
if command -v ollama &> /dev/null; then
    echo "✓ Ollama is installed"
    
    # Check if llama3.2 is available
    if ollama list | grep -q "llama3.2"; then
        echo "✓ llama3.2 model is installed"
    else
        echo "⚠ llama3.2 not found"
        echo ""
        echo "To install llama3.2, run:"
        echo "  ollama pull llama3.2"
    fi
else
    echo "⚠ Ollama not found"
    echo ""
    echo "Please install Ollama:"
    echo "  macOS/Linux: https://ollama.ai/download"
    echo "  Then run: ollama pull llama3.2"
fi

echo ""
echo "=================================="
echo "Setup Complete! 🎉"
echo "=================================="
echo ""
echo "Next steps:"
echo "  1. Add PDF files to the 'data/' directory"
echo "  2. Run: python src/ingest.py"
echo "  3. Run: streamlit run src/app.py"
echo ""
echo "Or try the notebook:"
echo "  jupyter notebook notebooks/demo.ipynb"
echo ""