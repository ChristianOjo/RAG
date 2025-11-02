"""
Download psychology research papers from arXiv and PubMed
"""

import urllib.request
from pathlib import Path


# Psychology papers from arXiv (open access)
PSYCHOLOGY_PAPERS = {
    "attention_cognitive_control.pdf": "https://arxiv.org/pdf/1910.03771.pdf",
    "memory_consolidation.pdf": "https://arxiv.org/pdf/1906.05433.pdf",
    "neural_networks_cognition.pdf": "https://arxiv.org/pdf/1807.03748.pdf",
    "decision_making_ai.pdf": "https://arxiv.org/pdf/1906.02736.pdf",
    "reinforcement_learning_psychology.pdf": "https://arxiv.org/pdf/1805.00909.pdf",
}


def download_paper(url: str, filename: str, data_dir: Path):
    """Download a single paper"""
    filepath = data_dir / filename
    
    if filepath.exists():
        print(f"⚠ {filename} already exists, skipping...")
        return False
    
    try:
        print(f"Downloading {filename}...")
        urllib.request.urlretrieve(url, filepath)
        size_mb = filepath.stat().st_size / (1024 * 1024)
        print(f"✓ Downloaded {filename} ({size_mb:.1f} MB)")
        return True
    except Exception as e:
        print(f"✗ Failed to download {filename}: {str(e)}")
        return False


def main():
    print("=" * 60)
    print("Downloading Psychology Research Papers")
    print("=" * 60)
    print()
    
    # Create data directory
    data_dir = Path("data_psychology")
    data_dir.mkdir(exist_ok=True)
    print(f"✓ Data directory: {data_dir.absolute()}\n")
    
    # Download papers
    downloaded = 0
    for filename, url in PSYCHOLOGY_PAPERS.items():
        if download_paper(url, filename, data_dir):
            downloaded += 1
        print()
    
    # Summary
    print("=" * 60)
    print(f"✓ Downloaded {downloaded}/{len(PSYCHOLOGY_PAPERS)} papers")
    print("=" * 60)
    print()
    
    if downloaded > 0:
        print("Next steps:")
        print("  1. Run: python src/ingest.py --data-dir data_psychology --vectorstore-dir vectorstore_psychology")
        print("  2. Run: python -m streamlit run src/app.py")
        print("  3. In the sidebar, change Vector Store Directory to 'vectorstore_psychology'")
    

if __name__ == "__main__":
    main()