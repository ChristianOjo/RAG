"""
Download sample research papers for testing
Fetches open-access papers from ArXiv
"""

import urllib.request
import os
from pathlib import Path


# Sample papers (famous ML/AI papers on ArXiv)
SAMPLE_PAPERS = {
    "attention_is_all_you_need.pdf": "https://arxiv.org/pdf/1706.03762.pdf",
    "bert.pdf": "https://arxiv.org/pdf/1810.04805.pdf",
    "gpt3.pdf": "https://arxiv.org/pdf/2005.14165.pdf",
    "resnet.pdf": "https://arxiv.org/pdf/1512.03385.pdf",
    "dropout.pdf": "https://arxiv.org/pdf/1207.0580.pdf",
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
    """Download all sample papers"""
    print("=" * 60)
    print("Downloading Sample Research Papers")
    print("=" * 60)
    print()
    
    # Create data directory
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    print(f"✓ Data directory: {data_dir.absolute()}\n")
    
    # Download papers
    downloaded = 0
    for filename, url in SAMPLE_PAPERS.items():
        if download_paper(url, filename, data_dir):
            downloaded += 1
        print()
    
    # Summary
    print("=" * 60)
    print(f"✓ Downloaded {downloaded}/{len(SAMPLE_PAPERS)} papers")
    print("=" * 60)
    print()
    
    if downloaded > 0:
        print("Next steps:")
        print("  1. Run: python src/ingest.py")
        print("  2. Run: streamlit run src/app.py")
        print()
        print("Or use your own papers by copying PDFs to data/")
    else:
        print("All papers already downloaded!")
    

if __name__ == "__main__":
    main()
