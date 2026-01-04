"""
Download Sample Legal Documents for Training
"""
import os
import requests
from pathlib import Path
import time

def download_sample_contracts():
    """Download sample legal documents for testing"""
    
    # Create data directory
    data_dir = Path("data/raw")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"✅ Created directory: {data_dir}")
    print("\n📋 Manual Download Instructions:")
    print("=" * 60)
    print("\n1. SEC EDGAR (Company Filings):")
    print("   URL: https://www.sec.gov/edgar/searchedgar/companysearch.html")
    print("   Steps:")
    print("   - Search for companies (Apple, Microsoft, Google, etc.)")
    print("   - Click on company → View filings")
    print("   - Download 10-K or 10-Q filings (PDF format)")
    print("   - Save to: data/raw/sec_filings/")
    
    print("\n2. GitHub Legal Contracts:")
    print("   URL: https://github.com/search")
    print("   Steps:")
    print("   - Search: 'legal contract pdf' OR 'employment agreement pdf'")
    print("   - Filter: File extension → .pdf")
    print("   - Download contracts")
    print("   - Save to: data/raw/github_contracts/")
    
    print("\n3. Kaggle Legal Datasets:")
    print("   URL: https://www.kaggle.com/datasets")
    print("   Search: 'legal documents', 'contracts', 'legal text'")
    print("   Save to: data/raw/kaggle/")
    
    print("\n4. Public Legal Document Repositories:")
    print("   - CourtListener: https://www.courtlistener.com/")
    print("   - Legal Information Institute: https://www.law.cornell.edu/")
    
    print("\n" + "=" * 60)
    print(f"\n💡 Target: Collect 1,000+ PDF documents")
    print(f"📁 Save all PDFs to: {data_dir}/")
    
    return data_dir

if __name__ == "__main__":
    download_sample_contracts()

