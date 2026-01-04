"""
Organize Downloaded Documents by Source
"""
from pathlib import Path
import shutil

def organize_documents():
    """Organize downloaded documents by source"""
    
    raw_dir = Path("data/raw")
    organized_dir = Path("data/raw/organized")
    organized_dir.mkdir(parents=True, exist_ok=True)
    
    # Count documents
    pdf_files = list(raw_dir.rglob("*.pdf"))
    print(f"📄 Found {len(pdf_files)} PDF files")
    
    if len(pdf_files) == 0:
        print("⚠️  No PDF files found. Please download documents first.")
        return 0
    
    # Organize by source
    organized_count = 0
    for pdf in pdf_files:
        # Skip if already in organized folder
        if "organized" in str(pdf):
            continue
            
        # Determine source
        if "sec" in str(pdf).lower() or "edgar" in str(pdf).lower():
            dest_dir = organized_dir / "sec_filings"
        elif "github" in str(pdf).lower():
            dest_dir = organized_dir / "github"
        elif "kaggle" in str(pdf).lower():
            dest_dir = organized_dir / "kaggle"
        else:
            dest_dir = organized_dir / "other"
        
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / pdf.name
        
        # Copy file
        try:
            shutil.copy2(pdf, dest)
            organized_count += 1
        except Exception as e:
            print(f"⚠️  Error copying {pdf.name}: {e}")
    
    print(f"✅ Organized {organized_count} documents")
    print(f"📁 Organized files saved to: {organized_dir}")
    
    # Show organization
    print("\n📊 Organization Summary:")
    for source_dir in organized_dir.iterdir():
        if source_dir.is_dir():
            count = len(list(source_dir.glob("*.pdf")))
            print(f"   {source_dir.name}: {count} files")
    
    return organized_count

if __name__ == "__main__":
    count = organize_documents()
    print(f"\n✅ Total documents organized: {count}")

