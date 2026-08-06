import argparse
import sys
from pathlib import Path

from app.config.logging import logger
from app.ingestion.pipeline import IngestionPipeline

def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest legal documents into the intelligence platform.")
    parser.add_argument("path", type=str, help="Path to a PDF file or directory containing PDFs")
    parser.add_argument("--force", action="store_true", help="Force re-indexing of documents")
    
    args = parser.parse_args()
    input_path = Path(args.path)
    
    pipeline = IngestionPipeline()
    
    if input_path.is_file():
        if input_path.suffix.lower() != ".pdf":
            logger.error(f"Error: {input_path} is not a PDF file.")
            sys.exit(1)
        try:
            chunks = pipeline.process_document(input_path, force_reindex=args.force)
            print(f"Successfully processed {input_path.name}: generated {len(chunks)} chunks.")
        except Exception as e:
            logger.error(f"Failed to process {input_path.name}: {e}")
            sys.exit(1)
            
    elif input_path.is_dir():
        pdf_files = list(input_path.glob("**/*.pdf"))
        if not pdf_files:
            logger.warning(f"No PDF files found in directory {input_path}")
            return
            
        logger.info(f"Found {len(pdf_files)} PDF files in {input_path}")
        successful = 0
        for pdf_file in pdf_files:
            try:
                chunks = pipeline.process_document(pdf_file, force_reindex=args.force)
                successful += 1
            except Exception as e:
                logger.error(f"Failed to process {pdf_file.name}: {e}")
                
        print(f"Batch processing completed. Processed {successful}/{len(pdf_files)} documents.")
    else:
        logger.error(f"Error: Path {input_path} does not exist.")
        sys.exit(1)

if __name__ == "__main__":
    main()
    
