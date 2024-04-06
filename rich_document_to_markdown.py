import os
from typing import List, Optional
from dotenv import load_dotenv
from pathlib import Path
import argparse


# check if .env file exists
print(load_dotenv())


from notion_interface.notion_cli import NotionClient
from notion_interface.pages.scientific_page_handler import SciPaperPage
from rich_docs.edocument_class import EDocument

# set logging level
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run(
    rich_document_paths: List[Path],
    output_dir: Path,
): 
    if isinstance(output_dir, str):
        output_dir = Path(output_dir)
    
    if output_dir.exists() and not output_dir.is_dir():
        raise ValueError("Output directory must be a directory")
    
    if not output_dir.exists():
        os.makedirs(output_dir)
        
    for rich_document_path in rich_document_paths:
        logger.info(f"Loading rich document from {rich_document_path}")
        rich_document = EDocument.from_json(rich_document_path)
        rich_document.write(output_dir, include_references=False)
    
    logger.info(f"Finished writing rich documents to {output_dir}")
    
    return None

def main():
    parser = argparse.ArgumentParser(description="Write eDocs to markdown")
    parser.add_argument("rich_document_paths", type=Path, nargs="+", help="Path to rich documents")
    parser.add_argument("-o", "--output_dir", type=Path, help="Output directory", default="output")
    
    args = parser.parse_args()
    
    run(
        rich_document_paths=args.rich_document_paths,
        output_dir=args.output_dir,
    )
    
    return None

if __name__ == "__main__":
    main()