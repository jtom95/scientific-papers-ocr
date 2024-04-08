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
from helper_functions.helpers_for_main import get_input_path

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
    parser.add_argument("-o", "--output_dir", type=Path, help="Output directory", default=".")

    args = parser.parse_args()

    if args.rich_document_paths is None:
        raise ValueError("Please provide at least one PDF file path")
    
    rich_document_paths = [
        get_input_path(pdf_path) for pdf_path in args.rich_document_paths
        ]
    if len(rich_document_paths) == 1:
        if rich_document_paths[0].is_dir():
            rich_document_paths = list(rich_document_paths[0].rglob("*.json"))

    output_dir = get_input_path(args.output_dir)

    run(
        rich_document_paths=rich_document_paths,
        output_dir=output_dir,
    )

    return None

if __name__ == "__main__":
    main()
