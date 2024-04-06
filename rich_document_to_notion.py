import os
from typing import List, Optional
from dotenv import load_dotenv
from pathlib import Path
from argparse import ArgumentParser

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
    page_id: str,
    notion_api_key: Optional[str] = None,
    notion_version: Optional[str] = None,
): 
    if isinstance(output_dir, str):
        output_dir = Path(output_dir)

    if output_dir.exists() and not output_dir.is_dir():
        raise ValueError("Output directory must be a directory")

    if not output_dir.exists():
        os.makedirs(output_dir)

    if notion_api_key is None:
        notion_api_key = os.environ.get("NOTION_API_KEY", None)
        if notion_api_key is None:
            notion_api_key = input("Please provide the NOTION_API_KEY or set NOTION_API_KEY in the environment variables:")
    
    if notion_version is None:
        notion_version = os.environ.get("NOTION_VERSION", None)
        if notion_version is None:
            notion_version = "2022-06-28"
            logger.warning("Notion version not provided. Using 2022-06-28 as default version. Please provide the version as NOTION_VERSION in the environment variables.")

    notion_cli = NotionClient(
        notion_api_key=notion_api_key,
        notion_version=notion_version,
    )

    for rich_document_path in rich_document_paths:
        logger.info(f"Loading rich document from {rich_document_path}")
        rich_document = EDocument.from_json(rich_document_path)
        sci_paper = SciPaperPage(
            edoc=rich_document,
            page_id=page_id,
            notion=notion_cli,
        )

        logger.info("Writing to Notion")
        sci_paper.write_to_Notion()

    logger.info(f"Finished writing rich documents to {output_dir}")
    return None

def main():
    parser = ArgumentParser(description="Write eDocs to Notion")
    parser.add_argument("rich_document_paths", type=Path, nargs="+", help="Path to rich documents")
    parser.add_argument("-p", "--page_id", type=str, help="Page ID of the Notion page")
    parser.add_argument("-k", "--notion_api_key", type=str, help="Notion API key")
    parser.add_argument("-v", "--notion_version", type=str, help="Notion version", default="2022-06-28")

    args = parser.parse_args()

    run(
        rich_document_paths=args.rich_document_paths,
        page_id=args.page_id,
        notion_api_key=args.notion_api_key,
        notion_version=args.notion_version,
    )


if __name__ == "__main__":
    main()
