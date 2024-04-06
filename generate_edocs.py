from typing import List, Optional, Tuple
from pathlib import Path
import logging
import pypdf
from pprint import pprint
import argparse


from extract_OCR.main_nougat import NougatOCR
from extract_OCR.elaborate_mmd_to_edoc.elaborate_batch_mmd_scientific import ElaborateMarkdownBatch
from extract_OCR.helper_classes.nougat_model_configs import LocalSettings, Configs
from rich_docs.edocument_class import EDocument

from extract_OCR.main import ExtractOCR

from collections import namedtuple
from matplotlib import patches

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run(
    pdf_paths: List[Path], 
    output_db: Path, 
    start_from_scratch: bool = False, 
    model_directory: Optional[str | Path] = None,
    model_size: Optional[str] = None
    ) -> None:
    if model_size is None:
        if model_directory is not None:
            model_directory = Path(model_directory) 
            local_settings = LocalSettings(model_directory=model_directory)
        else:
            local_settings = LocalSettings()
    else: 
        if model_directory is not None:
            model_directory = Path(model_directory)
            local_settings = LocalSettings(model_directory=model_directory)
        else:
            local_settings = LocalSettings()
            
    configs = Configs(batch_size=1, allow_skipping=False)

    nougat_ocr = NougatOCR(
        local_settings=local_settings,
        configs=configs,
    )

    extractor = ExtractOCR(
        nougat=nougat_ocr,
        pdfs=pdf_paths,
        output_dir=output_db,
    )
    logger.info("Starting the process")
    extractor.generate_structured_markdown_database(start_from_scratch=start_from_scratch)
    logger.info("Checking predictions")
    extractor.check_predictions()
    logger.info("Running Nougat on incomplete inferences")
    extractor.run_nougat_on_incomplete()
    logger.info("Generating EDoc database from markdown database")
    extractor.generate_Edoc_db_from_markdown_db(
        markdown_db=output_db,
        pdf_paths=pdf_paths,
        edoc_dir=output_db,
    )
    logger.info("Done")


def main():
    parser = argparse.ArgumentParser(description="Extract OCR and generate EDocument database")
    parser.add_argument("pdf_paths", nargs="+", type=Path, help="List of PDF file paths")
    parser.add_argument("output_db", type=Path, default=".", help="Output database directory")
    parser.add_argument("--start_from_scratch", action="store_true", help="Start from scratch (default: False)")
    parser.add_argument("--model_directory", type=Path, help="Model directory (optional)")
    parser.add_argument("--model_size", type=str, help="Model size (optional)")

    args = parser.parse_args()

    run(
        pdf_paths=args.pdf_paths,
        output_db=args.output_db,
        start_from_scratch=args.start_from_scratch,
        model_directory=args.model_directory,
        model_size=args.model_size
    )
    
if __name__ == "__main__":
    main()
