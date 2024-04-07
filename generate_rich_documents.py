from typing import List, Optional, Tuple, Literal
from pathlib import Path
import logging
import pypdf
from pprint import pprint
import argparse
import os


from extract_OCR.main_nougat import NougatOCR
from extract_OCR.helper_classes.nougat_model_configs import LocalSettings, Configs

from extract_OCR.main import ExtractOCR


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_input_path(str_path: str) -> Path:
    if isinstance(str_path, Path):
        str_path = str(str_path)
    if str_path[:2] in ('r"', 'r"'):
        str_path = str_path[2:-1]
    if str_path[:1] in ('"', "'"):
        str_path = str_path[1:-1]
    str_path = r"{}".format(str_path)
    return Path(str_path)

def run(
    pdf_paths: List[Path], 
    output_db: Path, 
    start_from_scratch: bool = True, 
    model_directory: Optional[str | Path] = None,
    model_size: Optional[Literal["small", "base"]] = "small",
    batch_size: int = 1
    ) -> None:

    if model_directory is None:
        model_directory = os.getenv("NOUGAT_MODEL_DIR")

    if model_directory is None:
        model_directory = input("Please provide the model directory or set the NOUGAT_MODEL_DIR environment variable: ")
    
    model_directory = get_input_path(model_directory)
    output_db = get_input_path(output_db)
    pdf_paths = [get_input_path(pdf_path) for pdf_path in pdf_paths]
    
    # check the model directory exists
    if not model_directory.exists() or not model_directory.is_dir():
        raise FileNotFoundError(f"Model directory {model_directory} does not exist")


    if not isinstance(model_size, str) or model_size not in ["small", "base"]:
        raise ValueError("Model size must be either 'small' or 'base'")

    local_settings = LocalSettings(
        model_directory=model_directory,
        model_type=model_size,
    )

    configs = Configs(batch_size=batch_size, allow_skipping=False)

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
    parser.add_argument("-o", "--output_db", type=Path, default=".", help="Output database directory")
    parser.add_argument("--start_from_scratch", default=True, action="store_true", help="Start from scratch (default: True)")
    parser.add_argument("--model_directory", type=Path, help="Model directory (optional)")
    parser.add_argument("--model_size", type=str, default="small", choices=["small", "base"], help="Model size (optional)")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size (default: 1)")

    args = parser.parse_args()
    
    if args.pdf_paths is None:
        raise ValueError("Please provide at least one PDF file path")
    
    pdf_paths = [Path(pdf_path) for pdf_path in args.pdf_paths] 
    if len(pdf_paths) == 1:
        if pdf_paths[0].is_dir():
            pdf_paths = list(pdf_paths[0].glob("*.pdf"))

    run(
        pdf_paths=args.pdf_paths,
        output_db=args.output_db,
        start_from_scratch=args.start_from_scratch,
        model_directory=args.model_directory,
        model_size=args.model_size,
        batch_size=args.batch_size
    )

if __name__ == "__main__":
    main()