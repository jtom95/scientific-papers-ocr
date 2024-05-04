from typing import List, Optional, Tuple, Literal, Union
from pathlib import Path
import logging
import pypdf
from pprint import pprint
import argparse
import os

from helper_functions.helpers_for_main import get_input_path
from extract_OCR.main_nougat import NougatOCR
from extract_OCR.helper_classes.nougat_model_configs import LocalSettings, Configs

from extract_OCR.main import ExtractOCR


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
    parser.add_argument("-o", "--output_dir", type=Union[Path, None], default=None, help="Output database directory")
    parser.add_argument("--start_from_scratch", default=0, action="store_true", help="Start from scratch (default: True)")
    parser.add_argument("--model_directory", type=Path, help="Model directory (optional)")
    parser.add_argument("--model_size", type=str, default="small", choices=["small", "base"], help="Model size (optional)")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size (default: 1)")

    args = parser.parse_args()
    
    if args.pdf_paths is None:
        raise ValueError("Please provide at least one PDF file path")
    
    pdf_paths = [Path(pdf_path) for pdf_path in args.pdf_paths] 

    output_dir = args.output_dir
    if output_dir is None:
        if len(pdf_paths)==1 and pdf_paths[0].is_dir():
            output_dir = pdf_paths[0]
        elif len(pdf_paths)>1:
            output_dirs = [pdf_path.parent for pdf_path in pdf_paths]
            if len(set(output_dirs)) == 1:
                output_dir = output_dirs[0]
        else:
            output_dir = Path.cwd()
    output_dir = Path(output_dir)
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)
            
    if len(pdf_paths) == 1:
        if pdf_paths[0].is_dir():
            pdf_paths = list(pdf_paths[0].glob("*.pdf"))
            
    start_from_scratch = args.start_from_scratch
    start_from_scratch = True if start_from_scratch == 1 else False

    run(
        pdf_paths=pdf_paths,
        output_db=output_dir,
        start_from_scratch=start_from_scratch,
        model_directory=args.model_directory,
        model_size=args.model_size,
        batch_size=args.batch_size
    )

if __name__ == "__main__":
    main()