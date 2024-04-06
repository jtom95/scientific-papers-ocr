from typing import List, Optional, Tuple
from pathlib import Path
from tqdm import tqdm
import logging

from .elaborate_single_mmd_scientific import ElaborateMarkdownPrediction
from rich_docs.edocument_class import EDocument



class ElaborateMarkdownBatch:
    @staticmethod
    def run_on_paths(pdfs: List[Path], predictions: List[List[str]]) -> List[EDocument]:
        if not isinstance(pdfs, (list, tuple)):
            pdfs = [pdfs]
        pdfs = ElaborateMarkdownBatch.string_to_path(pdfs)
        edocs = []
        for ii, (pdf, prediction) in enumerate(zip(pdfs, predictions)):
            logging.info(f"{ii}) Elaborating {pdf.name}")
            mmd_elab = ElaborateMarkdownPrediction(prediction, pdf)
            edoc = mmd_elab.generate_Edoc()
            edocs.append(edoc)
        return edocs

    @staticmethod
    def write(edocs: List[EDocument], output_dir: Path) -> None:
        for edoc in edocs:
            edoc.write(output_dir)
            
    @staticmethod
    def string_to_path(paths: List[str]) -> List[Path]:
        return [Path(path) for path in paths]
