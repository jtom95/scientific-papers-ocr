"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""
from typing import List, Optional, Tuple, Dict, Union, Optional, Generator

import sys
import io
from pathlib import Path
import logging
import re
import argparse
import re
import os
import numpy as np
from functools import partial
import torch
from torch.utils.data import ConcatDataset, Dataset, DataLoader
from tqdm import tqdm
from nougat import NougatModel
from nougat.utils.dataset import LazyDataset
from nougat.utils.device import move_to_device, default_batch_size
from nougat.utils.checkpoint import get_checkpoint
from nougat.postprocessing import markdown_compatible
import pypdf

from PIL import Image
import fitz
from datetime import datetime
import dill

from .helper_classes.nougat_model_configs import LocalSettings, Configs
from .helper_classes.extract_metadata_from_document import PdfMeta

from rich_docs.helper_classes.basic import DocumentMetaData, PublicationPaperMetadata
from .elaborate_mmd_to_edoc.elaborate_single_mmd_scientific import ElaborateMarkdownPrediction
from .elaborate_mmd_to_edoc.elaborate_batch_mmd_scientific import ElaborateMarkdownBatch
from .extraction_elaboration.pdf_page_layout import PageLayoutAnalyzer
from .extraction_elaboration.check_single_page import CheckPageMissingParts
from .main_nougat import NougatOCR
from .extraction_elaboration.check_transcription import HandleIncompletePdfExtraction


from rich_docs.edocument_class import EDocument

logging.basicConfig(level=logging.INFO)


class ExtractOCR:
    def __init__(
        self,
        nougat: NougatOCR,
        pdfs: Union[Path, List[Path]],
        output_dir: Path | str,
    ):
        self.nougat = nougat
        if isinstance(pdfs, (Path, str)):
            self.pdfs = [Path(pdfs)]
        else:
            self.pdfs = [Path(p) for p in pdfs]
        self.output_dir = Path(output_dir)

        # initialize the incomplete pages dict
        self.incomplete_predictions: Dict[int, HandleIncompletePdfExtraction] = {}

    @property
    def pdf_names(self):
        return [p.name.replace(".pdf", "") for p in self.pdfs]

    # @staticmethod
    # def structured_markdown_to_edoc(
    #     structured_markdown: str,
    #     pdf_path: Path,
    # ) -> EDocument:
    #     """Generate an EDocument from a structured markdown"""
    #     elab = ElaborateMarkdownPrediction(structured_markdown, pdf_path)
    #     edoc = elab.generate_Edoc()
    #     return edoc

    @staticmethod
    def generate_Edoc_db_from_markdown_db(
        markdown_db: Path,
        pdf_paths: List[Path],
        edoc_dir: Path,
    ) -> Dict[int, EDocument]:
        """Generate an EDocument from a structured markdown"""
        # if the pdf paths is a generator convert it to a list
        if isinstance(pdf_paths, Generator):
            pdf_paths = list(pdf_paths)
        pdf_prediction_dirs = [p for p in markdown_db.glob("*") if p.is_dir()]
        # associate the pdf paths with the prediction directories
        path_and_pred_dict: Dict[int, Tuple[Path, Path]] = {}
        for ii, pdf_path in enumerate(pdf_paths):
            pdf_name = pdf_path.name.replace(".pdf", "")
            prediction_dir = [p for p in pdf_prediction_dirs if p.name == pdf_name][0]
            path_and_pred_dict[ii] = (pdf_path, prediction_dir)

        # check that all the pdfs have a prediction directory
        if not len(path_and_pred_dict) == len(pdf_paths):
            raise ValueError("Not all pdfs have a prediction directory")

        # write the edocs
        logging.info("Make the edoc directory")
        edoc_dir.mkdir(exist_ok=True, parents=True)

        # generate the edocs
        edoc_dict: Dict[int, EDocument] = {}
        for ii, (pdf_path, prediction_dir) in enumerate(path_and_pred_dict.values()):
            logging.info(f"Generating EDoc for {pdf_path}")
            # flatten the directory
            doc_pred_dict = HandleIncompletePdfExtraction.get_mmd_paths_from_structured_db(
                prediction_dir
            )
            predictions = []
            # get the prediction files
            for page in doc_pred_dict.keys():
                pred = doc_pred_dict[page]
                if isinstance(pred, Path):
                    predictions.append(pred.read_text())
                elif isinstance(pred, List):
                    predictions.append(" ".join([p.read_text() for p in pred]))
                else:
                    logging.error(f"Unknown type {type(pred)}")
            # generate the edoc
            edoc = ElaborateMarkdownPrediction(predictions, pdf_path).generate_Edoc()
            logging.info(f"Generated EDoc for {pdf_path}")
            logging.info(f"writing EDoc to {edoc_dir}")
            edoc.to_json(filepath=edoc_dir)
            edoc_dict[ii] = edoc
        # # write the edocs
        # logging.info(f"Writing {len(edoc_dict)} edocs to {edoc_dir}")
        # edoc_dir.mkdir(exist_ok=True, parents=True)
        # for edoc in edoc_dict.values():
        #     edoc.to_json(filepath=edoc_dir)

    @staticmethod
    def flatten_db(output_dir: Path, eliminate_page_parts: bool = False):
        """Flatten the database of predictions"""
        # get all the pdf directories
        pdf_dirs = [p for p in output_dir.glob("*") if p.is_dir()]
        for pdf_dir in pdf_dirs:
            logging.info(f"Flattening {pdf_dir}")
            HandleIncompletePdfExtraction.flatten_all_to_page_files(
                pdf_dir, eliminate_page_parts=eliminate_page_parts
            )

    def generate_structured_markdown_database(self, start_from_scratch: bool = False):
        if start_from_scratch:
            logging.info("Running OCR")
            self.run_nougat()
            logging.info("Writing predictions")
            self.write_predictions()

        logging.info("Checking predictions")
        self.check_predictions()
        logging.info("Running OCR on incomplete pages")
        self.run_nougat_on_incomplete()

    def run_nougat(self):
        self.prediction_dict = self.nougat.run_on_paths(self.pdfs)
        return self

    def write_predictions(self):
        if not len(self.prediction_dict) == len(self.pdfs):
            logging.error("Not all pdfs have a prediction: len of dict {} len of pdfs {}".format(
                len(self.prediction_dict), len(self.pdfs)
            ))
        for k, (path, pred) in self.prediction_dict.items():
            pdf_name = path.name.replace(".pdf", "")
            # check the pdf_name is among the pdf_names
            if not pdf_name in self.pdf_names:
                logging.error(f"Pdf name {pdf_name} not found in pdf names")
            output_dir_pdf = self.output_dir / pdf_name
            output_dir_pdf.mkdir(exist_ok=True, parents=True)
            for ii, page in enumerate(pred):
                page_path = output_dir_pdf / f"page_{ii}.mmd"
                with open(page_path, "w") as f:
                    f.write(page)
        # for name in self.pdf_names:
        #     output_dir_pdf = self.output_dir / name
        #     output_dir_pdf.mkdir(exist_ok=True, parents=True)
        #     pred = self.prediction_dict[name]
        #     for ii, page in enumerate(pred):
        #         page_path = output_dir_pdf / f"page_{ii}.mmd"
        #         with open(page_path, "w") as f:
        #             f.write(page)
        return self

    def check_predictions(self):
        self.incomplete_predictions = {}
        for name in self.pdf_names:
            output_dir_pdf = self.output_dir / name
            # check if the directory exists
            pdf_path_index = self.pdf_names.index(name)
            handler = HandleIncompletePdfExtraction(
                pdf_path=self.pdfs[pdf_path_index],
                pages_directory=output_dir_pdf,
            )
            handler.run()
            self.incomplete_predictions[pdf_path_index] = handler

    def run_nougat_on_incomplete(self):
        for handler in self.incomplete_predictions.values():
            logging.info(f"Running on {handler.pdf_path}")
            handler.run_nougat_on_new_files(self.nougat)


if __name__ == "__main__":
    paths = [Path("sample.pdf"), Path("sample2.pdf")]
    local_settings = LocalSettings()
    configs = Configs(batch_size=1, allow_skipping=False)
    # generate_markdowns = NougatOCR(local_settings, configs)

    # # preds = generate_markdowns.run_on_paths(paths)
    # # # save the predictions with dill
    # # with open("predictions.dill", "wb") as f:
    # #     dill.dump(preds, f)

    # # laod the predictions with dill
    # with open("predictions.dill", "rb") as f:
    #     preds = dill.load(f)
    # print("Loaded predictions")
    # elab = ElaborateMarkdownPrediction(preds[0], paths[0])
    # edoc = elab.generate_Edoc()

    # print(f"{edoc.title} by {edoc.authors} ({edoc.metadata.creation_date})")
    # print(edoc.abstract)
    # generate_markdowns.extract_images(Path("sample.pdf"), Path("images"))
    # main()
