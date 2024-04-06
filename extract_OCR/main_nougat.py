"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""
from typing import List, Optional, Tuple, Dict

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


logging.basicConfig(level=logging.INFO)

class NougatOCR:
    def __init__(self, local_settings: LocalSettings, configs: Configs):
        self.local_settings = local_settings
        self.configs = configs
        self.batch_size = self.configs.batch_size or self.get_default_batch_size()

    def assert_cuda_available(self):
        if not torch.cuda.is_available():
            logging.warning("CUDA is not available. Using CPU.")
            return False
        return True

    def get_default_batch_size(self):
        cuda_memory = torch.cuda.get_device_properties(0).total_memory
        return int(cuda_memory / 1024 / 1024 / 1000 * 0.4)

    def init_model(self) -> NougatModel:
        checkpoint_path = self.local_settings.checkpoint_directory
        model = NougatModel.from_pretrained(checkpoint_path)
        if not self.assert_cuda_available():
            model = move_to_device(model, bf16=not self.configs.full_precision, cuda=False)
        else:
            use_cuda = self.batch_size > 0
            model = move_to_device(model, bf16=not self.configs.full_precision, cuda=use_cuda)
        model.eval()  # set model to evaluation mode
        return model

    def run_on_paths(self, pdfs: Path | List[Path]) -> Dict[int, Tuple[Path, List[str]]]:
        model = self.init_model()
        if not isinstance(pdfs, (list, tuple)):
            pdfs = [pdfs]
        pdfs = self.string_to_path(pdfs)
        datasets = []
        for pdf in pdfs:
            dataset = self.preprocess_pdf(pdf, model)
            if dataset is not None:
                datasets.append(dataset)
        dataloader = self.generate_dataloader(datasets)

        self.predictions = []
        predictions_per_document = []
        file_index = 0
        page_num = 0

        for i, (sample, is_last_page) in enumerate(tqdm(dataloader)):
            model_output = model.inference(
                image_tensors=sample, early_stopping=self.configs.allow_skipping
            )  # check if model output is faulty
            for jj, output in enumerate(model_output["predictions"]):
                if page_num == 0:
                    logging.info(
                        "Processing file %s with %i pages"
                        % (datasets[file_index].name, datasets[file_index].size)
                    )
                page_num += 1
                if output.strip() == "[MISSING_PAGE_POST]":
                    # uncaught repetitions -- most likely empty page
                    predictions_per_document.append(f"\n\n[MISSING_PAGE_EMPTY:{page_num}]\n\n")
                elif self.configs.allow_skipping and model_output["repeats"][jj] is not None:
                    if model_output["repeats"][jj] > 0:
                        # If we end up here, it means the output is most likely not complete and was truncated.
                        logging.warning(f"Skipping page {page_num} due to repetitions.")
                        predictions_per_document.append(f"\n\n[MISSING_PAGE_FAIL:{page_num}]\n\n")
                    else:
                        # If we end up here, it means the document page is too different from the training domain.
                        # This can happen e.g. for cover pages.
                        predictions_per_document.append(
                            f"\n\n[MISSING_PAGE_EMPTY:{i*self.batch_size+jj+1}]\n\n"
                        )
                else:
                    if self.configs.markdown:
                        output = markdown_compatible(output)
                    predictions_per_document.append(output)
                if is_last_page[jj]:
                    self.predictions.append(predictions_per_document)
                    predictions_per_document = []
                    page_num = 0
                    file_index += 1
                    
        self.predictions = {
            ii: (pdf, pred)
            for ii, (pdf, pred) in enumerate(zip(pdfs, self.predictions))
        }
        return self.predictions
    
    
    def write_predictions(self, predictions: List[str], output_dir: Path, filename: Path | str):
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        
        for ii, page in enumerate(predictions):
            logging.info(f"Writing page {ii}")
            out_path = output_dir / Path(filename).with_suffix(f".page_{ii}.mmd").name
            out_path.write_text(page, encoding="utf-8")
    

    def write_predictions_in_single_file(self, predictions: List[str], output_dir: Path, filename: Path | str):
        out = "".join(predictions).strip()
        out = re.sub(r"\n{3,}", "\n\n", out).strip()
        if output_dir:
            out_path = output_dir / Path(filename).with_suffix(".mmd").name
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(out, encoding="utf-8")
        else:
            print(out, "\n\n")

    def generate_dataloader(self, datasets: List[LazyDataset]) -> DataLoader:
        if len(datasets) == 0:
            logging.warning("No PDFs found. Exiting.")
            sys.exit(0)
        dataloader = DataLoader(
            ConcatDataset(datasets),
            batch_size=self.batch_size,
            collate_fn=LazyDataset.ignore_none_collate,
        )
        return dataloader

    def preprocess_pdf(self, pdf: Path, model: NougatModel, pages: Optional[List[int]] = None):
        if not pdf.exists():
            logging.warning(f"PDF {pdf} does not exist. Skipping.")
            return
        try:
            return LazyDataset(
                pdf,
                partial(model.encoder.prepare_input, random_padding=False),
                pages,
            )
        except pypdf.errors.PdfStreamError:
            logging.info(f"Could not load file {str(pdf)}.")
            return

    @staticmethod
    def string_to_path(string: str) -> List[Path]:
        if isinstance(string, (list, tuple)):
            return [Path(s) for s in string]
        elif isinstance(string, str):
            return [Path(string)]
        else:
            raise ValueError(f"Expected string or list of strings, got {type(string)}")

    @staticmethod
    def extract_images(pdf_path: Path, save_directory: Path):
        print("extracting images ...")
        # iterate over pdf pages
        pdf_file = fitz.open(pdf_path)
        for page_index in range(len(pdf_file)):
            # get the page itself
            page = pdf_file[page_index]
            image_list = page.get_images(full=True)
            # printing number of images found in this page
            if image_list:
                print(f"[+] Found a total of {len(image_list)} images in page {page_index}")
            else:
                print("[!] No images found on page", page_index)
            for image_index, img in enumerate(image_list, start=1):
                # get the XREF of the image
                xref = img[0]
                # extract the image bytes
                base_image = pdf_file.extract_image(xref)
                image_bytes = base_image["image"]
                # get the image extension
                image_ext = base_image["ext"]
                # load it to PIL
                image = Image.open(io.BytesIO(image_bytes))
                # save it to local disk
                image.save(open(f"{save_directory}/{page_index}_{image_index}.{image_ext}", "wb"))


if __name__ == "__main__":

    paths = [Path("sample.pdf"), Path("sample2.pdf")]
    local_settings = LocalSettings()
    configs = Configs(batch_size=1, allow_skipping=False)
    generate_markdowns = NougatOCR(local_settings, configs)

    # preds = generate_markdowns.run_on_paths(paths)
    # # save the predictions with dill
    # with open("predictions.dill", "wb") as f:
    #     dill.dump(preds, f)

    # laod the predictions with dill
    with open("predictions.dill", "rb") as f:
        preds = dill.load(f)
    print("Loaded predictions")
    elab = ElaborateMarkdownPrediction(preds[0], paths[0])
    edoc = elab.generate_Edoc()

    print(f"{edoc.title} by {edoc.authors} ({edoc.metadata.creation_date})")
    print(edoc.abstract)
    # generate_markdowns.extract_images(Path("sample.pdf"), Path("images"))
    # main()
