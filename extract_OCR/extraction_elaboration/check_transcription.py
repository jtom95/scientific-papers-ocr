from typing import Optional, Dict, Union, List, Tuple

from pathlib import Path
import logging
import shutil

from extract_OCR.extraction_elaboration.check_single_page import CheckPageMissingParts
from extract_OCR.extraction_elaboration.pdf_page_layout import PageLayoutAnalyzer
from extract_OCR.main_nougat import NougatOCR


class HandleIncompletePdfExtraction:
    def __init__(
        self,
        pdf_path: Path,
        pages_directory: Path,
    ):
        self.pdf_path = Path(pdf_path)
        self.pages_directory = Path(pages_directory)
        self.incomplete_pages = {}
        self.incomplete_pages_full = {}
        
    @staticmethod
    def get_mmd_paths_from_structured_db(
        structured_db: Path,
    ) -> Dict[int, Union[Path, List[Path]]]:
        """Get the paths from a structured database"""
        folders = [p for p in structured_db.glob("*") if p.is_dir()]
        page_mmds = [p for p in structured_db.glob("*.mmd")]

        page_paths = {}

        for f in folders:
            # the name is "page_{page_number}"
            page_number = int(f.name.replace("page_", ""))
            # the folder contains mmd files with the name "part_{part_number}"
            mmds = [p for p in f.glob("*.mmd")]
            mmds.sort(key=lambda x: int(x.stem.replace("part_", "")))
            page_paths[page_number] = mmds

        for mmd in page_mmds:
            page_number = int(mmd.stem.replace("page_", ""))
            if page_number in page_paths:
                continue
            page_paths[page_number] = mmd
        
        # sort the page paths
        page_paths = {k: v for k, v in sorted(page_paths.items(), key=lambda item: item[0])}
        return page_paths

    @staticmethod
    def flatten_all_to_page_files(
        predictions_directory: Path,
        eliminate_page_parts: bool = False,
    ) -> str:
        """Generate a single markdown file from a directory of predictions"""
        # all contents of the directory should have the name "page_*".
        # They can either be folders or markdown

        page_strings = {}

        # first get all folders
        folders = [p for p in predictions_directory.glob("*") if p.is_dir()]

        for folder in folders:
            logging.info(f"Found Multiple Parts for page {folder.name} in {predictions_directory}")
            page_number = int(folder.name.replace("page_", ""))
            # the folder should contain mmd files with the name "part_*"
            mmds = [p for p in folder.glob("*.mmd")]
            mmds.sort(key=lambda x: int(x.stem.replace("part_", "")))
            page_strings[page_number] = " ".join([mmd.read_text() for mmd in mmds])
        
        # substitute the mmd files with the extracted text from the folders with the same name
        for num, page in page_strings.items():
            filepath = predictions_directory / f"page_{num}.mmd"
            filepath.write_text(page)
        
        if eliminate_page_parts:
            logging.info(f"Eliminating page parts for {predictions_directory}")
            # eliminate all the folders, note that the folders contain files 
            for folder in folders:
                shutil.rmtree(folder)
                

    @staticmethod
    def generate_single_markdown_from_directory(
        predictions_directory: Path,
    ) -> str:
        """Generate a single markdown file from a directory of predictions"""
        # all contents of the directory should have the name "page_*".
        # They can either be folders or markdown

        page_strings = {}

        # first get all folders
        folders = [p for p in predictions_directory.glob("*") if p.is_dir()]

        for folder in folders:
            logging.info(f"Found Multiple Parts for page {folder.name} in {predictions_directory}")
            page_number = int(folder.name.replace("page_", ""))
            # the folder should contain mmd files with the name "part_*"
            mmds = [p for p in folder.glob("*.mmd")]
            mmds.sort(key=lambda x: int(x.stem.replace("part_", "")))
            page_strings[page_number] = " ".join([mmd.read_text() for mmd in mmds])
        # now get all mmds
        mmds = [p for p in predictions_directory.glob("*.mmd")]
        # add then to the page_strings. If the page number already exists (because the page was split)
        # then skip it
        for mmd in mmds:
            page_number = int(mmd.stem.replace("page_", ""))
            if page_number in page_strings:
                continue
            page_strings[page_number] = mmd.read_text()

        # now sort the page strings
        page_strings = {k: v for k, v in sorted(page_strings.items(), key=lambda item: item[0])}
        combined_string = " ".join(page_strings.values())
        return combined_string

    def run(self, nougat: Optional[NougatOCR] = None):
        """Chain of functions to run to extract the OCR from a pdf"""
        self.check_pages()
        self.generate_new_files_for_incomplete_pages()
        if nougat:
            self.run_nougat_on_new_files(nougat)
        return self

    def check_pages(self):
        for pagefile in self.pages_directory.glob("*.mmd"):
            page_number = int(pagefile.stem.replace("page_", ""))
            page = pagefile.read_text()
            checker = CheckPageMissingParts(
                pdf=self.pdf_path,
                page_number=page_number,
                transcription=page,
            ).run()
            if checker.incomplete:
                self.incomplete_pages[page_number] = checker
        return self

    def generate_new_files_for_incomplete_pages(self):
        if self.incomplete_pages == {}:
            logging.info("No incomplete pages found")
            return self

        for page_number, checker in self.incomplete_pages.items():
            loa = PageLayoutAnalyzer(checker)
            column_patches = loa.get_column_patches()
            if column_patches is None:
                logging.info(f"No column patches found for page {page_number}")
                continue

            new_im_left, new_im_right = loa.generate_new_pages(
                patch_block_A=column_patches["left"], patch_block_B=column_patches["right"]
            )
            # transform the image to a pdf with two pages
            name = f"page_{page_number}"
            path = self.pages_directory / name / ("split" + ".pdf")
            path.parent.mkdir(exist_ok=True, parents=True)
            loa.images_to_pdf([new_im_left, new_im_right], path=path)

            # save the paths
            self.incomplete_pages_full[page_number] = {
                "path": path,
                "page_number": page_number,
                "checker": checker,
            }
        return self

    def run_nougat_on_new_files(self, nougat: NougatOCR):
        if self.incomplete_pages_full == {}:
            logging.info(f"No incomplete pages found for pdf {self.pdf_path.name}")
            return self
        # run nougat
        logging.info(f"Running nougat on incomplete pages for pdf {self.pdf_path.name}")
        new_preds = nougat.run_on_paths([p["path"] for p in self.incomplete_pages_full.values()])
        
        # write the predictions
        logging.info(f"Writing predictions for pdf {self.pdf_path.name}")
        for ii, (path, pred)  in new_preds.items():
            output_dir_pdf: Path = path.parent
            output_dir_pdf.mkdir(exist_ok=True, parents=True)
            for ii, page in enumerate(pred):
                page_path = output_dir_pdf / f"part_{ii}.mmd"
                with open(page_path, "w") as f:
                    f.write(page)
