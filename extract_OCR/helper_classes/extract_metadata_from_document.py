import pypdf
from pathlib import Path
from datetime import datetime
import json
import re
from typing import List, Optional, Tuple

from rich_docs.helper_classes.basic import DocumentMetaData
from rich_docs.references.base import Author


class PdfMeta:
    def __init__(
        self,
        directory: str,
    ):
        # find all pdfs in directory
        self.directory = Path(directory)

    def run(self):
        self.pdf_filenames = self.get_pdfs()

        # create a dictionary such that the key is an integer number, and the values are the metadata plus filename
        self.pdf_metadata = {}
        for i, pdf_filename in enumerate(self.pdf_filenames):
            self.pdf_metadata[i] = self.get_pdf_metadata(pdf_filename)
            self.pdf_metadata[i]["filename"] = pdf_filename[:-4]
            self.pdf_metadata[i]["file_location"] = self.directory
            self.pdf_metadata[i]["extraction_date"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return self.pdf_metadata

    def save_metadata(self, path: Path):
        with open(path, "w") as f:
            json.dump(self.pdf_metadata, f)

    def get_pdf_metadata(self, path: Path):
        with open(path, "rb") as f:
            pdf = pypdf.PdfFileReader(f)
            info = pdf.getDocumentInfo()
        return info

    def get_pdfs(self):
        pdfs = []
        for file in self.directory.iterdir():
            if file.suffix == ".pdf":
                pdfs.append(file)
        return pdfs

    @staticmethod
    def capitalize_name(name):
        # Split the name by spaces and hyphens and capitalize each part
        parts = re.split(r"(\s|-)", name)
        capitalized_parts = [
            part.capitalize() if part not in [" ", "-"] else part for part in parts
        ]

        # Rejoin the parts
        return "".join(capitalized_parts)

    @staticmethod
    def extract_embedded_metadata(pdf: Path) -> DocumentMetaData:
        with open(pdf, "rb") as f:
            reader = pypdf.PdfReader(f)
            info = reader.metadata

        # check if title is present in metadata
        title: str = info.title
        try:
            creation_date: datetime = info.creation_date
        except:
            creation_date: datetime = datetime.now()

        #  transform the authors string into a list of strings
        authors: str = info.author
        if authors in ("", " "):
            authors = None
        if authors is not None:
            authors = re.split(r"[,;]", authors)
            authors = [PdfMeta.capitalize_name(auth.strip()) for auth in authors]
            authors = [Author.from_string(auth) for auth in authors]

        return DocumentMetaData(title=title, authors=authors, creation_date=creation_date)
