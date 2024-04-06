from typing import Dict, List, Optional, Tuple, Union
import string
from pathlib import Path
import re
import json

from .helper_classes.basic import PublicationPaperMetadata
from .helper_classes.paper_sections import Section, References
from .references.base import Author, Reference


class EDocument:
    def __init__(
        self,
        sections: Dict[int, Section],
        metadata: PublicationPaperMetadata,
    ):
        self.sections = sections
        self.metadata = metadata

    @property
    def title(self) -> Optional[Section]:
        return self.metadata.title

    @property
    def authors(self) -> Optional[List[Author]]:
        return self.metadata.authors

    @property
    def abstract(self) -> Optional[Section]:
        return self.metadata.abstract

    @property
    def references(self) -> Optional[References]:
        return self.metadata.references

    @property
    def text(self) -> str:
        text = ""
        # add title
        if self.title:
            title = string.capwords(self.title)
            text += f"# {title}\n\n"
        if self.authors:
            author_list = [str(author) for author in self.authors]
            author_string = ", ".join(author_list)
            text += f"**{author_string}**\n\n"
        if self.abstract:
            text += f"#### Abstract\n\n{self.abstract.text}\n\n"
        for _, section in self.sections.items():
            # add section title
            text += section.to_string(include_title=True)
            text += "\n\n"
        if self.references:
            text += self.references.to_string(include_title=True)
        
        # final cleaning
        text = re.sub(r"\n{3,}", "\n\n", text).strip()
        return text
    
    def generate_transcription(self, include_references: bool = False) -> str:
        text = ""
        # add title
        if self.title:
            title = string.capwords(self.title) 
            text += f"# {title}\n\n"
        # include authors
        if self.authors:
            author_list = [str(author) for author in self.authors]
            author_string = ", ".join(author_list)
            text += f"**{author_string}**\n\n"
        # include abstract if available
        if self.abstract:
            text += f"#### Abstract\n\n{self.abstract.text}\n\n"
        # compile text from sections
        for _, section in self.sections.items():
            if isinstance(section, Section):
                text += f"## {string.capwords(section.title)}\n\n{section.text}\n\n"
        # references if flag set
        if include_references and self.references:
            text += self.references.to_string(include_title=True)
        # final cleaning
        text = re.sub(r"\n{3,}", "\n\n", text).strip()
        return text
    
    
    def write(self, directory: Path, include_references = True) -> None:
        """Write the document to a directory"""
        # if the parent directory does not exist, create it
        full_savepath: Path = directory / self.metadata.pdf_filename
        full_savepath.parent.mkdir(parents=True, exist_ok=True)
        full_savepath = full_savepath.with_suffix(".md")
        text = self.generate_transcription(include_references)
        full_savepath.write_text(text, encoding="utf-8")

    def write_at_path(self, filepath: Path, include_references=True) -> None:
        """Write the document to a file"""
        # if the parent directory does not exist, create it
        filepath.parent.mkdir(parents=True, exist_ok=True)
        text = self.generate_transcription(include_references)
        filepath.write_text(text, encoding="utf-8")
        
    def to_dict(self)-> dict:
        return {
            "sections": {k: v.to_dict() for k, v in self.sections.items()},
            "metadata": self.metadata.to_dict(),
        }
    
    def to_json_str(self) -> str:   
        return json.dumps(self.to_dict(), indent=4)
    
    def to_json(self, filepath: Optional[Path]=None) -> None:
        if filepath is None:
            filepath = Path(".")
        # check if the provided path is a directory
        if filepath.is_dir():
            filepath = filepath / self.metadata.pdf_filename
            filepath = filepath.with_suffix(".json")
        # if the parent directory does not exist, create it
        filepath.parent.mkdir(parents=True, exist_ok=True)
        filepath.write_text(self.to_json_str(), encoding="utf-8")
    
    
    @classmethod
    def from_dict(cls, data: dict):
        return cls(
            sections={k: Section.from_dict(v) for k, v in data["sections"].items()},
            metadata=PublicationPaperMetadata.from_dict(data["metadata"]),
        )
    
    @classmethod
    def from_json_str(cls, json_string: str):
        return cls.from_dict(json.loads(json_string))
    
    @classmethod
    def from_json(cls, filepath: Path):
        return cls.from_json_str(filepath.read_text(encoding="utf-8"))
    
    
