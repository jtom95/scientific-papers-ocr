from typing import Optional, List
from dataclasses import dataclass, fields
from enum import Enum
from collections import namedtuple
from datetime import datetime
import json
from pathlib import Path

from .paper_sections import Section, Abstract, References
from ..references.base import Author


@dataclass
class DocumentMetaData:
    title: Optional[str] = None
    authors: Optional[List[Author]] = None
    creation_date: Optional[datetime] = None
    pdf_directory: Optional[str] = None
    pdf_filename: Optional[str] = None
    
    def to_dict_shallow(self):
        my_dict = {}
        for field in fields(self):
            my_dict[field.name] = getattr(self, field.name)
        return my_dict


@dataclass
class PublicationPaperMetadata(DocumentMetaData):
    doi: Optional[str] = None
    publisher: Optional[str] = None
    publication_type: Optional[str] = None
    url: Optional[str] = None
    keywords: Optional[List[str]] = None
    abstract: Optional[Abstract | str] = None
    references: Optional[References | List[str]] = None

    def to_dict(self):
        if isinstance(self.abstract, Section):
            abstract: Section = self.abstract
            abstract = abstract.to_dict()
        else:
            abstract = self.abstract
        if isinstance(self.references, References):
            references: References = self.references
            references = references.to_dict()
        else:
            references = self.references
        return {
            "title": self.title,
            "authors": [
                author.to_dict() if isinstance(author, Author) else author
                for author in self.authors
            ] if self.authors is not None else None,
            "creation_date": self.creation_date.timestamp()
            if isinstance(self.creation_date, datetime)
            else self.creation_date,
            "pdf_directory": str(self.pdf_directory),
            "pdf_filename": str(self.pdf_filename),
            "keywords": self.keywords,
            "abstract": abstract,
            "references": references,
            "doi": self.doi,
            "publisher": self.publisher,
            "publication_type": self.publication_type,
            "url": self.url,
            
        }

    def to_json(self):
        return json.dumps(self.to_dict(), indent=4)

    @classmethod
    def from_dict(cls, d: dict):
        if isinstance(d["creation_date"], float):
            try:
                d["creation_date"] = datetime.fromtimestamp(d["creation_date"])
            except:
                pass

        return cls(
            title=d["title"],
            authors=[
                Author(**author) if isinstance(author, dict) else author for author in d["authors"]
            ] if d["authors"] is not None else None,
            creation_date=d["creation_date"],
            pdf_directory=Path(d["pdf_directory"]),
            pdf_filename=d["pdf_filename"],
            keywords=d["keywords"],
            abstract=Abstract.from_dict(d["abstract"])
            if isinstance(d["abstract"], dict)
            else Abstract.from_dict(d["abstract"]),
            references=References.from_dict(d["references"])
            if isinstance(d["references"], dict)
            else References.from_dict(d["references"]),
            doi=d["doi"],
            publisher=d["publisher"],
            publication_type=d["publication_type"],
            url=d["url"],
        )

    @classmethod
    def from_json(cls, json_string: str):
        return cls.from_dict(json.loads(json_string))
