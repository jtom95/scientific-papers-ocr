from typing import List, Optional
from dataclasses import dataclass
from enum import Enum
import logging
from datetime import datetime
import json


class SearchURLs(Enum):
    scholar = "https://scholar.google.com/scholar?q={}"
    doi = "https://doi.org/{}"


@dataclass
class Author:
    surname: str
    given_name: str

    def __post_init__(self):
        if self.surname is None:
            self.surname = ""
        if self.given_name is None:
            self.given_name = ""
        self.surname = self.surname.lower()
        self.given_name = self.given_name.lower()

    def __str__(self):
        return f"{self.given_name.capitalize()} {self.surname.capitalize()}"

    def __repr__(self):
        return f"{self.given_name.capitalize()[0]}. {self.surname.capitalize()}"

    def to_dict(self):
        return {
            "surname": self.surname,
            "given_name": self.given_name,
        }

    def to_json(self):
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: dict):
        return cls(
            surname=data["surname"],
            given_name=data["given_name"],
        )

    @classmethod
    def from_json(cls, json_string: str):
        return cls.from_dict(json.loads(json_string))
    
    @classmethod
    def from_string(cls, string: str):
        if " " in string:
            parts = string.split(" ")
            name = parts[0]
            surname = parts[-1]
        elif "," in string:
            parts = string.split(",")
            name = parts[0]
            surname = parts[-1]
        else:
            surname = string
            name = ""
        return cls(surname=surname, given_name=name)


class ReferenceComplexityLevel(Enum):
    low = 0
    medium = 1
    high = 2


class Reference:
    def __init__(
        self,
        title: Optional[str] = None,
        authors: Optional[List[Author]] = None,
        publisher: Optional[str] = None,
        publication_type: Optional[str] = None,
        year: Optional[int] = None,
        date: Optional[datetime] = None,
        doi: Optional[str] = None,
        url: Optional[str] = None,
        keywords: Optional[List[str]] = None,
        raw: Optional[str] = None,
        reference_complexity_level: Optional[int] = ReferenceComplexityLevel.low.value,
        ref_number: Optional[int] = None,
    ):
        self.title = title
        self.authors = self.setup_authors(authors)
        self.publisher = publisher
        self.year = year
        self.raw = raw
        self.publication_type = publication_type
        self.date = date
        self.doi = doi
        self.url = url
        self.keywords = keywords
        self.reference_complexity_level = reference_complexity_level
        self.ref_number = ref_number

    def setup_authors(self, authors: List[Author]) -> List[Author]:
        if authors is None:
            return None
        parsed_authors = []
        for author in authors:
            if isinstance(author, str):
                if " " in author:
                    parts = author.split(" ")
                    name = parts[0]
                    surname = parts[-1]
                elif "," in author:
                    parts = author.split(",")
                    name = parts[0]
                    surname = parts[-1]
                else:
                    surname = author
                    name = ""
                parsed_authors.append(Author(surname=surname, given_name=name))
            elif isinstance(author, Author):
                parsed_authors.append(author)
            else:
                logging.warning(f"Could not parse author {author}")
        return parsed_authors

    def __str__(self):
        return self.raw

    def __repr__(self):
        return f"{self.title} ({self.year})- {self.authors[0], self.authors[-1]}"

    def to_dict(self):
        if self.authors is not None:
            authors = [
                author.to_dict() if isinstance(author, Author) else author
                for author in self.authors
            ]
        else:
            authors = None
        return {
            "title": self.title,
            "authors": authors,
            "publisher": self.publisher,
            "publication_type": self.publication_type,
            "year": self.year,
            "date": self.date.timestamp() if isinstance(self.date, datetime) else self.date,
            "doi": self.doi,
            "url": self.url,
            "keywords": self.keywords,
            "raw": self.raw,
            "reference_complexity_level": self.reference_complexity_level,
            "ref_number": self.ref_number,
        }

    def to_json(self):
        return json.dumps(self.to_dict(), indent=4)

    @classmethod
    def from_dict(cls, data: dict):
        if isinstance(data["date"], float):
            data["date"] = datetime.fromtimestamp(data["date"])
            
        # parse authors            
        if isinstance(data["authors"], list):
            authors = [Author.from_dict(author) for author in data["authors"]]
        elif isinstance(data["authors"], (str, dict)):
            raise ValueError(f"Authors should be a list of dicts, not a dict or a string: {data['authors']}")
        elif data["authors"] is None:
            authors = None
        else:
            raise ValueError("Authors in a strange format: {}".format(data["authors"]))
        
        return cls(
            title=data["title"],
            authors=authors,
            publisher=data["publisher"],
            publication_type=data["publication_type"],
            year=data["year"],
            date=data["date"],
            doi=data["doi"],
            url=data["url"],
            keywords=data["keywords"],
            raw=data["raw"],
            reference_complexity_level=data["reference_complexity_level"],
            ref_number=data["ref_number"],
        )
