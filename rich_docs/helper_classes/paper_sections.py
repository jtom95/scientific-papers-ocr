from typing import Optional, List, Tuple, Dict
from dataclasses import dataclass
from collections import namedtuple
import asyncio
import logging
import re
import json
from string import capwords

from ..references.base import Reference
from ..references.elaborator_class import ReferenceExtr


Position = namedtuple("Position", ["page", "pos"])


@dataclass
class Section:
    start_position: Optional[Position] = None
    end_position: Optional[Position] = None
    text: Optional[str] = None
    title: Optional[str] = None
    number: Optional[str] = None

    def to_string(self, include_title=False):
        txt = ""
        if include_title:
            txt += f"## {self.number} {capwords(self.title)}\n\n"
        txt += f"{self.text}\n\n"
        # final cleaning
        txt = re.sub(r"\n{3,}", "\n\n", txt).strip()
        return txt

    def to_dict(self):
        return {
            "start_position": self.start_position,
            "end_position": self.end_position,
            "text": self.text,
            "title": self.title,
            "number": self.number,
        }

    def to_json(self):
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: dict):
        if data is None: 
            return cls()
        return cls(
            start_position=data["start_position"],
            end_position=data["end_position"],
            text=data["text"],
            title=data["title"],
            number=data["number"],
        )

    @classmethod
    def from_json(cls, json_string: str):
        return cls.from_dict(json.loads(json_string))


@dataclass
class Abstract(Section):
    pass


@dataclass
class References(Section):
    references: Dict[int, Reference] = None

    def __post_init__(self):
        if self.references is None:
            raise ValueError("references must be provided")
        for key, value in self.references.items():
            if not isinstance(key, int):
                raise ValueError("keys of references must be integers")

            if isinstance(value, str):
                self.references[key] = Reference(raw=value)

            elif not isinstance(value, Reference):
                raise ValueError("values of references must be of type Reference")

    def get(self, index: int):
        return self.references[index]

    def num_refs(self):
        return len(self.references)

    def to_string(self, include_title=False):
        string = ""
        if include_title:
            string += f"## References\n\n"
        for ref_number, ref in self.references.items():
            string += f"* [{ref_number}] {ref.raw}\n"
        return string

    def elaborate_references_(self, minimum_complexity_level: int = 0):
        for i in self.references:
            if self.references[i].reference_complexity_level > minimum_complexity_level:
                continue
            ref_extractor = ReferenceExtr(self.references[i].raw)
            self.references[i] = ref_extractor.run()
            self.references[i].reference_complexity_level = 2

    async def elaborate_references(self, minimum_complexity_level: int = 0):
        tasks = []

        for i in self.references.keys():
            if self.references[i].reference_complexity_level > minimum_complexity_level:
                continue
            ref_extractor = ReferenceExtr(self.references[i].raw, ref_number=i)
            task = asyncio.create_task(ref_extractor.async_run())
            tasks.append(task)

        references: List[Reference] = await asyncio.gather(*tasks)

        self.references = {}
        for reference in references:
            if reference.ref_number is None:
                raise ValueError("Reference number cannot be None")
            if reference.ref_number in self.references.keys():
                raise ValueError(f"Reference number {reference.ref_number} already exists")
            self.references[reference.ref_number] = reference
            self.references[reference.ref_number].reference_complexity_level = 2

        logging.info("Done elaborating references.")

    def to_dict(self):
        return {
            "start_position": self.start_position,
            "end_position": self.end_position,
            "text": self.text,
            "title": self.title,
            "number": self.number,
            "references": {key: value.to_dict() for key, value in self.references.items()},
        }
    
    @classmethod
    def from_dict(cls, data: dict):
        if data is None: 
            return None
        return cls(
            start_position=data["start_position"],
            end_position=data["end_position"],
            text=data["text"],
            title=data["title"],
            number=data["number"],
            references={int(key): Reference.from_dict(value) for key, value in data["references"].items()},
        )
