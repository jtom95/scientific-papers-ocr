from dataclasses import dataclass, fields
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any
import logging
import requests
import datetime
from enum import Enum
import re

from notion_interface.notion_cli import NotionClient

from rich_docs.edocument_class import EDocument

from .database import Database, DatabasePage, DatabaseProperty, ParentTypes, DatabaseEntryStatus
from ..basics.constants import NotionAddresses, DatabasePropertyTypes
from ..basics.configs import ScientificDatabaseProperties
from ..pages.scientific_page_handler import SciPaperPage


class PaperDatabase(Database):
    default_properties = [
        DatabaseProperty(name=f.name, type=f.default) for f in fields(ScientificDatabaseProperties)
    ]

    def __init__(
        self,
        database_id: str,
        notion: NotionClient,
    ):
        super().__init__(database_id=database_id, notion=notion)
        self.papers: List[SciPaperPage] = []

    def add_papers(self, papers: List[EDocument] | EDocument):
        if isinstance(papers, EDocument):
            papers = [papers]
        self.add_papers_entry(papers)
        for paper in self.papers:
            paper.create_transcript()
            paper.write_header("Notes: ", header_level=1)
            paper.write_rich_text("\n\n" * 3)
            paper.add_reference_database()

    def add_papers_entry(self, papers: List[EDocument]):
        for paper in papers:
            self.add_paper_entry(paper)

    def add_paper_entry(self, paper: EDocument):
        response = self.add_paper_db_entry(paper)
        if response is None:
            return None
        new_page_id = response["id"]
        new_paper = SciPaperPage(edoc=paper, page_id=new_page_id, notion=self.notion)
        # check if the page is already in the database by comparing the title
        # if it is, then update the page

        for saved_paper in self.papers:
            if saved_paper.edoc.title == new_paper.edoc.title:
                # substitute the saved paper with the new one
                saved_paper = new_paper
                logging.info(f"Updated paper {saved_paper.edoc.title} in the database")
                return saved_paper

        self.papers.append(new_paper)
        logging.info(f"Added paper {new_paper.edoc.title} to the database")
        return new_paper
    
    def edoc_to_SciPaperPage(self, edoc: EDocument, page_id: str) -> SciPaperPage:
        return SciPaperPage(edoc=edoc, page_id=page_id, notion=self.notion)

    def add_paper_db_entry(self, paper: EDocument):
        paper_dict_original = paper.to_dict()

        paper_dict = {}

        paper_dict["authors"] = [str(author) for author in paper.authors]
        paper_dict["title"] = paper.title

        creation_date = paper_dict_original["metadata"].get("creation_date", 0)
        if creation_date == 0:
            paper_dict["publication_date"] = ""
        else:
            # transform the timestamp to year and month
            paper_dict["publication_date"] = datetime.datetime.fromtimestamp(
                creation_date
            ).strftime("%Y-%m")

        paper_dict["publisher"] = paper_dict_original["metadata"].get("publisher", [])
        paper_dict["publication_type"] = paper_dict_original["metadata"].get("publication_type", [])
        paper_dict["doi"] = paper_dict_original["metadata"].get("doi", None)
        paper_dict["url"] = paper_dict_original["metadata"].get("url", None)
        paper_dict["keywords"] = paper_dict_original["metadata"].get("keywords", [])
        paper_dict["file_directory"] = paper_dict_original["metadata"].get("pdf_directory", "")
        paper_dict["filename"] = paper_dict_original["metadata"].get("pdf_filename", "")

        # use the current date in the ISO format
        ADDED = datetime.datetime.now().isoformat()
        STATUS = DatabaseEntryStatus.ToRead.value
        HOT = False

        properties_payload = {}
        properties_payload["ADDED"] = {"date": {"start": ADDED}}
        properties_payload["STATUS"] = {"select": {"name": STATUS}}
        properties_payload["HOT"] = {"checkbox": HOT}

        return self.create_page(
            all_db_properties=self.default_properties,
            papers_property_dict=paper_dict,
            extra_properties_payload=properties_payload,
        )

    @classmethod
    def create_database(
        cls,
        notion: NotionClient,
        name: str,
        parent_id: str,
        edocs: Optional[Any] = None,
        extra_properties: Optional[List[DatabaseProperty]] = None,
        parent_type: ParentTypes = ParentTypes.page,
    )-> "PaperDatabase":
        if edocs is None:
            return cls.create_empty_database(
                notion=notion,
                name=name,
                parent_id=parent_id,
                extra_properties=extra_properties,
                parent_type=parent_type,
            )

        elif isinstance(edocs, EDocument):
            edocs = [edocs]
        elif isinstance(edocs, (str, Path)):
            edocs: Path = Path(edocs)
            # check if the path is a directory or a file
            if edocs.is_dir():
                edocs = [EDocument.from_json(f) for f in edocs.glob("*.json")]
            elif edocs.is_file():
                edocs = [EDocument.from_json(edocs)]
        elif isinstance(edocs, (list, tuple)):
            for ii, edoc in enumerate(edocs):
                if isinstance(edoc, (str, Path)):
                    edocs[ii] = EDocument.from_json(edoc)
                elif isinstance(edoc, dict):
                    edocs[ii] = EDocument.from_dict(edoc)
                elif isinstance(edoc, EDocument):
                    pass
                else:
                    raise TypeError(
                        f"The type of the {ii}th element of the edocs argument is not valid"
                    )
        else:
            raise TypeError("The type of the edocs argument is not valid")

        db = cls.create_empty_database(
            notion=notion,
            name=name,
            parent_id=parent_id,
            extra_properties=extra_properties,
            parent_type=parent_type,
        )
        
        db.add_papers(edocs)
        return db
        
        
    @classmethod
    def create_empty_database(
        cls,
        notion: NotionClient,
        name: str,
        parent_id: str,
        extra_properties: Optional[List[DatabaseProperty]] = None,
        parent_type: ParentTypes = ParentTypes.page,
    ):
        if extra_properties is None:
            extra_properties = []

        all_properties: List[DatabaseProperty] = cls.default_properties + extra_properties

        if isinstance(parent_type, str):
            parent_type = ParentTypes[parent_type]
        if parent_type == ParentTypes.workspace:
            parent_id = None

        url = f"{NotionAddresses.base_address}/databases"
        header = notion._generate_header()
        properties_payload = {}

        # boolean property
        properties_payload["ADDED"] = {"date": {}}
        # select property
        properties_payload["STATUS"] = {
            "select": {"options": [{"name": status.value} for status in DatabaseEntryStatus]}
        }
        # checkbox property
        properties_payload["HOT"] = {"checkbox": {}}

        for prop in all_properties:
            if prop.type == DatabasePropertyTypes.title:
                properties_payload[prop.name] = {"title": {}}
            elif prop.type == DatabasePropertyTypes.multi_select:
                properties_payload[prop.name] = {
                    "multi_select": {"options": []}
                }  # Assuming no options are predefined
            elif prop.type == DatabasePropertyTypes.date:
                properties_payload[prop.name] = {"date": {}}
            elif prop.type == DatabasePropertyTypes.url:
                properties_payload[prop.name] = {"url": {}}
            elif prop.type == DatabasePropertyTypes.rich_text:
                properties_payload[prop.name] = {"rich_text": {}}
            elif prop.type == DatabasePropertyTypes.select:
                properties_payload[prop.name] = {"select": {"options": []}}

        payload = {
            "parent": {
                "type": parent_type.value,
                parent_type.value: parent_id,
            },
            "title": [
                {
                    "type": "text",
                    "text": {
                        "content": name,
                    },
                },
            ],
            "properties": properties_payload,
        }
        response = requests.post(url, json=payload, headers=header)
        if response.status_code == 200:
            logging.info("Successfully created database in Notion")
            data = response.json()
            database_id = data["id"]
            logging.info(f"Generating Class for Database ID: {database_id}")
            return cls(database_id, notion)
        else:
            logging.error("Failed to create database in Notion")
            logging.error(response.status_code)
            logging.error(response.text)
            return None
