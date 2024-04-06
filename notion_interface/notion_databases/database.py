from typing import Optional, List, Dict
import re
from pprint import pprint
import json
import requests
import logging
from dataclasses import dataclass
from enum import Enum

from ..notion_cli import NotionClient
from ..basics.constants import NotionAddresses, DatabasePropertyTypes


class ParentTypes(Enum):
    page = "page_id"
    database = "database_id"
    workspace = "workspace"


@dataclass
class DatabaseProperty:
    name: str
    type: DatabasePropertyTypes
    id: Optional[str] = None


@dataclass
class DatabasePage:
    properties: List[DatabaseProperty]
    id: Optional[str] = None
    created_time: Optional[str] = None
    last_edited_time: Optional[str] = None


class DatabaseEntryStatus(Enum):
    ToRead = "To Read"
    Reading = "Reading"
    Skimmed = "Skimmed"
    Read = "Read"
    Analyzed = "Analyzed"


class Database:
    def __init__(
        self,
        database_id: str,
        notion: NotionClient,
    ):
        self.id = database_id
        self.notion = notion

        self._get_database_info()
        results = self.get_pages()
        self.pages = self._generate_pages_from_results(results)

        #

    @property
    def database_url(self):
        return f"{NotionAddresses.base_address}/databases/{self.id}"

    @classmethod
    def from_id(
        cls,
        database_id: str,
        notion_api_key: Optional[str] = None,
        notion_version: Optional[str] = None,
    ):
        notion = NotionClient(
            notion_api_key=notion_api_key,
            notion_version=notion_version,
        )
        return cls(database_id, notion)

    def _generate_pages_from_results(self, results: dict):
        pages = []
        for result in results:
            properties = result["properties"]
            page = DatabasePage(
                properties=self._find_properties(properties),
                id=result["id"],
                created_time=result["created_time"],
                last_edited_time=result["last_edited_time"],
            )
            pages.append(page)
        return pages

    def _find_properties(self, properties: dict) -> List[DatabaseProperty]:
        props = []
        for k, value in properties.items():
            prop = DatabaseProperty(
                name=k,
                type=DatabasePropertyTypes.__dict__[value["type"]],
                id=value["id"],
            )
            props.append(prop)
        return properties

    def _get_database_info(self):
        header = self.notion._generate_header()
        self.response = requests.get(self.database_url, headers=header)
        if self.response.status_code == 200:
            logging.info("Successfully retrieved database info from Notion")
            data = self.response.json()
            if data["object"] != "database":
                logging.error("Retrieved object is not a database. It is a " + data["object"])
                return None
            self.url = data["url"]
            self.title = data["title"][0]["plain_text"]
            self.properties = data["properties"]
            self.parent = data["parent"]
            if self.parent["type"] == "workspace":
                self.workspace = self.parent["workspace"]
                self.parent_id = ParentTypes.workspace
            if self.parent["type"] == "page_id":
                self.workspace = None
                self.parent_id = ParentTypes.page
            if self.parent["type"] == "database_id":
                self.workspace = None
                self.parent_id = ParentTypes.database

        else:
            logging.error("Failed to retrieve database info from Notion")
            logging.error(self.response.status_code)
            logging.error(self.response.text)
            return None

    def get_pages(self, **kwargs):
        url = f"{self.database_url}/query"
        header = self.notion._generate_header()
        default_payload = {
            "page_size": 100,
        }
        payload = {**default_payload, **kwargs}
        self.response = requests.post(url, json=payload, headers=header)
        if self.response.status_code == 200:
            logging.info("Successfully retrieved pages from Notion")
            data = self.response.json()
            return data["results"]
        else:
            logging.error("Failed to retrieve pages from Notion")
            logging.error(self.response.status_code)
            logging.error(self.response.text)
            return None

    def create_page(
        self,
        all_db_properties: List[DatabaseProperty],
        papers_property_dict: Dict[str, str],
        extra_properties_payload: Optional[dict]=None,
    ):
        if extra_properties_payload is None:
            extra_properties_payload = {}
        properties_payload = extra_properties_payload.copy()
        for property in all_db_properties:
            if property.name in extra_properties_payload:
                continue

            prop_type = property.type
            key = property.name
            value = papers_property_dict[key]
            if prop_type == DatabasePropertyTypes.title:
                if value is None:
                    # get the value from the raw citation
                    matches = re.split(r"\.", papers_property_dict["citation"])
                    value = matches[0]
                properties_payload[key] = {"title": [{"text": {"content": value}}]}
            elif prop_type == DatabasePropertyTypes.multi_select:
                if value is None:
                    properties_payload[key] = {"multi_select": []}
                else:
                    properties_payload[key] = {"multi_select": [{"name": v} for v in value]}
            elif prop_type == DatabasePropertyTypes.date:
                properties_payload[key] = {"date": {"start": value}}
            elif prop_type == DatabasePropertyTypes.url:
                properties_payload[key] = {"url": value}
            elif prop_type == DatabasePropertyTypes.rich_text:
                properties_payload[key] = {"rich_text": [{"text": {"content": value}}]}
            elif prop_type == DatabasePropertyTypes.select:
                if value is None:
                    continue
                properties_payload[key] = {"select": {"name": value}}
            elif prop_type == DatabasePropertyTypes.number:
                properties_payload[key] = {"number": value}

        url = f"{NotionAddresses.base_address}/pages"
        header = self.notion._generate_header()

        payload = {"parent": {"database_id": self.id}, "properties": properties_payload}
        


        response = requests.post(url, json=payload, headers=header)
        if response.status_code == 200:
            logging.info("Successfully added new paper to the database")
            return response.json()  # or any other relevant information from the response
        else:
            logging.error("Failed to add new paper to the database")
            logging.error(response.status_code)
            logging.error(response.text)
            return None

    @classmethod
    def create_database(
        cls,
        notion: NotionClient,
        name: str,
        parent_id: str,
        properties: List[DatabaseProperty],
        parent_type: ParentTypes = ParentTypes.page,
        payload_kwargs: Optional[dict] = {},
    ):
        if isinstance(parent_type, str):
            parent_type = ParentTypes[parent_type]
        if parent_type == ParentTypes.workspace:
            parent_id = None

        url = f"{NotionAddresses.base_address}/databases"
        header = notion._generate_header()
        properties_payload = {}

        for prop in properties:
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
            elif prop.type == DatabasePropertyTypes.number:
                properties_payload[prop.name] = {"number": {}}
            else:
                logging.error(f"Property type {prop.type} is not supported")
                
        
        # make sure the properties_payload has the keys in the same order as the properties
        properties_payload = {prop.name: properties_payload.get(prop.name) for prop in properties}


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
        
        # update the payload with the payload_kwargs
        payload = {**payload, **payload_kwargs}
        
        
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
