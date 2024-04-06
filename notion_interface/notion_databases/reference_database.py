from dataclasses import fields
from pprint import pprint
from typing import List, Optional, Tuple, Dict
import datetime

from ..notion_cli import NotionClient

from .database import Database, DatabaseProperty, ParentTypes
from ..basics.configs import ReferenceProperties
from ..basics.constants import DatabasePropertyTypes
from ..pages.notion_page_handler import NotionPage

from rich_docs.references.base import Reference

default_properties = [
    DatabaseProperty(name=f.name, type=f.default) for f in fields(ReferenceProperties)
]

# # change the order of the default properties to have "title" -> "authors" -> "ref_number" -> "year" and the rest in any order

# default_properties = [
#     DatabaseProperty(name="title", type=DatabasePropertyTypes.title),
#     DatabaseProperty(name="authors", type=DatabasePropertyTypes.multi_select),
#     DatabaseProperty(name="ref_number", type=DatabasePropertyTypes.number),
#     DatabaseProperty(name="year", type=DatabasePropertyTypes.number),
# ]

# for f in fields(ReferenceProperties):
#     if f.name not in ["title", "authors", "ref_number", "year"]:
#         default_properties.append(DatabaseProperty(name=f.name, type=f.default))


class RefDatabase(Database):
    default_properties = default_properties
    def __init__(
        self,
        database_id: str,
        notion: NotionClient,
    ):
        super().__init__(database_id=database_id, notion=notion)
        self.references: List[NotionPage] = []
        
    def add_references(self, references: Dict[int, Reference]):
        """Adds a list of references to the database."""
        for reference in references.values():
            self.add_reference_entry(reference)

    def add_reference_entry(self, reference: Reference):
        """Adds a reference entry to the database."""
        ref_dict_original = reference.to_dict()

        ref_dict = {}

        ref_dict["title"] = ref_dict_original["title"]
        if reference.authors is None:
            reference.authors = []
        else:
            ref_dict["authors"] = [str(author) for author in reference.authors]
        creation_date = ref_dict_original.get("date", None)
        if creation_date == None:
            ref_dict["publication_date"] = ""
        else:
            # transform the timestamp to year and month
            ref_dict["publication_date"] = datetime.datetime.fromtimestamp(creation_date).strftime(
                "%Y-%m"
            )

        for f in fields(ReferenceProperties):
            field_name = f.name
            if field_name not in ref_dict:
                ref_dict[field_name] = ref_dict_original.get(field_name, "")

        # the citatio field is saved as "raw" in the reference
        ref_dict["citation"] = ref_dict_original.get("raw", "")

        # keywords are multiselect, so , . are not allowed
        if ref_dict.get("keywords", None) is not None:
            ref_dict["keywords"] = [k.replace(",", "").replace(".", "") for k in ref_dict["keywords"]]
        else:
            ref_dict["keywords"] = []
            
        
        return self.create_page(
            all_db_properties=self.default_properties,
            papers_property_dict=ref_dict,
        )

    @classmethod
    def create_database(
        cls,
        notion: NotionClient,
        name: str,
        parent_id: str,
        extra_properties: Optional[List[DatabaseProperty]] = [],
        parent_type: ParentTypes = ParentTypes.page,
    ) -> "RefDatabase":
        all_properties = cls.default_properties + extra_properties
        database = super().create_database(
            notion=notion,
            name=name,
            parent_id=parent_id,
            properties=all_properties,
            parent_type=parent_type,
            payload_kwargs={"is_inline": True, "icon": {"type": "emoji", "emoji": "📚"}},
        )
        return database
