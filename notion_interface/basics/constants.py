from typing import Dict
from enum import Enum
from typing import NamedTuple

class NotionAddresses:
    base_address = r"https://api.notion.com/v1"


class DatabasePropertyTypes:
    title = "title"
    rich_text = "rich_text"
    number = "number"
    select = "select"
    multi_select = "multi_select"
    date = "date"
    people = "people"
    files = "files"
    checkbox = "checkbox"
    url = "url"
    email = "email"
    phone_number = "phone_number"
    formula = "formula"
    relation = "relation"
    rollup = "rollup"
    created_time = "created_time"
    created_by = "created_by"
    last_edited_time = "last_edited_time"
    last_edited_by = "last_edited_by"
    page = "page"
    files = "files"
    number = "number"




class DictBlockTypes(Enum):
    TEXT = "text"
    MATH = "math"
    HEADER1 = "header1"
    HEADER2 = "header2"
    HEADER3 = "header3"

# define the DictBlock type that is a dictionary with keys "type" and "content"
class DictBlock(dict):
    def __init__(self, type: DictBlockTypes, content: str):
        super().__init__(type=type, content=content)
