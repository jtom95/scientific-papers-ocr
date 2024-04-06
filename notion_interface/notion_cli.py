import os
from typing import Optional
import requests
import getpass
import logging

NOTION_VERSION = "2022-06-28"

class NotionClient:
    def __init__(
        self,
        notion_api_key: Optional[str]=None,
        notion_version: Optional[str]=None,
    ):
        if notion_api_key is None:
            notion_api_key = os.environ.get("NOTION_API_KEY")
        if notion_api_key is None:
            notion_api_key = getpass.getpass("Enter Notion API Key: ")
        if notion_version is None:
            notion_version = NOTION_VERSION

        self.notion_api_key = notion_api_key
        self.notion_version = notion_version

    def _generate_header(self):
        header = {
            "Authorization": "Bearer " + self.notion_api_key,
            "Content-Type": "application/json",
            "Notion-Version": self.notion_version,
        }
        return header
