from typing import List, Dict, Any, Optional
from pprint import pprint
import requests
from pathlib import Path
import logging
import re
from ratelimit import limits, sleep_and_retry

from ..notion_cli import NotionClient
from ..basics.constants import NotionAddresses, DatabasePropertyTypes, DictBlockTypes, DictBlock
from ..markdown_to_notion.markdown_to_notion_blocks import MarkdownToNotionPayload, _TextSplitter

from rich_docs.edocument_class import EDocument


class NotionPage:
    def __init__(
        self,
        page_id: str,
        notion: NotionClient,
    ):
        self.id = page_id
        self.notion: NotionClient = notion

        # self._get_page_info()

    @property
    def url_global_endpoint(self) -> str:
        return f"{NotionAddresses.base_address}/pages/{self.id}"

    @property
    def url_block_endpoint(self) -> str:
        return f"{NotionAddresses.base_address}/blocks/{self.id}/children"

    @property
    def header(self):
        return self.notion._generate_header()

    # @property
    # def title(self):
    #     return self.page.title

    # @property
    # def children(self):
    #     return self.page.children

    # def _get_page_info(self):
    #     self.page = self.notion.get_block(self.id)
    #     self.children_ids = [child.id for child in self.children]
    #     self.children_titles = [child.title for child in self.children]

    def create_subpage(self, title: str):
        """Creates a subpage under the current page."""
        url = f"{NotionAddresses.base_address}/pages"
        payload = {
            "parent": {"page_id": self.id},
            "properties": {"title": {"title": [{"text": {"content": title}}]}},
        }
        response = requests.post(url, json=payload, headers=self.header)
        if response.status_code == 200:
            logging.info(f"Successfully created subpage {title}")
            return response.json()
        logging.error(f"Failed to create subpage {title}")
        logging.error(response.status_code)
        logging.error(response.text)

    def text2blocks(self, text: str, max_blocks=100, max_len_block=1000):
        paragraph_block_generator = MarkdownToNotionPayload().split_mathpix_markdown_into_blocks(
            text, max_len_block=max_len_block
        )

        blocks = []
        for ii, block_dict in enumerate(paragraph_block_generator):
            logging.info(f"Adding block {ii} of type {block_dict['type']} to the notion page")
            if block_dict["type"] == DictBlockTypes.TEXT:
                paragraph_text = block_dict["content"]
                blocks += self._write_rich_text_blocks_list(paragraph_text)
            elif block_dict["type"] == DictBlockTypes.MATH:
                equation_text = block_dict["content"]
                blocks += self._write_math_block(equation_text)
            elif block_dict["type"] == DictBlockTypes.HEADER1:
                header_text = block_dict["content"]
                blocks += MarkdownToNotionPayload.text_to_header_payload(header_text, 1)["children"]
            elif block_dict["type"] == DictBlockTypes.HEADER2:
                header_text = block_dict["content"]
                blocks += MarkdownToNotionPayload.text_to_header_payload(header_text, 2)["children"]
            elif block_dict["type"] == DictBlockTypes.HEADER3:
                header_text = block_dict["content"]
                blocks += MarkdownToNotionPayload.text_to_header_payload(header_text, 3)["children"]
            else:
                raise ValueError("Invalid block type")

            # if len(blocks) > max_blocks:
            #     blocks_to_send = blocks[:max_blocks]
            #     blocks = blocks[max_blocks:]
            #     self._send_blocks(blocks_to_send)
            # elif len(blocks) == max_blocks:
            #     self._send_blocks(blocks)
            #     blocks = []
        return blocks

    def write_section_to_notion(self, text: str, max_blocks=100, max_len_block=1000):
        """Writes a section of text to the notion page."""
        blocks = self.text2blocks(text, max_blocks=max_blocks, max_len_block=max_len_block)
        self.send_blocks(blocks)
        
    def send_blocks(self, blocks: List[DictBlock], max_blocks=100):
        """Sends blocks to the notion page."""
        for ii in range(0, len(blocks), max_blocks):
            blocks_to_send = blocks[ii:ii + max_blocks]
            self._send_blocks(blocks_to_send)

    def _send_blocks(self, blocks):
        payload = {"children": blocks}
        response = self._rate_limited_request(
            url=self.url_block_endpoint, payload=payload, headers=self.header
        )
        if response.status_code == 200:
            logging.info(f"Successfully added {len(blocks)} blocks to the notion page")
            return response.json()
        logging.error("Failed to add blocks to the notion page")
        logging.error(response.status_code)
        logging.error(response.text)

    def _write_rich_text_blocks_list(self, text):
        return MarkdownToNotionPayload().text_to_rich_text_payload(text=text)["children"]

    def _write_math_block(self, text):
        return MarkdownToNotionPayload().math_text_to_math_block_payload(text=text)["children"]

    def write_math_block(self, text: str):
        url = self.url_block_endpoint
        headers = self.header
        payload = MarkdownToNotionPayload().math_text_to_math_block_payload(text=text)
        response = self._rate_limited_request(url=url, payload=payload, headers=headers)
        if response.status_code == 200:
            logging.info("Successfully added math block to the notion page")
            return response.json()
        logging.error("Failed to add math block to the notion page")
        logging.error(response.status_code)
        logging.error(response.text)

    def write_header(self, text, header_level=1):
        url = self.url_block_endpoint
        headers = self.header
        payload = MarkdownToNotionPayload().text_to_header_payload(
            text=text, header_level=header_level
        )
        response = self._rate_limited_request(url=url, payload=payload, headers=headers)
        if response.status_code == 200:
            logging.info("Successfully added header to the notion page")
            return response.json()
        logging.error("Failed to add header to the notion page")
        logging.error(response.status_code)
        logging.error(response.text)

    def write_rich_text(self, text):
        url = self.url_block_endpoint
        headers = self.header
        text_chunks = _TextSplitter().split_text(text)

        for ii, chunk in enumerate(text_chunks):
            payload = MarkdownToNotionPayload().text_to_rich_text_payload(text=chunk)
            response = self._rate_limited_request(url=url, payload=payload, headers=headers)

            if response.status_code != 200:
                logging.error("Failed to add paragraph to the notion page")
                logging.error(response.status_code)
                logging.error(response.text)
                break
            logging.info(
                "Successfully added chunk {}/{} to the notion page".format(ii + 1, len(text_chunks))
            )

    @sleep_and_retry
    @limits(calls=3, period=1)
    def _rate_limited_request(self, url, payload, headers):
        """NOTION API limit to 3 payloads a second: https://developers.notion.com/reference/request-limits"""
        return requests.patch(url, json=payload, headers=headers)

    @classmethod
    def from_id(
        cls,
        page_id: str,
        notion: NotionClient,
    ) -> "NotionPage":
        page = cls(page_id=page_id, notion=notion)
        return page
