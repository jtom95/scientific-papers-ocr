from typing import Optional
from dataclasses import dataclass, fields
from pprint import pprint
import logging
from .notion_page_handler import NotionPage
from ..markdown_to_notion.markdown_to_notion_blocks import MarkdownToNotionPayload
from ..notion_cli import NotionClient
from ..notion_databases.reference_database import RefDatabase

from rich_docs.edocument_class import EDocument


class SciPaperPage(NotionPage):
    def __init__(self, edoc: EDocument, page_id: str, notion: NotionClient):
        super().__init__(page_id, notion)
        self.edoc = edoc

    def write_to_Notion(self):
        self.create_transcript()
        self.write_header("Notes: ", header_level=1)
        self.write_rich_text("\n\n" * 3)
        self.add_reference_database()
        
    
    
    def create_transcript(self, new_page: bool = True):
        if new_page:
            transcript_page = self.create_new_empty_transcript_page()
            if transcript_page is None:
                logging.error("Failed to create the transcript page")
                return None
        else:
            transcript_page = self

        self.write_title(page=transcript_page)
        self.write_abstract(page=transcript_page)
        self.write_sections(page=transcript_page)
        logging.info("Successfully generated the transcript")

    def create_new_empty_transcript_page(self, transcript_title="Paper Transcript") -> NotionPage:
        title = transcript_title
        resp = self.create_subpage(title)
        if resp is None:
            return None
        self.full_paper_page_id = resp.get("id", None)
        if self.full_paper_page_id is None:
            logging.error("Failed to retrieve the new page id")
            return None
        self.full_paper_page = NotionPage(self.full_paper_page_id, self.notion)
        return self.full_paper_page

    def write_title(self, page: Optional[NotionPage] = None):
        if page is None:
            page = self
        title = self.edoc.title
        if title is None:
            return
        page.write_section_to_notion("#" + title)

    def write_abstract(self, page: Optional[NotionPage] = None):
        abstract = self.edoc.abstract.text
        abstract_blocks = self.text2blocks("**Abstract:** " + abstract)
        annotated_blocks = []
        if isinstance(abstract_blocks, dict):
            abstract_blocks = [abstract_blocks]
        for abstr_block in abstract_blocks:
            annotated_blocks.append(
                MarkdownToNotionPayload.add_annotation_to_rich_text_block(
                    payload_block=abstr_block,
                    italic=True,
                )
            )

        if page is None:
            page = self
        page._send_blocks(annotated_blocks)

    def write_section_number(self, section_number: int, page: Optional[NotionPage] = None):
        section_number = str(section_number)
        section_title = self.edoc.sections[section_number].title
        section_text = self.edoc.sections[section_number].text

        title_blocks = self.text2blocks("##" + f"{section_number}. " + section_title)
        section_blocks = self.text2blocks(section_text)

        if page is None:
            page = self
        page._send_blocks(title_blocks + section_blocks)

    def write_sections(self, page: Optional[NotionPage] = None):
        for section_name in self.edoc.sections:
            self.write_section_number(section_name, page=page)

    def add_reference_database(
        self, page: Optional[NotionPage] = None, references_title: str = "References"
    ):
        if page is None:
            page = self

        ref_db = RefDatabase.create_database(
            notion=self.notion,
            name=references_title,
            parent_id=page.id,
        )

        # new references are added at the top, so let's reverse the order
        # self.edoc.references.references is a dict with ordered integers as keys
        # so we can just reverse the keys and get the references in the right order
        references = {}
        for key in reversed(self.edoc.references.references):
            references[key] = self.edoc.references.references[key]

        ref_db.add_references(references)
