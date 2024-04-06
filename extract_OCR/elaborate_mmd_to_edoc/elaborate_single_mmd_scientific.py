from typing import List, Optional, Tuple, Dict

from pathlib import Path
import re
import pypdf
import roman
import asyncio
import logging
from datetime import datetime

from ..helper_classes.extract_metadata_from_document import PdfMeta


from rich_docs.helper_classes.paper_sections import Abstract, References, Section, Position
from rich_docs.references.elaborator_class import ReferenceExtr
from rich_docs.references.base import Author, Reference
from rich_docs.helper_classes.basic import DocumentMetaData, PublicationPaperMetadata
from rich_docs.edocument_class import EDocument
import re


class ElaborateMarkdownPrediction:
    title_prefixes = ["## ", "##", "# ", "#"]

    def __init__(self, predictions: List[str], pdf: Path):
        self.predictions = predictions
        self.pdf = pdf
        self.abstract = None
        self.references = None
        self.reference_start = None
        self.reference_start_page = None
        self.abstract_end = None
        self.abstract_end_page = None

    def generate_Edoc(self) -> EDocument:
        asyncio.run(self._run())
        self.meta.references = self.references
        return EDocument(
            sections=self.sections,
            metadata=self.meta,
        )

    async def _run(self):
        logging.info("Extracting metadata ...")
        self.extract_metadata()
        logging.info("Start Extracting Updating The References with CrossRef...")
        if self.references is not None:
            await self.references.elaborate_references()
        logging.info("Breaking Into Sections ...")
        self.sections = self.extract_sections()
        logging.info("Waiting for References to be elaborated ...")

    def extract_section_pages(self) -> Tuple[List[str], str]:
        # the valid pages are the pages between the end of the abstract and the start of the references
        valid_pages = self.predictions[self.abstract_end_page : self.reference_start_page]
        # include the page of the references
        valid_pages.append(self.predictions[self.reference_start_page])

        # end at the start of the references
        valid_pages[-1] = valid_pages[-1][: self.reference_start]
        # # start from after the abstract
        # valid_pages[0] = valid_pages[0][self.abstract_end :]
        # get the introduction starting point
        intro_info = self.find_introduction_start(
            valid_pages,
            start_position=self.abstract_end,
        )

        if intro_info is None:
            valid_pages[0] = valid_pages[0][self.abstract_end :]
            return valid_pages, None

        numeral = intro_info["numeral"]
        intro_page = intro_info["page_number"]
        intro_start = intro_info["match_pos"]

        # save the real start position of the introduction
        self.numeral = numeral
        self.intro_page = intro_page
        self.intro_start = intro_start

        valid_pages = [valid_pages[intro_page][intro_start:]] + valid_pages[intro_page + 1 :]
        return valid_pages, numeral

    def check_if_sections_have_numeral(self, section_titles: List[str]) -> bool:
        # Regex pattern to match numerals (Arabic or Roman) potentially followed by punctuation
        numeral_pattern = r"^(?:\d+|[IVXLCDM]+)\.?"

        first_words = [title.split(" ")[0] for title in section_titles]

        # check how which sections have the first word that can be considered a numeral
        is_numeral = []
        for word in first_words:
            if re.match(numeral_pattern, word):
                is_numeral.append(True)
            else:
                is_numeral.append(False)

        return is_numeral

    def extract_sections(self) -> Dict[int, Section]:
        if self.abstract_end_page is None or self.reference_start_page is None:
            # return all as one section
            return {1: Section(start_position=(0, 0), end_position=(0, len(self.predictions[0])), text=" ".join(self.predictions), title="Full Text", number=1)}
        
        section_positions = self.find_all_section_positions(
            start_position=(self.abstract_end_page, self.abstract_end),
            end_position=(self.reference_start_page, self.reference_start),
        )

        # check if the sections have a numeral
        section_titles = [section[1] for section in section_positions.values()]
        isnumeral = self.check_if_sections_have_numeral(section_titles)

        numerals = []
        for isnum, title in zip(isnumeral, section_titles):
            if isnum:
                numeral = title.split(" ")[0]
            else:
                numeral = None
            numerals.append(numeral)

        self.numerals = numerals

        # section_positions.update({1: intro})
        # order the dictionary by the keys
        section_positions = {k: v for k, v in sorted(section_positions.items())}

        all_keys = list(section_positions.keys())
        if len(all_keys) == 0:
            logging.warning("No sections found")
            logging.warning("Returning the full text as one section")
            return {1: Section(start_position=(0, 0), end_position=(0, len(self.predictions[0])), text=" ".join(self.predictions), title="Full Text", number=1)}
        largest_section_number = max(section_positions.keys())
        

        sections = {}
        for jj, start_position in enumerate(section_positions.values()):
            section_number = all_keys[jj]
            if section_number == largest_section_number:
                end_position = Position(self.reference_start_page, self._reference_match_pos)
            else:
                next_section_number = all_keys[jj + 1]
                end_position = section_positions[next_section_number][0]

            start_page, start_pos = start_position[0]
            end_page, end_pos = end_position

            # get the real start position of the section text
            title = start_position[1]
            # the text start position is after the title

            if end_page == start_page:
                text = self.predictions[start_page][start_pos:end_pos]
                # the title is included in the beginning of the text start from the first newline
                text = text.split("\n\n", 1)[1]
            else:
                pages_of_interest = self.predictions[start_page : end_page + 1]
                pages_of_interest[0] = pages_of_interest[0][start_pos:]
                pages_of_interest[0] = pages_of_interest[0].split("\n\n", 1)[1] # title is included in the text. Remove it
                pages_of_interest[-1] = pages_of_interest[-1][:end_pos]
                text = " ".join(pages_of_interest)

            numeral = numerals[jj]

            if numeral is None:
                section_numeral_number = ""
            elif  numeral.isdigit():
                section_numeral_number = int(section_number)
            elif  numeral.isupper():
                section_numeral_number = roman.toRoman(section_number)
            elif  numeral.islower():
                section_numeral_number = roman.toRoman(section_number.upper()).lower()
            else:
                logging.warning(f"Numeral {numeral} is not recognized")
                section_numeral_number = str(jj+1)

            section_numeral_number = str(section_numeral_number) if section_numeral_number is not None else None

            title_nice = title.replace("#", "").strip()
            title_nice = title_nice.lstrip(section_numeral_number).replace("\n", "").strip()

            sections.update(
                {
                    section_number: Section(
                        start_position=start_position[0],
                        end_position=end_position,
                        text=text,
                        title=title_nice,
                        number=section_numeral_number,
                    )
                }
            )
        return sections

    def find_all_section_positions(
        self, start_position: Position, end_position: Position
    ) -> Dict[int, Tuple[Position, str]]:
        section_positions = {}

        start_page, start_pos = start_position
        end_page, end_pos = end_position

        pattern = r"\n##\s(?!#)(.*?)\n\n"
        section_regex = re.compile(pattern, re.IGNORECASE)

        section_number = 1
        for page_number in range(start_page, end_page + 1):
            page = self.predictions[page_number]

            # Determine the appropriate start and end positions for each page
            if page_number == start_page:
                current_start_pos = start_pos
            else:
                current_start_pos = 0

            if page_number == end_page:
                current_end_pos = end_pos
            else:
                current_end_pos = len(page)

            # Search within the specified segment of the page
            matches = section_regex.finditer(page[current_start_pos:current_end_pos])
            for match in matches:
                section_title = match.group(1).strip()  # Using group(1) to capture the title
                section_position = Position(page=page_number, pos=match.start() + current_start_pos)
                section_positions[section_number] = (section_position, section_title)
                section_number += 1

            # if the last value found is "references" remove the key
            if section_positions !={} and section_positions[section_number - 1][1].lower() == "references":
                section_positions.pop(section_number - 1)

        return section_positions

    def check_more_metadata_from_crossref(self, metadata: dict):
        title = metadata.get("title", None)
        if title is None:
            return metadata

        info = {
            "title": title,
            "year": metadata.get("year", None),
        }

        xinfo = ReferenceExtr.search_from_crossref(info)

        ## check if the title is in the crossref database
        if xinfo is None:
            return metadata

        if (
            xinfo.get("title") is None
            or xinfo.get("title")[0].lower().strip() != title.lower().strip()
        ):
            return metadata

        ## for logging
        xinfo_year = xinfo.get("created")
        if xinfo_year is not None:
            xinfo_year = xinfo_year.get("date-parts")
            if xinfo_year is not None:
                xinfo_year = xinfo_year[0][0]

        logging.info(
            "CrossRef search for {} returned info for {} ({})".format(
                title, xinfo.get("title"), xinfo_year
            )
        )

        publisher = xinfo.get("publisher")
        doi = xinfo.get("DOI")
        publication_type = xinfo.get("type")
        xinfo_author = xinfo.get("author", metadata.get("authors"))

        if xinfo_author is not None:
            authors = [
                Author(surname=a.get("family"), given_name=a.get("given")) for a in xinfo_author
            ]
        else:
            authors = info.get("authors")

        url = xinfo.get("URL")
        xkeywords = xinfo.get("subject")
        if xkeywords is None:
            keywords = metadata.get("keywords")
        else:
            keywords = [PdfMeta.capitalize_name(kw.replace(",", "").strip()) for kw in xkeywords]

        if xinfo.get("created") is not None:
            if xinfo.get("created").get("date-parts") is not None:
                creation_date_parts = xinfo.get("created").get("date-parts")[0]
                creation_date_parts = [int(d) for d in creation_date_parts]
                creation_date = datetime(*creation_date_parts)
                year = creation_date.year
            else:
                creation_date = None
                year = info.get("year")
        else:
            creation_date = None
            year = info.get("year")

        metadata.update(
            {
                "publisher": publisher,
                "publication_type": publication_type,
                "doi": doi,
                "url": url,
                "keywords": keywords,
                "date": creation_date,
                "authors": authors,
                "year": year,
            }
        )
        return metadata

    def extract_metadata(self) -> DocumentMetaData:
        embedded_meta = PdfMeta.extract_embedded_metadata(self.pdf)
        embedded_meta_dict = embedded_meta.to_dict_shallow()

        ## add keywords from doc metadata
        keywords = self.extract_keywords(self.pdf)
        embedded_meta_dict["keywords"] = keywords
        ## add filename and directory
        ## check if self.pdf is a full path or a relative path
        if self.pdf.is_absolute():
            embedded_meta_dict["file_directory"] = Path(str(self.pdf.parent))
        else:
            embedded_meta_dict["file_directory"] = Path.cwd() / self.pdf.parent
        embedded_meta_dict["filename"] = self.pdf.name

        logging.info(f"Loading more metadata from CrossRef for {embedded_meta_dict.get('title')}")
        complete_meta = self.check_more_metadata_from_crossref(embedded_meta_dict)

        references = self.extract_references()
        abstract = self.extract_abstract()

        self.meta = PublicationPaperMetadata(
            title=complete_meta.get("title"),
            authors=complete_meta.get("authors"),
            creation_date=complete_meta.get("creation_date"),
            pdf_directory=complete_meta.get("file_directory"),
            pdf_filename=complete_meta.get("filename"),
            doi=complete_meta.get("doi"),
            publisher=complete_meta.get("publisher"),
            publication_type=complete_meta.get("publication_type"),
            url=complete_meta.get("url"),
            keywords=complete_meta.get("keywords"),
            abstract=abstract,
            references=references,
        )

        return self.meta

    def extract_references(self) -> References:
        references_start = self.find_references_start()
        if references_start is None:
            self.references = None
            return
        start_page, key, match_pos = (
            references_start["page_number"],
            references_start["key_name"],
            references_start["match_pos"],
        )
        references_real_start = self.from_first_reference(self.predictions[start_page], match_pos)

        # save the real start position of the references
        self._reference_match_pos = match_pos
        self.reference_start = references_real_start
        self.reference_start_page = start_page

        references_pages = [
            self.predictions[start_page][references_real_start:]
        ] + self.predictions[start_page + 1 :]
        references_text = " ".join(references_pages)

        ref_dict = self.extract_ref_list_from_text(references_text)

        self.references = References(
            references=ref_dict,
            start_position=Position(page=start_page, pos=references_real_start),
            title=key,
        )
        return self.references

    @staticmethod
    def extract_ref_list_from_text(text: str) -> dict:
        # Regular expression to match reference patterns
        reference_regex = re.compile(r"\[(\d+)\](.*?)(?=\n\n|\n\*)", re.DOTALL)

        # Search for all matches in the text
        matches = reference_regex.findall(text)

        # Create a dictionary with reference numbers as keys and reference texts as values
        references = {int(num): ref.strip() for num, ref in matches}

        return references

    def extract_abstract(self) -> Abstract:
        abstract_start = self.find_abstract_start()
        if abstract_start is None:
            self.abstract = None
            return
        page, key, match_pos = (
            abstract_start["page_number"],
            abstract_start["key_name"],
            abstract_start["match_pos"],
        )
        abstract_alpha_numeric_start = self.from_next_alphanumeric(
            self.predictions[page], key, match_pos
        )
        abstract_end = self.find_abstract_end(
            self.predictions, key, page, abstract_alpha_numeric_start
        )

        # save the real end position of the abstract
        self.abstract_end = abstract_end
        self.abstract_end_page = page

        self.abstract = Abstract(
            start_position=Position(page=page, pos=abstract_alpha_numeric_start),
            end_position=Position(page=page, pos=abstract_end),
            text=self.predictions[page][abstract_alpha_numeric_start:abstract_end],
            title=key,
        )
        return self.abstract

    def extract_keywords(self, pdf: Path) -> Optional[List[str]]:
        with open(pdf, "rb") as f:
            reader = pypdf.PdfReader(f)
            info = reader.metadata
        keywords = None
        for key in ["/Keywords", "Keywords", "keywords", "KEYWORDS"]:
            if key in info.keys():
                keywords = info.get(key)
                break
        if keywords is not None:
            # split keywords into a list of strings
            keywords = re.split(r"[,;]", keywords)
            keywords = [PdfMeta.capitalize_name(kw.strip()) for kw in keywords]
        return keywords

    def find_abstract_end(
        self, predictions: List[str], keyword: str, page_num: int, start_position: int
    ) -> Dict:
        """
        Finds the end of an abstract in a given page. The abstract is assumed to start
        after the first alphanumeric character following the keyword, and ends at the first
        occurrence of two consecutive newlines.
        """
        page = predictions[page_num]

        # Find the first occurrence of two newlines to determine the end of the abstract
        end_regex = re.compile(r"\n\n")
        end_match = end_regex.search(page, start_position)
        if not end_match:
            return len(page)  # End not found, assume end of page

        # The position of the end of the abstract
        end_position = end_match.start()

        return end_position

    def find_reference_end(
        self, predictions: List[str], page_num: int, start_position: int
    ) -> Dict:
        """This function works the same way as 'find_abstract_end' but for the references section
        there is no hypothesis that the section should end on the same page where it starts.

        """
        for jj in range(page_num, len(predictions)):
            page = predictions[jj]
            # Find the first occurrence of two newlines to determine the end of the abstract
            end_regex = re.compile(r"\n\n")
            end_match = end_regex.search(page, start_position)
            if end_match:
                # The position of the end of the abstract
                end_position = end_match.start()
                return {
                    "page_number": jj,
                    "position": end_position,
                }
        # if no end is found, the end of the references is the end of the document
        return {
            "page_number": len(predictions),
            "position": len(predictions[-1]),
        }

    @staticmethod
    def from_next_alphanumeric(text: str, keyword: str, start_position: int) -> int:
        # Find the real start of the section
        # Use a regular expression to find the first alphanumeric character after the keyword
        start_regex = re.compile(
            r"{}\s*[^\S\r\n]*\S*([A-Za-z0-9])".format(re.escape(keyword)), re.DOTALL
        )
        start_match = start_regex.search(text, start_position)
        if not start_match:
            return -1  # Keyword or alphanumeric character not found

        # The actual starting position excluding the keyword
        return start_match.start(1)

    @staticmethod
    def from_first_reference(text: str, match_pos: int) -> int:
        # Regular expression to match common enumeration patterns like [1] or (1)
        enumeration_regex = re.compile(r"\[\d+\]|\(\d+\)")

        # Search for the pattern in the text
        match = enumeration_regex.search(text, match_pos)
        if match:
            # Return the start position of the first match
            return match.start()
        else:
            # Return -1 if no match is found
            return -1

    @staticmethod
    def determine_numeral_pattern(numeral_type: str, number: int) -> str:
        if numeral_type.isdigit():
            # Pattern for Arabic numerals: a word boundary, the numeral, possibly followed by a period, and another word boundary
            return rf"\b{number}(?:\.)?\b"
        elif numeral_type.isupper():
            # Convert the number to an uppercase Roman numeral and create a regex pattern
            roman_numeral = roman.toRoman(number)
            # Pattern for uppercase Roman numerals: word boundary, numeral, optionally a period, and another word boundary
            return rf"\b{roman_numeral}(?:\.)?\b"
        elif numeral_type.islower():
            # Convert the number to a lowercase Roman numeral and create a regex pattern
            roman_numeral = roman.toRoman(number).lower()
            # Pattern for lowercase Roman numerals: word boundary, numeral, optionally a period, and another word boundary
            return rf"\b{roman_numeral}(?:\.)?\b"
        else:
            return ""

    def find_introduction_start(self, pages: List[str], start_position: int) -> Optional[Dict]:
        key_names = ["introduction"]
        regex_template = "##\s*(\d+|\d+\)|\d+\.|I+\.?|i+\.?)?\s*{}\n\n"

        # go through the pages in reverse order
        for name in key_names:
            intro_regex = re.compile(regex_template.format(name), re.IGNORECASE)
            for page_number in range(len(pages)):
                current_page = pages[page_number]
                match = intro_regex.search(current_page, start_position)
                if match:
                    return {
                        "page_number": page_number,
                        "key_name": match.group(0),
                        "match_pos": match.start(),
                        "numeral": match.group(1) or "",
                    }

    def find_references_start(self) -> Optional[Dict]:
        key_names = self.generate_keyname_variations("References")

        # go through the pages in reverse order
        for name in key_names:
            for page_number in range(len(self.predictions) - 1, -1, -1):
                current_page = self.predictions[page_number]
                match_pos = self.find_title_match_in_text(current_page, name)
                if match_pos is not None:
                    return {
                        "page_number": page_number,
                        "key_name": name,
                        "match_pos": match_pos,
                    }

    def find_abstract_start(self) -> Optional[Dict]:
        key_names = self.generate_keyname_variations("Abstract")

        # go through the pages in reverse order
        for name in key_names:
            for page_number in range(len(self.predictions)):
                current_page = self.predictions[page_number]
                if name in current_page:
                    return {
                        "page_number": page_number,
                        "key_name": name,
                        "match_pos": current_page.find(name),
                    }

    @staticmethod
    def find_title_match_in_text(text: str, keyname: str) -> Optional[int]:
        """Returns the startposition of the keyname in the text."""
        # Define a regular expression pattern for matching the keyname at the beginning of a line
        pattern = r"^.*?" + re.escape(keyname) + r"\b.*?$"

        # Use re.MULTILINE to search for the pattern in each line of the text
        matches = list(re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE))

        if matches:
            # Get the start position of the first match
            start = matches[0].start()
            return start

        return None

    @staticmethod
    def generate_keyname_variations(keyname: str) -> List[str]:
        """Returns a list of strings that are variations of the keyname.
        Note that the order of the symbols is important as they will be used in this
        same order to search for the keyname in the text.
        """
        keyname_variations = []
        for s in ElaborateMarkdownPrediction.title_prefixes:
            keyname_variations.append(s + keyname)
        return keyname_variations
