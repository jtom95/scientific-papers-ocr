from datetime import datetime
from typing import List, Optional
import logging

import asyncio
from concurrent.futures import ThreadPoolExecutor
from habanero import Crossref
from bs4 import BeautifulSoup
import requests

from refextract import extract_references_from_string
from .base import Reference, Author, SearchURLs



class ReferenceExtr:
    def __init__(self, ref: str, ref_number: Optional[int] = None):
        self.ref = ref
        self.doi = None
        self.ref_number = ref_number

    async def async_run(self) -> Reference:
        info = self.extract_info_from_string()

        if info.get("title") is None:
            return Reference(
                raw=self.ref,
                authors=info.get("authors"),
                year=info.get("year"),
                ref_number=self.ref_number,
            )

        xinfo = await self.async_search_from_crossref(info)
        if xinfo is None:
            return Reference(
                title=info.get("title"),
                authors=info.get("authors"),
                year=info.get("year"),
                raw=self.ref,
                ref_number=self.ref_number,
            )
        publisher = xinfo.get("publisher")
        doi = xinfo.get("DOI")
        publication_type = xinfo.get("type")
        xinfo_author = xinfo.get("author")

        if xinfo_author is not None:
            authors = [
                Author(surname=a.get("family"), given_name=a.get("given")) for a in xinfo_author
            ]
        else:
            authors = info.get("authors")

        url = xinfo.get("URL")
        keywords = xinfo.get("subject")
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

        return Reference(
            title=info.get("title"),
            authors=authors,
            publisher=publisher,
            publication_type=publication_type,
            year=year,
            date=creation_date,
            doi=doi,
            url=url,
            keywords=keywords,
            raw=self.ref,
            ref_number=self.ref_number,
        )

    def run(self) -> Reference:
        info = self.extract_info_from_string()

        if info.get("title") is None:
            return Reference(raw=self.ref, authors=info.get("authors"), year=info.get("year"))

        xinfo = self.search_from_crossref(info)
        if xinfo is None:
            return Reference(
                title=info.get("title"),
                authors=info.get("authors"),
                year=info.get("year"),
                raw=self.ref,
                ref_number=self.ref_number,
            )
        publisher = xinfo.get("publisher")
        doi = xinfo.get("DOI")
        publication_type = xinfo.get("type")
        xinfo_author = xinfo.get("author")

        if xinfo_author is not None:
            authors = [
                Author(surname=a.get("family"), given_name=a.get("given")) for a in xinfo_author
            ]
        else:
            authors = info.get("authors")

        url = xinfo.get("URL")
        keywords = xinfo.get("subject")
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

        return Reference(
            title=info.get("title"),
            authors=authors,
            publisher=publisher,
            publication_type=publication_type,
            year=year,
            date=creation_date,
            doi=doi,
            url=url,
            keywords=keywords,
            raw=self.ref,
            ref_number=self.ref_number,
        )

    async def async_search_from_crossref(self, info):
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor() as pool:
            return await loop.run_in_executor(pool, self.search_from_crossref, info)

    @staticmethod
    def search_from_crossref(info: dict, limit=3) -> Optional[dict]:
        cr = Crossref()
        title = info.get("title")
        year = info.get("year")

        if year is not None:
            search_filter = {"from-pub-date": str(year), "until-pub-date": str(year)}
        else:
            search_filter = None

        result = cr.works(query=title, filter=search_filter, limit=limit)
        
        if result and "message" in result and "items" in result["message"]:
            items = result["message"]["items"]
            if len(items) == 1:
                return items[0]
            # find if any of the items has exactly the same title
            for item in items:
                if item.get("title", None) is None:
                    continue
                if item.get("title")[0].strip().lower() == title.strip().lower():
                    return item
                
            logging.info(f"The returned items do not have the same title as {title}")
            logging.info(f"titles: {[item.get('title') for item in items]}")
            return None
        else: 
            logging.info(f"Could not find DOI for {title}")
            logging.info(f"Result: {result}")
            return None
        # if (
        #     result
        #     and "message" in result
        #     and "items" in result["message"]
        #     and result["message"]["items"]
        # ):
        #     return result["message"]["items"][0]
        # else:
        #     return None

    def extract_info_from_string(self) -> dict:
        # remove _ from string
        ref = self.ref.replace("_", "")
        info = extract_references_from_string(ref)[0]
        authors = info.get("author")
        title = info.get("title")
        year = info.get("year")
        if authors is not None:
            authors = authors[0].split(",")
            # remove "and "
            authors = [a.strip() for a in authors]
            if len(authors) == 1:
                authors = authors[0].split(" and ")
            authors = [a.replace("and", "").strip() for a in authors]

        if title is not None:
            title = title[0]
        if year is not None:
            year = int(year[0])

        if title is None:
            if authors is None:
                pass
            else:
                parts = ref.split(",")
                for p in parts:
                    if not any([auth in p for auth in authors]):
                        title = p
                        break

        return {
            "authors": authors,
            "title": title,
            "year": year,
        }

    def search_doi_online(self, site: SearchURLs = SearchURLs.scholar):
        if isinstance(site, str):
            site = SearchURLs(site)

        query_url = site.value.format(self.ref)
        response = requests.get(query_url)
        if response.status_code != 200:
            return self.parse_results(response.content)
        else:
            logging.info(f"Could not find DOI for {self.ref}")
            return None

    def parse_results(self, html_content):
        soup = BeautifulSoup(html_content, "html.parser")
        return soup


if __name__ == "__main__":
    url_test = r'Z. Yu, J. A. Mix, S. Sajuyigke, K. P. Slattery, and J. Fan, "An improved dipole-moment model based on near-field scanning for characterizing near-field coupling and far-field radiation from nC," _IEEE Trans. Electromagn. Comput._, vol. 55, no. 1, pp. 97-108, Feb. 2013.'

    ref = ReferenceExtr(url_test)
    ref.search_doi_online()
