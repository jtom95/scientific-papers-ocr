from dataclasses import dataclass
from .constants import DatabasePropertyTypes


@dataclass
class SciProperties:
    title: DatabasePropertyTypes = DatabasePropertyTypes.title
    authors: DatabasePropertyTypes = DatabasePropertyTypes.multi_select
    publication_date: DatabasePropertyTypes = DatabasePropertyTypes.rich_text
    publisher: DatabasePropertyTypes = DatabasePropertyTypes.select
    publication_type: DatabasePropertyTypes = DatabasePropertyTypes.select
    doi: DatabasePropertyTypes = DatabasePropertyTypes.url
    url: DatabasePropertyTypes = DatabasePropertyTypes.url
    keywords: DatabasePropertyTypes = DatabasePropertyTypes.multi_select

@dataclass
class ReferenceProperties(SciProperties):
    year: DatabasePropertyTypes = DatabasePropertyTypes.number
    ref_number: DatabasePropertyTypes = DatabasePropertyTypes.number
    citation: DatabasePropertyTypes = DatabasePropertyTypes.rich_text


@dataclass
class ScientificDatabaseProperties(SciProperties):
    keywords: DatabasePropertyTypes = DatabasePropertyTypes.multi_select
    file_directory: DatabasePropertyTypes = DatabasePropertyTypes.rich_text
    filename: DatabasePropertyTypes = DatabasePropertyTypes.rich_text
    
    ##
    ADDED: DatabasePropertyTypes = DatabasePropertyTypes.date
    STATUS: DatabasePropertyTypes = DatabasePropertyTypes.select
    HOT: DatabasePropertyTypes = DatabasePropertyTypes.checkbox
