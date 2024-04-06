from typing import Dict, List, Tuple, Optional
from collections import namedtuple
import pytesseract
import logging
from PIL import Image
import pdf2image
import pytesseract
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from dataclasses import dataclass, fields
from pathlib import Path
from pprint import pprint
from spellchecker import SpellChecker
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
import string
import fitz
from sentence_transformers import SentenceTransformer


from .static_method_classes import AnalyzeTranscriptMethods, TextUtils

# Pretrained BERT model local path
BERT_MODEL_PATH = Path(r"D:\stored_models\bert\all-MiniLM-L6-v2")
URL = "all-MiniLM-L6-v2"


@dataclass
class PatchBlock:
    x: int
    y: int
    w: int
    h: int

    @property
    def coordinates(self):
        """returns coords in clockwise order starting from top left"""
        return (
            (self.x, self.y),
            (self.x, self.y + self.h),
            (self.x + self.w, self.y + self.h),
            (
                self.x + self.w,
                self.y,
            ),
        )

    @property
    def area(self):
        return self.w * self.h

    def to_dict(self):
        return {
            "x": self.x,
            "y": self.y,
            "w": self.w,
            "h": self.h,
        }


@dataclass
class CheckPageMissingConfigs:
    num_meaningful_sentences: int = 3
    sentence_length: int = 6
    max_non_alpha: int = 1
    min_length_word: int = 2
    min_alpha_ratio: float = 0.7


class CheckPageMissingParts:
    def __init__(self, pdf: Path, page_number: int, transcription: str, dpi=300, **kwargs):
        self.pdf = pdf
        self.transcription = transcription
        self.page_number = page_number
        self.dpi = dpi  # dots per inch
        self.patch_blocks: Dict[int, PatchBlock] = {}
        self.configs = CheckPageMissingConfigs(**kwargs)
        self.incomplete = False
        self.page_block = None

    @property
    def max_score(self):
        return self.configs.num_meaningful_sentences

    @property
    def threshold(self):
        return self.max_score - 1

    def run(self):
        is_page_incomplete = self.is_page_incomplete()
        self.page_block = self.extract_page_block()
        if is_page_incomplete:
            self.incomplete = True
            self.incomplete_blocks = [
                k for k, score in self.scores.items() if 0 <= score < self.threshold
            ]

            self.incomplete_patch_blocks = {
                block_number: self.patch_blocks[block_number]
                for block_number in self.incomplete_blocks
            }

            self.complete_patch_blocks = {
                block_number: self.patch_blocks[block_number]
                for block_number in self.scores.keys()
                if block_number not in self.incomplete_blocks
            }

        else:
            self.incomplete = False
            self.incomplete_blocks = []
            self.incomplete_patch_blocks = {}
            self.complete_patch_blocks = self.patch_blocks

        return self

    def extract_page_block(self):
        if not hasattr(self, "page_image") or not hasattr(self, "tesseract_data"):
            logging.error("Please run create_text_blocks_OCR first")
            return
        data = self.tesseract_data
        # Draw bounding boxes around the text blocks
        for i in range(len(data["level"])):
            if data["level"][i] == 1:
                (x, y, w, h) = (
                    data["left"][i],
                    data["top"][i],
                    data["width"][i],
                    data["height"][i],
                )
                return PatchBlock(x, y, w, h)

    def draw_incomplete_blocks(self, ax=None, color="red"):
        if ax is None:
            fig, ax = plt.subplots(1, figsize=(12, 8))
            ax.imshow(self.page_image)
        data = self.tesseract_data

        # Draw bounding boxes around the text blocks
        for block_number, patch in self.incomplete_patch_blocks.items():
            rect = patches.Rectangle(
                (patch.x, patch.y),
                patch.w,
                patch.h,
                linewidth=1,
                edgecolor=color,
                facecolor="none",
            )
            ax.add_patch(rect)
            # Add the text block number
            ax.text(
                patch.x,
                patch.y,
                str(block_number),
            )
        return ax

    def get_patch_blocks(self) -> Dict[int, PatchBlock]:
        if self.tesseract_data is None:
            return {}
        patch_blocks = {}
        data = self.tesseract_data
        for i in range(len(data["level"])):
            if data["level"][i] == 2:  # Level 2 corresponds to block level
                (x, y, w, h) = (
                    data["left"][i],
                    data["top"][i],
                    data["width"][i],
                    data["height"][i],
                )
                patch_blocks[data["block_num"][i]] = PatchBlock(x, y, w, h)
        return patch_blocks

    def is_page_incomplete(self):
        self.create_text_blocks_OCR()
        self.assign_scores()

        if any(0 <= score < self.threshold for score in self.scores.values()):
            return True
        else:
            return False

    def assign_scores(self, blocks: Optional[Dict[str, str]] = None):
        if blocks is None:
            blocks = self.blocks
        scores = {}

        for block_number, block in blocks.items():
            if not AnalyzeTranscriptMethods.is_meaningful(block):
                scores[block_number] = -1
                continue
            if not AnalyzeTranscriptMethods.is_long_enough(block):
                scores[block_number] = -1
                continue

            scores[block_number] = AnalyzeTranscriptMethods.return_block_is_present_score(
                block,
                self.transcription,
                max_non_alpha=self.configs.max_non_alpha,
                sentence_length=self.configs.sentence_length,
                num_meaningful_sentences=self.configs.num_meaningful_sentences,
                min_word_length=self.configs.min_length_word,
            )
        self.scores = scores
        return self.scores

    def create_text_blocks_fitz(self):
        self.blocks = AnalyzeTranscriptMethods.extract_text_by_blocks_fitz(
            self.pdf, page_number=self.page_number
        )
        return self.blocks

    def create_text_blocks_OCR(self):
        # get the page image
        logging.info(f"Extracting page {self.page_number} from {self.pdf}")
        self.page_image = pdf2image.convert_from_path(
            self.pdf, dpi=self.dpi, first_page=self.page_number + 1, last_page=self.page_number + 1
        )[0]
        self.page_image = self.page_image.convert(
            "RGB"
        )  # Convert the PIL Image to a format pytesseract can understand
        logging.info(f"Extracting text from page {self.page_number} - tesseract")
        self.tesseract_data = pytesseract.image_to_data(
            self.page_image, output_type=pytesseract.Output.DICT
        )
        self.blocks = AnalyzeTranscriptMethods.extract_text_by_blocks_OCR(self.tesseract_data)
        self.patch_blocks = self.get_patch_blocks()
        return self.blocks

    def draw_page_block(self):
        fig, ax = plt.subplots(1, figsize=(12, 8))
        ax.imshow(self.page_image)
        data = self.tesseract_data
        # Draw bounding boxes around the text blocks
        for i in range(len(data["level"])):
            if data["level"][i] == 1:  # Level 1 corresponds to page level
                (x, y, w, h) = (
                    data["left"][i],
                    data["top"][i],
                    data["width"][i],
                    data["height"][i],
                )
                rect = patches.Rectangle(
                    (x, y), w, h, fill=False, color="blue", linewidth=2
                )  # Use blue to differentiate blocks
                ax.add_patch(rect)
        return ax

    def draw_blocks(self):
        fig, ax = plt.subplots(1, figsize=(12, 8))
        ax.imshow(self.page_image)
        data = self.tesseract_data
        # Draw bounding boxes around the text blocks
        for i in range(len(data["level"])):
            if data["level"][i] == 2:  # Level 2 corresponds to block level
                (x, y, w, h) = (
                    data["left"][i],
                    data["top"][i],
                    data["width"][i],
                    data["height"][i],
                )
                rect = patches.Rectangle(
                    (x, y), w, h, fill=False, color="blue", linewidth=2
                )  # Use blue to differentiate blocks
                ax.add_patch(rect)
        return ax

    def draw_single_block(self, block_number: int, ax=None, color="red"):
        if ax is None:
            _, ax = plt.subplots(1, figsize=(12, 8))
            ax.imshow(self.page_image)
        data = self.tesseract_data
        # Draw bounding boxes around the text blocks
        for i in range(len(data["level"])):
            if data["level"][i] == 2 and data["block_num"][i] == block_number:
                (x, y, w, h) = (
                    data["left"][i],
                    data["top"][i],
                    data["width"][i],
                    data["height"][i],
                )
                rect = patches.Rectangle(
                    (x, y), w, h, fill=False, color=color, linewidth=2
                )  # Use blue to differentiate blocks
                ax.add_patch(rect)
        return ax

    def clean_blocks(self):
        parsed = {}
        for block_number, block in self.blocks.items():
            parsed_block = AnalyzeTranscriptMethods.remove_non_alphanumeric(block)
            parsed_block = AnalyzeTranscriptMethods.remove_misspelled_words(parsed_block)
            if AnalyzeTranscriptMethods.is_meaningful(
                parsed_block
            ) and AnalyzeTranscriptMethods.is_long_enough(parsed_block):
                parsed[block_number] = parsed_block
        self.parsed_blocks = parsed
        return self.parsed_blocks

    def load_model(self, model_path: Optional[Path] = None):
        if model_path is None:
            model_path = BERT_MODEL_PATH
        self.model = SentenceTransformer(URL)
        return self.model

    def find_similar_segments(self, transcription: str, block_text: str):
        return AnalyzeTranscriptMethods.find_most_similar_segment(
            transcription, block_text, self.model
        )
