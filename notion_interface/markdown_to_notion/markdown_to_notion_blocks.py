from typing import List, Optional, Tuple, Dict
import re
import logging

from .mathpix_to_katex import LatexToKatexEquationTranslator
from ..basics.constants import DictBlockTypes, DictBlock
from typing import List, Optional
from pprint import pprint


class _MarkdownToRichText:
    def format_rich_text(self, text: str):
        """Formats the given text into rich text with various styles."""
        # First format math equations, since we don't want to format inside these
        text_with_equations_formatted = self._format_math_equations(text)
        # Split the text into equation and non-equation parts
        segments = self._split_equation_and_non_equation_parts(text_with_equations_formatted)
        # Apply other formats to non-equation parts
        formatted_segments = [
            self._format_non_equation_segment(segment) if not is_equation else segment
            for segment, is_equation in segments
        ]
        # Concatenate the segments back together
        formatted_text = "".join(formatted_segments)
        return self._split_text_into_rich_text(formatted_text)

    @staticmethod
    def _split_equation_and_non_equation_parts(text: str) -> List[Tuple[str, bool]]:
        """Splits text into segments of equation and non-equation parts."""
        pattern = r"<equation>.+?</equation>"
        segments = []
        start = 0
        for match in re.finditer(pattern, text):
            # Add the non-equation part
            segments.append((text[start : match.start()], False))
            # Add the equation part
            segments.append((match.group(), True))
            start = match.end()
        # Add the remaining non-equation part
        segments.append((text[start:], False))
        return segments

    def _format_non_equation_segment(self, segment: str) -> str:
        """Applies formatting to non-equation segments."""
        return self._format_italic(self._format_bold(self._format_code(segment)))

    @staticmethod
    def _format_math_equations(text: str):
        """Formats math equations enclosed in \( and \)."""
        return re.sub(r"\\\((.+?)\\\)", r"<equation>\1</equation>", text)

    @staticmethod
    def _format_code(text: str):
        """Formats code enclosed in ` and `."""
        return re.sub(r"`(.+?)`", r"<code>\1</code>", text)

    @staticmethod
    def _format_bold(text: str):
        """Formats bold text enclosed in ** and **."""
        return re.sub(r"\*\*(.+?)\*\*", r"<bold>\1</bold>", text)

    @staticmethod
    def _format_italic(text: str):
        """Formats italic text enclosed in * and * or _ and _, without leading/trailing whitespace."""
        # This pattern matches text between * or _, ensuring no leading/trailing whitespace
        return re.sub(r"(?<!\w)(\*|_)(\S.+?\S)\1(?!\w)", r"<italic>\2</italic>", text)

    @staticmethod
    def _split_text_into_rich_text(formatted_text: str):
        """Splits the formatted text into rich text objects for Notion."""
        tags = {
            "<equation>": {"type": "equation", "element": "expression"},
            "<code>": {"type": "text", "annotations": {"code": True}},
            "<bold>": {"type": "text", "annotations": {"bold": True}},
            "<italic>": {"type": "text", "annotations": {"italic": True}},
        }

        rich_text = []
        # More specific regular expression for splitting text
        parts = re.split(
            r"(\<equation\>)|(\<\/equation\>)|(\<code\>)|(\<\/code\>)|(\<bold\>)|(\<\/bold\>)|(\<italic\>)|(\<\/italic\>)",
            formatted_text,
        )

        open_tag = None
        for part in parts:
            if part is None:
                continue
            if part in tags:  # Opening tag
                open_tag = part
            elif part.startswith("</") and open_tag:  # Closing tag
                open_tag = None
            else:  # Text content
                if open_tag:
                    tag_info = tags[open_tag]
                    if tag_info["type"] == "equation":
                        if not part.strip():
                            continue  # Skip empty equations
                        else:
                            rich_text.append(
                                {
                                    "type": "equation",
                                    "equation": {
                                        "expression": LatexToKatexEquationTranslator(part).parse(),
                                    },
                                }
                            )
                    else:
                        if not part.strip():
                            continue  # Skip empty text
                        rich_text.append(
                            {
                                "type": tag_info["type"],
                                tag_info["type"]: {
                                    "content": part,
                                },
                                "annotations": tag_info.get("annotations", {}),
                            }
                        )
                else:
                    # Regular text
                    rich_text.append({"type": "text", "text": {"content": part}})

        return rich_text


class _TextSplitter:
    def __init__(self, max_size=1000):
        self.max_size = max_size

    def find_nearest_split(self, text, pattern):
        """Find the nearest split to max_size based on the pattern."""
        parts = re.split(f"({pattern})", text)
        temp_chunk = ""
        last_valid_split = None

        for i, part in enumerate(parts):
            if len(temp_chunk + part) <= self.max_size:
                temp_chunk += part
                last_valid_split = i
            else:
                break

        if last_valid_split is not None:
            split_point = "".join(parts[: last_valid_split + 1])
            return split_point, text[len(split_point) :]
        return text, ""

    def split_after_period(self, text):
        """Splits the text after the nearest valid period to max_size."""
        pattern = r"\. (?![0-9])"
        matches = list(re.finditer(pattern, text))
        temp_chunk = ""

        for match in matches:
            if len(temp_chunk) + len(match.group(0)) <= self.max_size:
                temp_chunk = text[: match.end()]
            else:
                break

        if temp_chunk:
            return temp_chunk, text[len(temp_chunk) :]
        return text, ""

    def find_best_split(self, text):
        """Finds the best place to split the text."""
        for pattern in ["\n\n", "\n"]:
            if pattern in text:
                chunk, remaining = self.find_nearest_split(text, pattern)
                if chunk != text:
                    return chunk, remaining

        return self.split_after_period(text)

    def split_text(self, text):
        """Splits text into chunks, each within max_size characters."""
        if len(text) <= self.max_size:
            return [text]

        chunks = []
        while len(text) > self.max_size:
            chunk, text = self.find_best_split(text)
            if len(chunk) > self.max_size:  # If no suitable split found
                chunk = text[: self.max_size]
                text = text[self.max_size :]
            chunks.append(chunk)

        if text:
            chunks.append(text)
        return chunks

    def split_text(self, text):
        """Splits text into chunks for Notion API, each within min_size characters."""
        if len(text) <= self.max_size:
            return [text]

        chunks = []
        while len(text) > self.max_size:
            chunk, text = self.find_best_split(text)
            if chunk == text:  # No suitable split found
                chunk = text[: self.max_size]
                text = text[self.max_size :]
            chunks.append(chunk)
        if text:
            chunks.append(text)
        return chunks


class MarkdownToNotionPayload:
    """
    Collection of functions to convert markdown to Notion API payloads.
    """

    @staticmethod
    def add_annotation_to_rich_text_block(payload_block: dict, **kwargs):
        if payload_block.get("paragraph") == None:
            logging.error(
                "Failed to add annotation to text: payload block does not have a paragraph key"
            )
            return payload_block
        if payload_block["paragraph"].get("rich_text") == None:
            logging.error(
                "Failed to add annotation to text: payload block does not have a rich_text key"
            )
            return payload_block

        rich_text_list = payload_block["paragraph"]["rich_text"]
        for rich_text in rich_text_list:
            if rich_text.get("annotations") == None:
                rich_text.update({"annotations": {}})
            rich_text["annotations"].update(kwargs)
        payload_block["paragraph"]["rich_text"] = rich_text_list
        return payload_block

    @staticmethod
    def text_to_rich_text_payload(text: str):
        markdown_to_rich_text = _MarkdownToRichText()
        payload = {
            "children": [
                {
                    "object": "block",
                    "type": "paragraph",
                    "paragraph": {
                        "rich_text": markdown_to_rich_text.format_rich_text(text),
                    },
                }
            ]
        }
        return payload

    @staticmethod
    def math_text_to_math_block_payload(text: str):
        payload = {
            "children": [
                {
                    "object": "block",
                    "type": "equation",
                    "equation": {
                        "expression": text,
                        # "format": "mathml",
                    },
                }
            ]
        }
        return payload

    @staticmethod
    def text_to_header_payload(text: str, header_level: int):
        if header_level not in [1, 2, 3]:
            raise ValueError("Header level must be 1, 2, or 3")
        payload = {
            "children": [
                {
                    "object": "block",
                    "type": f"heading_{header_level}",
                    f"heading_{header_level}": {
                        "rich_text": _MarkdownToRichText().format_rich_text(text)
                    },
                }
            ]
        }
        return payload

    @staticmethod
    def split_mathpix_markdown_into_blocks(
        mathpix_markdown: str, max_len_block: Optional[int] = None
    ) -> List[DictBlock]:
        ## split into paragraphs at the double newlines
        paragraphs = mathpix_markdown.split("\n\n")
        ## split into blocks at the math delimiters

        blocks = []
        for paragraph in paragraphs:
            (
                math_open_positions,
                math_close_positions,
            ) = MarkdownToNotionPayload.get_mathblock_positions(paragraph)
            if math_open_positions is None and math_close_positions is None:
                blocks.append(DictBlock(type=DictBlockTypes.TEXT, content=paragraph))
                continue
            # Initial position for slicing text blocks
            last_position = 0

            for open_pos, close_pos in zip(math_open_positions, math_close_positions):
                # Add the text block before the math block
                blocks.append(
                    DictBlock(type=DictBlockTypes.TEXT, content=paragraph[last_position:open_pos])
                )
                blocks.append(
                    DictBlock(type=DictBlockTypes.MATH, content=paragraph[open_pos:close_pos])
                )
                # Update the last position
                last_position = close_pos

            # Add any remaining text after the last math block
            if last_position < len(paragraph):
                blocks.append(
                    DictBlock(type=DictBlockTypes.TEXT, content=paragraph[last_position:])
                )

        cleaned_blocks = MarkdownToNotionPayload.parse_blocks(
            blocks, max_length_block=max_len_block
        )
        blocks_with_headers = MarkdownToNotionPayload.parse_header_blocks(cleaned_blocks)
        return blocks_with_headers

    @staticmethod
    def parse_header_blocks(blocks: List[DictBlock]):
        parsed_blocks = []
        for block in blocks:
            if block["type"] == DictBlockTypes.TEXT:
                content = block["content"]
                if content.startswith("###"):
                    block_type = DictBlockTypes.HEADER3
                    content = content[3:].strip()  # Remove the '### ' prefix
                elif content.startswith("##"):
                    block_type = DictBlockTypes.HEADER2
                    content = content[2:].strip()  # Remove the '## ' prefix
                elif content.startswith("#"):
                    block_type = DictBlockTypes.HEADER1
                    content = content[1:].strip()  # Remove the '# ' prefix
                else:
                    block_type = block["type"]
                parsed_blocks.append(DictBlock(type=block_type, content=content))
            else:
                parsed_blocks.append(block)
        return parsed_blocks

    @staticmethod
    def parse_blocks(blocks: List[DictBlock], max_length_block: Optional[int] = None):
        for block in blocks:
            if block["content"] != "":
                # check if text is too long

                if block["type"] == DictBlockTypes.MATH:
                    math_text = block["content"]
                    parsed_math_text = math_text.replace("\[", "").replace("\]", "")
                    parsed_math_text = LatexToKatexEquationTranslator(parsed_math_text).parse()
                    block.update({"content": parsed_math_text})

                if block["content"].strip() == "":
                    continue

                if max_length_block is not None and len(block["content"]) > max_length_block:
                    logging.info("Text block too long, splitting")
                    text = _TextSplitter(max_size=max_length_block).split_text(block["content"])
                    # if the text is split, yield each part
                    for t in text:
                        if t == "":
                            continue
                        block.update({"content": t})
                        yield block
                else:
                    yield block

    @staticmethod
    def get_mathblock_positions(text: str) -> Tuple[List[int], List[int]]:
        """
        The math block is the text between the math delimiters \[ and \]. Note there may be more than one math block in the text.
        """
        opener_positions = re.finditer(r"\\\[", text)
        closer_positions = re.finditer(r"\\\]", text)

        opener_positions = [m.start() for m in opener_positions]
        closer_positions = [m.end() for m in closer_positions]

        if opener_positions == [] and closer_positions == []:
            return None, None

        # Removing extra openers or closers
        positions = sorted(opener_positions + closer_positions)
        # the positions should be opened first, then closed and so on:
        # if there are two openers or two closers in a row, the first one is the valid one
        valid_openers = []
        valid_closers = []

        equation_open = False
        for i, p in enumerate(positions):
            if equation_open:
                if p in closer_positions:
                    valid_closers.append(p)
                    equation_open = False
                else:
                    logging.info("Unopened mathblock found: skipping")
            else:
                if p in opener_positions:
                    valid_openers.append(p)
                    equation_open = True
                else:
                    logging.info("Unopened mathblock found: skipping")

        return valid_openers, valid_closers
