from typing import List, Optional, Tuple, Dict
from enum import Enum
import pandas as pd
from PIL import Image
from pathlib import Path
import logging
from .check_single_page import CheckPageMissingParts, PatchBlock


class PatchBlockType(Enum):
    HALF_COLUMN_LEFT = "half_left"
    HALF_COLUMN_RIGHT = "half_right"
    FULL = "full"
    OTHER = "other"
    
    @staticmethod
    def get_block_type(block_patch: PatchBlock, page_block: PatchBlock):
        # compare the length of the incomplete block with the page block
        if block_patch.w < 0.5 * page_block.w:
            if block_patch.x < 0.5 * page_block.w:
                if block_patch.x + block_patch.w < 0.5 * page_block.w:
                    return PatchBlockType.HALF_COLUMN_LEFT
                else:
                    return PatchBlockType.OTHER
            else:
                return PatchBlockType.HALF_COLUMN_RIGHT
        else:
            return PatchBlockType.FULL


class PageLayoutAnalyzer:
    def __init__(self, checker: CheckPageMissingParts):
        self.checker = checker
        if not hasattr(self.checker, "complete_patch_blocks"):
            self.checker.run()
        self.patch_blocks = checker.patch_blocks
        self.page_block = checker.page_block
        self.incomplete = checker.incomplete
        self.page_image = checker.page_image

    def get_blocks_in_columns(self):
        btypes = {
            k: PatchBlockType.get_block_type(block, self.page_block)
            for k, block in self.patch_blocks.items()
        }

        right_column = {
            k: self.patch_blocks[k]
            for k, v in btypes.items()
            if v == PatchBlockType.HALF_COLUMN_RIGHT
        }
        left_column = {
            k: self.patch_blocks[k]
            for k, v in btypes.items()
            if v == PatchBlockType.HALF_COLUMN_LEFT
        }
        return {
            "right": right_column,
            "left": left_column,
        }

    def get_column_dataframe(self):
        blocks_in_columns = self.get_blocks_in_columns()
        right_column_dict = {k: patch.to_dict() for k, patch in blocks_in_columns["right"].items()}
        left_column_dict = {k: patch.to_dict() for k, patch in blocks_in_columns["left"].items()}

        right_column_pd = pd.DataFrame(right_column_dict).T
        left_column_pd = pd.DataFrame(left_column_dict).T
        
        if right_column_pd.empty or left_column_pd.empty:
            return None

        # the column names are x, y, w, h
        right_column_pd.columns = ["x", "y", "w", "h"]
        left_column_pd.columns = ["x", "y", "w", "h"]

        right_column_pd["x2"] = right_column_pd["x"] + right_column_pd["w"]
        left_column_pd["x2"] = left_column_pd["x"] + left_column_pd["w"]
        right_column_pd["y2"] = right_column_pd["y"] + right_column_pd["h"]
        left_column_pd["y2"] = left_column_pd["y"] + left_column_pd["h"]
        return {
            "right": right_column_pd,
            "left": left_column_pd,
        }

    def get_column_patches(self):
        columns_df = self.get_column_dataframe()
        if columns_df is None:
            return None
        right_column = columns_df["right"]
        left_column = columns_df["left"]
        return {
            "right": self._get_column_bounds(right_column),
            "left": self._get_column_bounds(left_column),
        }

    @staticmethod
    def _get_column_bounds(column_df: pd.DataFrame) -> PatchBlock:
        # get the rightmost point right_column
        rightmost_x = column_df["x2"].max()
        # get leftmost point right_column
        leftmost_x = column_df["x"].min()
        # get the topmost point right_column
        bottommost_y = column_df["y2"].max()
        # get the bottommost point right_column
        topmost_y = column_df["y"].min()

        return PatchBlock(
            x=leftmost_x,
            y=bottommost_y,
            w=rightmost_x - leftmost_x,
            h=topmost_y - bottommost_y,
        )

    def generate_new_pages(self, patch_block_A: List[int], patch_block_B: List[int]):
        output_size = self.page_image.size  # keep the same size as the original image
        patch_block_A = self.create_patch_image(patch_block_A, output_size)
        patch_block_B = self.create_patch_image(patch_block_B, output_size)
        return patch_block_A, patch_block_B

    def create_patch_image(self, patch: PatchBlock, output_size: Tuple[int, int]):
        left = patch.x
        bottom = patch.y
        right = patch.x + patch.w
        top = patch.y + patch.h
        patch_crop = self.page_image.crop((left, top, right, bottom))
        new_image = self.generate_white_background_image(output_size)

        # Calculate the size to resize the patch to fill the new image while maintaining aspect ratio
        patch_aspect_ratio = patch_crop.width / patch_crop.height
        new_image_aspect_ratio = output_size[0] / output_size[1]

        if patch_aspect_ratio > new_image_aspect_ratio:
            # Patch is wider than new image
            resized_patch = patch_crop.resize(
                (output_size[0], int(output_size[0] / patch_aspect_ratio))
            )
        else:
            # Patch is taller than new image
            resized_patch = patch_crop.resize(
                (int(output_size[1] * patch_aspect_ratio), output_size[1])
            )

        # Calculate top-left coordinates to paste the resized patch centered in the new image
        paste_position = (
            (output_size[0] - resized_patch.width) // 2,
            (output_size[1] - resized_patch.height) // 2,
        )

        # Paste the resized patch into the new image, centered
        new_image.paste(resized_patch, paste_position)
        return new_image

    @staticmethod
    def generate_white_background_image(output_size: Tuple[int, int]):
        return Image.new("RGB", output_size, (255, 255, 255))

    @staticmethod
    def images_to_pdf(image_list: List[Image.Image], path: Path):
        """
        Converts a list of PIL Image objects into a single PDF file, with each image on a separate page.

        :param image_list: List of PIL Image objects.
        :param path: Path (string) where the output PDF should be saved.
        """
        # Ensure there's at least one image in the list
        if not image_list:
            raise ValueError("The image list is empty.")

        # Convert all images to RGB (required for PDF)
        rgb_images = [img.convert("RGB") if img.mode != "RGB" else img for img in image_list]

        # Save the first image, appending the rest
        rgb_images[0].save(path, save_all=True, append_images=rgb_images[1:])


# class TesseractBlocksMethods:
#     @staticmethod
#     def get_block_type(block_patch: PatchBlock, page_block: PatchBlock):
#         # compare the length of the incomplete block with the page block
#         if block_patch.w < 0.5 * page_block.w:
#             if block_patch.x < 0.5 * page_block.w:
#                 if block_patch.x + block_patch.w < 0.5 * page_block.w:
#                     return PatchBlockType.HALF_COLUMN_LEFT
#                 else:
#                     return PatchBlockType.OTHER
#             else:
#                 return PatchBlockType.HALF_COLUMN_RIGHT
#         else:
#             return PatchBlockType.FULL


# class ExaminePrediction:
#     def __init__(self, predictions: List[str], pdf_path: Path):
#         self.pdf_path = pdf_path
#         self.predictions = predictions
#         self.checkers: List[CheckMissingWithParts] = []
#         self.incomplete_pages = []
#         logging.info(f"Initialize Checkers and Check for Incomplete Pages")
#         self._init_checkers()
#         self._check_pages()

#     def get_document_line_split(self):
#         for ii in self.incomplete_pages:
#             checker: CheckMissingWithParts = self.checkers[ii]
#             # get_page_bounding_box
#             patches = checker.patch_blocks
#             print(patches)

#     def _init_checkers(self):
#         for ii, _ in enumerate(self.predictions):
#             self.checkers.append(
#                 CheckMissingWithParts(
#                     pdf=self.pdf_path,
#                     page_number=ii,
#                     transcriptions=self.predictions,
#                 )
#             )

#     def _check_pages(self):
#         for ii, checker in enumerate(self.checkers):
#             if checker.run():
#                 self.incomplete_pages.append(ii)
