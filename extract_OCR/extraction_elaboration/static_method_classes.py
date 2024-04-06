from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import string
import re
import nltk
from nltk.corpus import stopwords
from spellchecker import SpellChecker
import fitz

class AnalyzeTranscriptMethods:
    # Function to split the larger text into sentences or chunks

    @staticmethod
    def is_long_enough(text: str, min_length=150):
        return len(text) >= min_length

    @staticmethod
    def is_meaningful(text: str, min_alpha_ratio=0.7, min_avg_word_length=3):
        alpha_chars = [char for char in text if char.isalpha()]
        alpha_ratio = len(alpha_chars) / len(text) if text else 0

        words = text.split()
        avg_word_length = sum(len(word) for word in words) / len(words) if words else 0

        return alpha_ratio >= min_alpha_ratio and avg_word_length >= min_avg_word_length

    @staticmethod
    def return_block_is_present_score(
        block: str,
        transcription: str,
        max_non_alpha=0,
        sentence_length=6,
        num_meaningful_sentences=3,
        min_word_length=2,
    ):
        meaningful_sentences = TextUtils.extract_meaningful_sentences(
            block,
            sentence_length=sentence_length,
            max_non_alpha=max_non_alpha,
            num_meaningful_sentences=num_meaningful_sentences,
            min_length_word=min_word_length,
        )
        if len(meaningful_sentences) < num_meaningful_sentences:
            return -1
        # check if the block is present in any of the sentences
        score = 0
        for sentence in meaningful_sentences:
            if sentence in transcription:
                score += 1
        return score

    @staticmethod
    # Function to find the most similar segment to block_text in transcript
    def find_most_similar_segment(transcript: str, block_text: str, model: SentenceTransformer):
        segments = transcript.split("\n\n")
        block_text_embedding = model.encode([block_text])  # Encode block_text to get its embedding

        # Track the highest similarity and the most similar segment
        highest_similarity = 0
        most_similar_segment = None

        # Compare each segment in transcript to block_text
        for segment in segments:
            segment_embedding = model.encode([segment])  # Encode the segment
            similarity = cosine_similarity(block_text_embedding, segment_embedding)[0][
                0
            ]  # Compute cosine similarity

            if similarity > highest_similarity:
                highest_similarity = similarity
                most_similar_segment = segment

        return most_similar_segment, highest_similarity

    @staticmethod
    def preprocess_text(text: str):
        # stop_words = set(stopwords.words("english"))
        text = text.lower()
        text = "".join([char for char in text if char not in string.punctuation])
        # words = nltk.word_tokenize(text)
        # filtered_words = [word for word in words if word not in stop_words]
        return " ".join(text)

    @staticmethod
    def extract_text_by_blocks_OCR(data):
        """
        Extracts and groups text by block-level bounding boxes from pytesseract OCR data.

        :param data: The dictionary obtained from pytesseract.image_to_data.
        :return: A list of strings, where each string contains the text from a block-level bounding box.
        """
        # Initialize a dictionary to hold text for each block
        block_texts = {}

        # Iterate through each item in the OCR data
        for i in range(len(data["text"])):
            # Only consider items with non-empty text
            if data["text"][i].strip():
                block_num = data["block_num"][i]  # Identify the block number for this text element

                # If the block hasn't been seen before, initialize it in the dictionary
                if block_num not in block_texts:
                    block_texts[block_num] = data["text"][i]
                else:
                    # If the block already exists, append the text to it, adding a space for readability
                    block_texts[block_num] += " " + data["text"][i]

        return block_texts

    @staticmethod
    def extract_text_by_blocks_fitz(pdf_path, page_number=0):
        """
        Extracts and groups text by block-level elements from a PDF using PyMuPDF (fitz).

        :param pdf_path: Path to the PDF file.
        :return: A dictionary where each key is a block number and each value is the text contained within that block.
        """
        doc = fitz.open(pdf_path)
        block_texts = {}

        for page_num, page in enumerate(doc):
            if page_number and page_number != page_num:
                continue
            # Extract text as a dictionary, which includes blocks
            blocks = page.get_text("dict")["blocks"]
            for block in blocks:
                if block["type"] == 0:  # Ensure we're looking at a text block
                    block_num = f"Page{page_num + 1}-Block{block['number']}"
                    block_text = " ".join([line["spans"][0]["text"] for line in block["lines"]])
                    block_texts[block_num] = block_text

        doc.close()
        return block_texts

    @staticmethod
    def remove_misspelled_words(text: str):
        spell = SpellChecker()
        words = text.split()
        # Identify unknown words, which are likely misspelled
        misspelled = spell.unknown(words)
        # Remove misspelled words from the text
        filtered_text = [word for word in words if word not in misspelled]
        return " ".join(filtered_text)

    @staticmethod
    def remove_non_alphanumeric(text: str):
        return re.sub(r"[^a-zA-Z0-9\s]", "", text)


class TextUtils:
    unacceptable_word_patterns = [
        # any word like mth, nth etc
        r"\S*[mn]th\S*",
    ]

    @staticmethod
    def extract_meaningful_sentences(
        noisy_text: str,
        sentence_length: int = 6,
        max_non_alpha: int = 1,
        num_meaningful_sentences: int = 5,
        min_length_word: int = 3,
    ):
        meaningful_sentences = []

        # Split the text into words
        words = noisy_text.split()

        # Iterate over the words to form sentences of sentence_length
        pos = 0
        for _ in range(len(words) - sentence_length + 1):
            num_non_alph_characters = 0
            # Form a candidate sentence
            candidate_sentence = " ".join(words[pos : pos + sentence_length])
            if not candidate_sentence:
                break
            acceptable_candidate = True
            for word_index, word in enumerate(candidate_sentence.split()):
                if len(word) < min_length_word:
                    pos += word_index + 1
                    acceptable_candidate = False
                    break
                # check if the word contains a capital letter
                if any(char.isupper() for char in word):
                    pos += word_index + 1
                    acceptable_candidate = False
                    break
                # check if the word contains a non-alphanumeric character
                if not word.isalpha():
                    number_of_non_alpha_in_word = sum(not char.isalpha() for char in word)
                    num_non_alph_characters += number_of_non_alpha_in_word
                    if num_non_alph_characters > max_non_alpha:
                        pos += word_index + 1
                        acceptable_candidate = False
                        break

                # check if the word contains any unacceptable patterns
                if any(
                    re.search(pattern, word) for pattern in TextUtils.unacceptable_word_patterns
                ):
                    pos += word_index + 1
                    acceptable_candidate = False
                    break
                if word[-1] in "#$%&'()*+-/<=>?@[\\]^_`{|}~":
                    pos += word_index + 1
                    acceptable_candidate = False
                    break

            if not acceptable_candidate:
                continue

            pos += len(candidate_sentence.split())
            meaningful_sentences.append(candidate_sentence)

            # If disqualified, move to the next word (increment i by 1 due to the for loop)
            if len(meaningful_sentences) >= num_meaningful_sentences:
                break

        return meaningful_sentences
