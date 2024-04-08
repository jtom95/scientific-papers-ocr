This is a wrapper around facebook's NOUGAT model for scientific paper OCR.


## Improvements on Regular Nougat:
1. For lower end GPUs, it is necessary to set lower precision and use the smaller model 0.1.0-small. As a consequence the model is prone to skip parts of the papers. In this implementation, tesseract is used to double check if some parts of the paper are skipped. It does this by chosing representative sequences of words sampled from the text and verifying that they appear in the nougat extraction.

2. By default nougat returns .mmd files corresponding to each page. This project returns large .json files that collect, not only the raw text, but also useful metadata of the pdf paper (such as references information, authors, year etc.). The .json files are named "rich_documents" in the code.

3. Functions have been developed to write the full markdown text (.md) or write on a Notion page from the rich document. 

![TrailerOCRscientific-MadewithClipchamp-ezgif com-video-to-gif-converter](https://github.com/jtom95/scientific-papers-ocr/assets/66060633/8c5434a2-97bd-4cc4-b3bf-81ad4ef34e9d)

## Setup
* Download the nougat transformer model of your choice from [...](https://github.com/facebookresearch/nougat/releases) [small model is recommended for lower end GPUs].

* Install pytorch with access to cuda: https://pytorch.org/get-started/locally/
    - verify cuda is installed an available on your pc by opening the CMD and typing: 
        `nvidia-smi`
    - verify that pytorch is intalled with cuda available by opening a CMD and typing: 
        ```
        python

        import torch
        torch.cuda.is_available()
        ```
        should return True
* Install tesseract on your PC (for windows: https://github.com/UB-Mannheim/tesseract/wiki)
    - verify it is installed by opening a CMD and typing:
        `tesseract` 
    + After you have installed tesseract, install pytesseract with pip. 

* install the requirements listed in the pyproject.toml file. 

## Setup for Notion
In order to use the integration with Notion you must: 
- setup a new integration on [notion/integrations]https://www.notion.com/my-integrations.

- save the Notion API key: "secret...."

- go to the page of interest. Allow the integration on this page by clicking on `...` then on `+ Add Connections` and selcting your integration. 

- save the page ID (you can obtain the page id by copying the page link). 

Full tutorial at: https://developers.notion.com/docs/create-a-notion-integration


## Suggestion
In order to run the provided high level functions you need to provide the: 
1. path to the NOUGAT tranformer model
2. notion API key (and optionally notion version)

While it is possible to pass these in the functions themselves, the simpler solution is to set them as environment variables. If you're working from this directory you can achieve this simply by uncommenting and filling out the .env file. 


## High-Level Functions Description and Examples

#### 1. `generate_rich_documents.py`

**Description**: This script processes PDF files using the NOUGAT OCR model to generate rich document databases. These databases contain structured data extracted from the PDFs, including raw text and metadata such as references, authors, and publication year. The script supports batch processing of multiple PDFs, allows specifying the model size (small or base), and can start processing from scratch or continue from a previous state.

**Arguments**:
- `pdf_paths`: List of paths to PDF files to be processed.
- `--output_dir (-o)`: The directory where the output database will be stored. Defaults to the current directory.
- `--start_from_scratch`: Flag to indicate whether to start processing from scratch. Defaults to True.
- `--model_directory`: The directory where the NOUGAT model is stored. If not provided, it attempts to use the `NOUGAT_MODEL_DIR` environment variable.
- `--model_size`: The size of the NOUGAT model to use (`small` or `base`). Defaults to `small`.
- `--batch_size`: The number of PDFs to process in a batch. Defaults to 1.

**Example**:
```shell
python generate_rich_documents.py "path/to/pdf1.pdf" "path/to/pdf2.pdf" -o "path/to/output_dir" --model_directory "path/to/model_dir" --model_size small --batch_size 2
```

#### 2. `rich_documents_to_markdown.py`

**Description**: This script converts rich documents (in JSON format) into Markdown files. It reads the structured data from the rich documents and generates Markdown files that can include references and other metadata. The script supports processing multiple rich documents in a batch.

**Arguments**:
- `rich_document_paths`: List of paths to rich document JSON files to be converted.
- `--output_dir (-o)`: The directory where the Markdown files will be stored. Defaults to the current directory.

**Example**:
```shell
python rich_documents_to_markdown.py "path/to/rich_document1.json" "path/to/rich_document2.json" -o "path/to/output_dir"
```

#### 3. `rich_documents_to_notion.py`

**Description**: This script uploads rich documents (in JSON format) to a specified Notion page. It uses the Notion API to create or update pages with the content from the rich documents, including text, references, and metadata. The script supports uploading multiple rich documents to the same or different Notion pages.

**Arguments**:
- `rich_document_paths`: List of paths to rich document JSON files to be uploaded.
- `--page_id (-p)`: The ID of the Notion page where the documents will be uploaded. If not provided, the script will prompt for it.
- `--notion_api_key (-k)`: The Notion API key used for authentication. If not provided, the script attempts to use the `NOTION_API_KEY` environment variable.
- `--notion_version (-v)`: The version of the Notion API to use. Defaults to "2022-06-28".

**Example**:
```shell
python rich_documents_to_notion.py "path/to/rich_document1.json" "path/to/rich_document2.json" -p "notion_page_id" -k "your_notion_api_key" -v "2022-06-28"
```




