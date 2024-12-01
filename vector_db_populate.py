import argparse
import os
import shutil
from langchain_community.vectorstores import Chroma
# from langchain.document_loaders import PyPDFDirectoryLoader
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
import pandas as pd
import get_embedding

DATA_PATH = "knowledge-base"
VECTOR_DB_PATH = "database"
CSV_DATA_PATH = "csv-data"


def populate_db():
    documents = load_documents()
    chunks = split_documents(documents)
    return add_to_vector_db(chunks)


def load_documents():
    # convert_all_csv_to_pdf(CSV_DATA_PATH, DATA_PATH)
    loader = PyPDFDirectoryLoader(DATA_PATH)
    documents = loader.load()
    return documents


def load_csv_documents(data_path):
    # List to store dataframes
    documents = []
    # Iterate over all files in the directory
    for filename in os.listdir(data_path):
        if filename.endswith('.csv'):
            # Construct full file path
            file_path = os.path.join(data_path, filename)
            try:
                # Try to load the CSV file with utf-8 encoding
                df = pd.read_csv(file_path, encoding='utf-8')
            except UnicodeDecodeError:
                # Fallback to latin1 encoding if utf-8 fails
                df = pd.read_csv(file_path, encoding='latin1')
            # Append dataframe to the list
            documents.append(df)

    return documents


def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        length_function=len,
        is_separator_regex=False,
    )
    documents = text_splitter.split_documents(documents)
    return documents


def add_to_vector_db(chunks: list[Document]):
    vector_db = Chroma(
        persist_directory=VECTOR_DB_PATH, embedding_function=get_embedding.get_embedding_function()
    )
    # calculate Page Ids

    chunks_with_ids = calculate_chunk_ids(chunks)
    existing_items = vector_db.get(include=[])
    existing_ids = set(existing_items["ids"])
    add_db_message = f"Number of existing documents in DB: {len(existing_ids)}"

    new_chunks = []
    for chunk in chunks_with_ids:
        if chunk.metadata["id"] not in existing_ids:
            new_chunks.append(chunk)

    if len(new_chunks):
        added_db_message = f" Adding new documents: {len(new_chunks)}"
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        vector_db.add_documents(new_chunks, ids=new_chunk_ids)
        vector_db.persist()
    else:
        added_db_message = " No new documents to add"

    response_messages = [add_db_message, added_db_message]
    return response_messages


def calculate_chunk_ids(chunks):
    # This will create IDs like "data/pdf_1.pdf:6:2"
    # Page Source : Page Number : Chunk Index

    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"

        # If the page ID is the same as the last one, increment the index.
        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        # Calculate the chunk ID.
        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id

        # Add it to the page meta-data.
        chunk.metadata["id"] = chunk_id

    return chunks


def clear_vector_db():
    if os.path.exists(VECTOR_DB_PATH):
        shutil.rmtree(VECTOR_DB_PATH)

