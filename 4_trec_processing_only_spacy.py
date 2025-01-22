"""
This script processes collector files and converts them into TREC format using SpaCy for advanced text processing.
It reads URLs from the collector files, processes the content using SpaCy and pySBD, and saves the processed documents in TREC format.

How to Run the Script:

1. Ensure the Required Files and Databases are Available:
   - Collector files: Files containing the URLs and their metadata.
   - `map_ids.db`: SQLite database containing the URL mappings.
   - SpaCy model: The SpaCy model to use for text processing.

2. Set Up the Directory Structure:
   - Place the collector files in the appropriate subfolders within the input folder.
   - Ensure the `map_ids.db` file is in the appropriate directory.

3. Run the Script:
   - Use the command line to navigate to the directory containing `4_trec_processing_only_spacy.py`.
   - Execute the script with the following command:
     ```sh
     python 4_trec_processing_only_spacy.py <input_folder> <db_path> <output_folder> <model_path> --batch-size <batch_size>
     ```
   - Replace `<input_folder>`, `<db_path>`, `<output_folder>`, `<model_path>`, and `<batch_size>` with the appropriate values.

Example Command:
```sh
python 4_trec_processing_only_spacy.py /path/to/input /path/to/map_ids.db /path/to/output /path/to/spacy_model --batch-size 200
```

Script Overview:
"""

from concurrent.futures import ProcessPoolExecutor
import spacy
import sqlite3
import json
import gzip
import os
import argparse
from tqdm import tqdm
from itertools import islice
import logging
import pysbd

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Global variable for SpaCy model
_nlp = None

def init_spacy_model(model_path):
    """
    Initialize the SpaCy model for a process.
    This function is called once per process when using ProcessPoolExecutor.
    """
    global _nlp
    _nlp = spacy.load(model_path)

def chunked_iterable(iterable, size):
    """
    Yield successive chunks of a given size from an iterable.
    """
    it = iter(iterable)
    while chunk := list(islice(it, size)):
        yield chunk

def ensure_db_index(db_path):
    """
    Ensures the SQLite database has an index on the `url` column in the `mapping` table for faster lookups.
    Skips index creation if already indexed.
    """
    logging.info("Ensuring database indexing...")
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        # Check if the index already exists
        cursor.execute("PRAGMA index_list(mapping);")
        indices = [row[1] for row in cursor.fetchall()]
        if "idx_url" not in indices:
            cursor.execute("CREATE INDEX idx_url ON mapping (url);")
            conn.commit()
            logging.info("Database indexing completed.")
        else:
            logging.info("Index already exists. Skipping creation.")

def load_db(db_path):
    """
    Loads the SQLite database and returns a dictionary mapping URLs to IDs.
    """
    logging.info("Loading database...")
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT id, url FROM mapping")
        url_to_id = {url: f"doc{id}" for id, url in cursor.fetchall()}
    logging.info(f"Database loaded with {len(url_to_id)} entries.")
    return url_to_id

def decode_url(encoded_url):
    """
    Decodes an encoded URL.
    """
    from urllib.parse import unquote
    base_url, encoded_part = encoded_url.split('/docid/', 1)
    decoded_part = unquote(encoded_part)
    return decoded_part

def process_content_spacy(content, nlp):
    """
    Processes content using SpaCy and pySBD for advanced segmentation and semantic representation.
    Ensures readability and compatibility with translation systems.
    """
    # Initialize the pySBD segmenter
    segmenter = pysbd.Segmenter(language="fr", clean=False)
    
    # Segment the content into sentences using pySBD
    sentences = segmenter.segment(content)
    
    # Process each sentence with SpaCy
    processed_sentences = []
    for sentence in sentences:
        doc = nlp(sentence)
        processed_sentences.append(doc.text)
    
    # Combine processed sentences
    processed_content = "\n".join(processed_sentences)
    
    return processed_content

def process_batch(documents, url_to_id):
    """
    Processes a batch of documents using the global SpaCy model (_nlp) and formats them for TREC.
    """
    trec_documents = []
    for doc in documents:
        url = doc.get('url')
        if '/docid/' in url:
            url = decode_url(url)

        if url in url_to_id:
            doc_id = url_to_id[url]
            content = doc.get('content', "")
            processed_content = process_content_spacy(content, _nlp)
            trec_doc = f"<DOC>\n<DOCNO>{doc_id}</DOCNO>\n<DOCID>{doc_id}</DOCID>\n<TEXT>\n{processed_content}\n</TEXT>\n</DOC>"
            trec_documents.append(trec_doc)

    return trec_documents

def process_collector_file(file_path, output_file, url_to_id, model_path, batch_size=50):
    """
    Processes all documents in a single collector file in parallel batches.
    Documents are streamed from the file to avoid memory issues.
    """
    open_func = gzip.open if file_path.endswith('.gz') else open

    with open_func(file_path, 'rt', encoding='utf-8') as file:
        # Stream documents in chunks
        document_stream = (json.loads(line) for line in file)

        trec_documents = []
        with ProcessPoolExecutor(initializer=init_spacy_model, initargs=(model_path,)) as executor:
            futures = [
                executor.submit(process_batch, list(batch), url_to_id)
                for batch in chunked_iterable(document_stream, batch_size)
            ]
            for future in tqdm(futures, desc=f"Processing batches in {file_path}", leave=False):
                trec_documents.extend(future.result())

        if trec_documents:
            save_trec_file(output_file, trec_documents)

def save_trec_file(output_file, trec_documents):
    """
    Saves the TREC formatted documents to the output file.
    """
    logging.info(f"Saving TREC file to: {output_file}")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as file:
        file.write("\n\n".join(trec_documents) + "\n\n")
    logging.info(f"TREC file saved: {output_file}")

def process_folder(input_folder, output_folder, url_to_id, model_path, batch_size=50):
    """
    Processes all collector files in the input folder and saves the TREC files in the output folder.
    """
    subfolders = [os.path.join(input_folder, d) for d in os.listdir(input_folder) if os.path.isdir(os.path.join(input_folder, d))]
    logging.info(f"Found {len(subfolders)} subfolders to process.")

    for subfolder in tqdm(subfolders, desc="Processing subfolders"):
        files_to_process = []
        for root, _, files in os.walk(os.path.join(subfolder, 'collection')):
            files = sorted(
                [f for f in files if f.startswith('collector_') and f.split('_')[-1].split('.')[0].isdigit()],
                key=lambda x: int(x.split('_')[-1].split('.')[0])
            )
            for file_name in files:
                input_file = os.path.join(root, file_name)
                relative_path = os.path.relpath(root, input_folder)
                output_file = os.path.join(output_folder, relative_path, file_name.replace('.txt', '.trec'))
                files_to_process.append((input_file, output_file))

        for input_file, output_file in tqdm(files_to_process, desc=f"Processing files in {os.path.basename(subfolder)}", leave=False):
            process_collector_file(input_file, output_file, url_to_id, model_path, batch_size)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process collector files in a folder and save in TREC format.")
    parser.add_argument('input_folder', type=str, help="The path to the input folder containing subfolders with collector files.")
    parser.add_argument('db_path', type=str, help="The path to the SQLite database file.")
    parser.add_argument('output_folder', type=str, help="The path to the output folder where TREC files will be saved.")
    parser.add_argument('model_path', type=str, help="The path to the SpaCy model to use.")
    parser.add_argument('--batch-size', type=int, default=200, help="Number of documents to process in a batch.")
    args = parser.parse_args()

    # Ensure the database is indexed for faster lookups
    ensure_db_index(args.db_path)

    # Load URL to ID mappings
    url_to_id = load_db(args.db_path)

    # Process the folder
    process_folder(args.input_folder, args.output_folder, url_to_id, args.model_path, args.batch_size)