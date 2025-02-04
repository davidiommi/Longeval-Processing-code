"""
This script processes `qrels.txt` files by substituting the query IDs and document IDs with new IDs from the provided SQLite databases.
It reads the old IDs from the `qrels.txt` file, maps them to new IDs using the databases, and writes the processed data to a new file.

How to Run the Script:

1. Ensure the Required Files and Databases are Available:
   - `qrels.txt`: The file containing the original query and document IDs.
   - `id2url.json`: A JSON file mapping old document IDs to URLs.
   - `queries.db`: SQLite database containing the mapping of old query IDs to new query IDs.
   - `map_ids.db`: SQLite database containing the mapping of URLs to new document IDs.

2. Set Up the Directory Structure:
   - Place the `qrels.txt` and `id2url.json` files in the appropriate subfolders within the collection folder.

3. Run the Script:
   - Use the command line to navigate to the directory containing `process_qrels.py`.
   - Execute the script with the following command:
     ```sh
     python process_qrels.py <collection_path> --db_path <queries_db_path> --map_ids_db_path <map_ids_db_path> --output_folder <output_folder>
     ```
   - Replace `<collection_path>`, `<queries_db_path>`, `<map_ids_db_path>`, and `<output_folder>` with the appropriate paths.

Example Command:
```sh
python process_qrels.py /path/to/collection --db_path /path/to/queries.db --map_ids_db_path /path/to/map_ids.db --output_folder /path/to/output
```

Script Overview:
"""

import os
import json
import gzip
import sqlite3
import glob
import argparse
from tqdm import tqdm

def read_qrels(file_path):
    """
    Reads the qrels.txt file and parses its content.

    Args:
        file_path (str): The path to the qrels.txt file.

    Returns:
        list: A list of tuples containing query ID, iteration, document ID, and relevance.
    """
    qrels = []
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) != 4:
                continue
            query_id, iteration, doc_id, relevance = parts
            # Append the IDs as strings instead of converting to integers
            qrels.append((query_id, iteration, doc_id, int(relevance)))
    return qrels

def read_id2url(file_path):
    """
    Reads the id2url.json or id2url.json.gz file and parses its content.

    Args:
        file_path (str): The path to the id2url.json or id2url.json.gz file.

    Returns:
        dict: A dictionary mapping docids to URLs.
    """
    if file_path.endswith('.gz'):
        with gzip.open(file_path, 'rt', encoding='utf-8') as file:
            return json.load(file)
    else:
        with open(file_path, 'r', encoding='utf-8') as file:
            return json.load(file)

def get_new_query_id(old_id, folder_name, db_path):
    """
    Retrieves the new query ID from the SQLite database based on the old ID and folder name.

    Args:
        old_id (str): The old query ID.
        folder_name (str): The name of the folder (month).
        db_path (str): The path to the SQLite database.

    Returns:
        int: The new query ID.
    """
    conn = sqlite3.connect(db_path, timeout=30)
    cursor = conn.cursor()
    old_id_with_folder = f"{folder_name}:{old_id}"
    cursor.execute("SELECT id FROM queries WHERE old_ids LIKE ?", (f"%{old_id_with_folder}%",))
    result = cursor.fetchone()
    conn.close()
    return result[0] if result else None

def get_new_doc_id(old_doc_id, id2url, db_path):
    """
    Retrieves the new doc ID from the SQLite database based on the old doc ID and URL.

    Args:
        old_doc_id (int): The old document ID.
        id2url (dict): A dictionary mapping old doc IDs to URLs.
        db_path (str): The path to the SQLite database.

    Returns:
        int: The new doc ID.
    """
    url = id2url.get(str(old_doc_id))
    if not url:
        return None

    conn = sqlite3.connect(db_path, timeout=30)
    cursor = conn.cursor()
    cursor.execute("SELECT id FROM mapping WHERE url = ?", (url,))
    result = cursor.fetchone()
    conn.close()
    return result[0] if result else None

def process_qrels_file(qrels_file_path, db_path, map_ids_db_path, output_folder):
    """
    Processes a single qrels.txt file, substituting the query IDs and doc IDs with new IDs from the databases.

    Args:
        qrels_file_path (str): The path to the qrels.txt file.
        db_path (str): The path to the SQLite database for queries.
        map_ids_db_path (str): The path to the SQLite database for doc IDs.
        output_folder (str): The folder to save the processed qrels.txt file.

    Returns:
        None
    """
    folder_name = os.path.basename(os.path.dirname(qrels_file_path))
    qrels = read_qrels(qrels_file_path)

    id2url_path = os.path.join(os.path.dirname(qrels_file_path), 'id2url.json')
    if not os.path.exists(id2url_path):
        id2url_path = os.path.join(os.path.dirname(qrels_file_path), 'id2url.json.gz')

    id2url = read_id2url(id2url_path)

    processed_qrels = []

    for query_id, iteration, doc_id, relevance in tqdm(qrels, desc=f"Processing {os.path.basename(qrels_file_path)}"):
        new_query_id = get_new_query_id(query_id, folder_name, db_path)
        new_doc_id = get_new_doc_id(doc_id, id2url, map_ids_db_path)
        if new_query_id is not None and new_doc_id is not None:
            processed_qrels.append((new_query_id, iteration, new_doc_id, relevance))

    output_file_path = os.path.join(output_folder, folder_name, 'qrels_processed.txt')
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
    with open(output_file_path, 'w', encoding='utf-8') as file:
        for new_query_id, iteration, new_doc_id, relevance in processed_qrels:
            file.write(f"{new_query_id} {iteration} {new_doc_id} {relevance}\n")

def main():
    parser = argparse.ArgumentParser(description='Process qrels.txt files and substitute query IDs and doc IDs with new IDs from the databases.')
    parser.add_argument('collection_path', type=str, help='The path to the collection folder containing subfolders with qrels.txt files.')
    parser.add_argument('--db_path', type=str, default='queries.db', help='The path to the SQLite database file for queries.')
    parser.add_argument('--map_ids_db_path', type=str, default='map_ids.db', help='The path to the SQLite database file for doc IDs.')
    parser.add_argument('--output_folder', type=str, default='processed_qrels', help='The folder to save the processed qrels.txt files.')

    args = parser.parse_args()

    qrels_files = glob.glob(os.path.join(args.collection_path, '**', 'qrels.txt'), recursive=True)
    for qrels_file_path in tqdm(qrels_files, desc="Processing qrels files"):
        process_qrels_file(qrels_file_path, args.db_path, args.map_ids_db_path, args.output_folder)

if __name__ == "__main__":
    main()