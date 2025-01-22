"""
This script processes `id2query.json` files and stores the queries in an SQLite database.
It reads the old IDs and queries from the `id2query.json` files, normalizes the queries, and inserts them into the database with unique IDs.
Optionally, it can export the database content to a JSON file.

How to Run the Script:

1. Ensure the Required Files and Databases are Available:
   - `id2query.json`: JSON files containing the original query IDs and queries.
   - `queries.db`: SQLite database to store the queries.

2. Set Up the Directory Structure:
   - Place the `id2query.json` files in the appropriate subfolders within the collection folder.

3. Run the Script:
   - Use the command line to navigate to the directory containing `query_db.py`.
   - Execute the script with the following command:
     ```sh
     python query_db.py <collection_path> --db_path <queries_db_path> --export_json <yes/no>
     ```
   - Replace `<collection_path>`, `<queries_db_path>`, and `<yes/no>` with the appropriate values.

Example Command:
```sh
python query_db.py /path/to/collection --db_path /path/to/queries.db --export_json yes
```

Script Overview:
"""

import os
import json
import sqlite3
import glob
import re
import argparse
from unidecode import unidecode

def initialize_db(db_path):
    """
    Initializes the SQLite database for queries.

    Args:
        db_path (str): The path to the SQLite database.

    Returns:
        None
    """
    conn = sqlite3.connect(db_path, timeout=30)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS queries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            query TEXT UNIQUE,
            old_ids TEXT
        )
    """)
    conn.commit()
    conn.close()

def load_json_files(folder_path):
    """
    Loads all id2query.json files from subfolders.

    Args:
        folder_path (str): The path to the folder containing subfolders with id2query.json files.

    Returns:
        list: A list of tuples containing old ID, query, and folder name from all JSON files.
    """
    queries = []
    for file_path in glob.glob(os.path.join(folder_path, '**', 'id2query.json'), recursive=True):
        folder_name = os.path.basename(os.path.dirname(file_path))
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            queries.extend([(old_id, query, folder_name) for old_id, query in data.items()])
    return queries

def normalize_query(query):
    """
    Normalizes a query by removing Unicode escape sequences, punctuation, and extra spaces.

    Args:
        query (str): The query string to normalize.

    Returns:
        str: The normalized query string.
    """
    query = query.lower()  # Convert to lowercase
    query = unidecode(query)  # Remove accents and special characters
    query = query.replace("-", " ").replace("_", " ")  # Replace punctuation with spaces
    query = " ".join(query.split())  # Remove extra spaces
    return query

def process_queries(queries, db_path):
    """
    Processes queries and inserts them into the SQLite database with unique IDs.

    Args:
        queries (list): A list of tuples containing old ID, query, and folder name to process.
        db_path (str): The path to the SQLite database.

    Returns:
        None
    """
    conn = sqlite3.connect(db_path, timeout=30)
    cursor = conn.cursor()
    normalized_queries = {}

    for old_id, query, folder_name in queries:
        normalized_query = normalize_query(query)
        old_id_with_folder = f"{folder_name}:{old_id}"
        if normalized_query not in normalized_queries:
            normalized_queries[normalized_query] = []
            try:
                cursor.execute("INSERT INTO queries (query, old_ids) VALUES (?, ?)", (normalized_query, old_id_with_folder))
            except sqlite3.IntegrityError:
                pass  # Ignore duplicates
        else:
            cursor.execute("SELECT old_ids FROM queries WHERE query = ?", (normalized_query,))
            existing_old_ids = cursor.fetchone()[0]
            new_old_ids = existing_old_ids + ',' + old_id_with_folder
            cursor.execute("UPDATE queries SET old_ids = ? WHERE query = ?", (new_old_ids, normalized_query))
        normalized_queries[normalized_query].append(old_id_with_folder)

    conn.commit()
    conn.close()

def export_to_json(db_path, json_path):
    """
    Exports the queries from the SQLite database to a JSON file.

    Args:
        db_path (str): The path to the SQLite database.
        json_path (str): The path to the output JSON file.

    Returns:
        None
    """
    conn = sqlite3.connect(db_path, timeout=30)
    cursor = conn.cursor()
    cursor.execute("SELECT id, query, old_ids FROM queries")
    rows = cursor.fetchall()
    queries = [{"id": row[0], "query": row[1], "old_ids": row[2].split(',')} for row in rows]
    with open(json_path, 'w', encoding='utf-8') as json_file:
        json.dump(queries, json_file, ensure_ascii=False, indent=4)
    conn.close()

def main():
    parser = argparse.ArgumentParser(description='Process id2query.json files and store queries in an SQLite database.')
    parser.add_argument('collection_path', type=str, help='The path to the collection folder containing subfolders with id2query.json files.')
    parser.add_argument('--db_path', type=str, default='queries.db', help='The path to the SQLite database file.')
    parser.add_argument('--export_json', type=str, choices=['yes', 'no'], default='no', help='Option to export the database to a JSON file.')

    args = parser.parse_args()

    initialize_db(args.db_path)
    queries = load_json_files(args.collection_path)
    process_queries(queries, args.db_path)

    if args.export_json == 'yes':
        json_path = os.path.splitext(args.db_path)[0] + '.json'
        export_to_json(args.db_path, json_path)

if __name__ == "__main__":
    main()