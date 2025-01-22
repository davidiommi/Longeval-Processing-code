"""
This script processes collector files and updates the mapping in an SQLite database.
It reads URLs from the collector files, updates the database with the URLs and their metadata, and optionally exports the database content to a JSON file.

How to Run the Script:

1. Ensure the Required Files and Databases are Available:
   - Collector files: Files containing the URLs and their metadata.
   - `urls_no_adult.json`: JSON files containing the list of URLs to include in the mapping.
   - `map_ids.db`: SQLite database to store the URL mappings.

2. Set Up the Directory Structure:
   - Place the collector files and `urls_no_adult.json` files in the appropriate subfolders within the collection folder.

3. Run the Script:
   - Use the command line to navigate to the directory containing `1_main_mapping.py`.
   - Execute the script with the following command:
     ```sh
     python 1_main_mapping.py --export-json
     ```
   - The `--export-json` flag is optional and will export the database content to a JSON file if provided.

Example Command:
```sh
python 1_main_mapping.py --export-json
```

Script Overview:
"""

import os
import json
import re
import sqlite3
import argparse
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Lock
from utils.file_reader import read_collector_file, read_url_list

def initialize_db(db_path):
    """
    Initializes the SQLite database.

    Args:
        db_path (str): The path to the SQLite database.

    Returns:
        None
    """
    conn = sqlite3.connect(db_path, timeout=30)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS mapping (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            url TEXT UNIQUE,
            last_updated_at TEXT,
            date TEXT
        )
    """)
    conn.commit()
    conn.close()

def process_file(file_path, url_list, db_path, lock):
    """
    Processes a single collector file and updates the SQLite database.

    Args:
        file_path (str): The path to the collector file.
        url_list (set): A set of URLs to include in the mapping.
        db_path (str): The path to the SQLite database.
        lock (multiprocessing.Lock): A lock to ensure only one process writes to the database at a time.

    Returns:
        None
    """
    with lock:
        conn = sqlite3.connect(db_path, timeout=30, check_same_thread=False)  # Thread-safe connection
        try:
            read_collector_file(file_path, url_list, conn)  # Pass the connection
        finally:
            conn.close()

def export_to_json(db_path, output_file):
    """
    Exports the SQLite database to a JSON file.

    Args:
        db_path (str): The path to the SQLite database.
        output_file (str): The path to the output JSON file.

    Returns:
        None
    """
    conn = sqlite3.connect(db_path, timeout=30)
    cursor = conn.cursor()
    cursor.execute("SELECT id, url, last_updated_at, date FROM mapping")
    rows = cursor.fetchall()
    mapping = {}
    for row in rows:
        mapping[row[1]] = {
            'id': row[0],
            'url': row[1],
            'last_updated_at': json.loads(row[2]),
            'date': json.loads(row[3])
        }
    with open(output_file, 'w', encoding='utf-8') as file:
        json.dump(mapping, file, indent=4)
    conn.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process collector files and update mapping.")
    parser.add_argument('--export-json', action='store_true', help="Export the final mapping to a JSON file.")
    args = parser.parse_args()

    base_directory = './collection_2022_2023'
    db_path = 'map_ids.db'
    output_file = 'map_ids.json'
    
    # Initialize the SQLite database
    initialize_db(db_path)
    
    # Create a lock for database access
    lock = Lock()
    
    # List all collection folders
    collection_folders = [d for d in os.listdir(base_directory) if os.path.isdir(os.path.join(base_directory, d)) and re.match(r'20\d{2}-\d{2}_fr', d)]
    
    # Process each collection folder
    for folder in collection_folders:
        directory_path = os.path.join(base_directory, folder, 'collection')
        url_list_file = os.path.join(base_directory, folder, 'urls_no_adult.json')
        
        # Check if the URL list file exists
        if os.path.exists(url_list_file):
            # Read the list of URLs from the JSON file
            url_list = read_url_list(url_list_file)
        else:
            # If the URL list file does not exist, use an empty set
            url_list = set()
        
        # List all files in the collection directory
        files = [f for f in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, f)) and f.startswith('collector_')]
        files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))

        # Process all files in the sorted list with a progress bar for each folder
        with ThreadPoolExecutor() as executor:
            futures = []
            for file_name in files:
                file_path = os.path.join(directory_path, file_name)
                futures.append(executor.submit(process_file, file_path, url_list, db_path, lock))

            for future in tqdm(futures, desc=f"Processing files in {folder}"):
                future.result()

    # Export the SQLite database to a JSON file if the option is selected
    if args.export_json:
        export_to_json(db_path, output_file)
        print(f"URL mapping saved to {output_file}")
    else:
        print(f"URL mapping saved to {db_path}")