# Longeval-Processing-code
This repository contains scripts for processing and managing query and document IDs in the LongEval Map Index Collection. The scripts utilize SQLite databases to map old IDs to new IDs and provide functionalities for inspecting and exporting database contents.

## Scripts Overview

### `process_qrels.py`

This script processes `qrels.txt` files by substituting the query IDs and document IDs with new IDs from the provided SQLite databases. It reads the old IDs from the `qrels.txt` file, maps them to new IDs using the databases, and writes the processed data to a new file.

### `query_db.py`

This script processes `id2query.json` files and stores the queries in an SQLite database. It reads the old IDs and queries from the `id2query.json` files, normalizes the queries, and inserts them into the database with unique IDs. Optionally, it can export the database content to a JSON file.

### `inspect_db.py`

This script inspects the structure of an SQLite database and retrieves a specific record based on the provided ID. It prints the structure of the database and, if an ID is provided, retrieves and displays the corresponding record.

## Detailed Instructions

For detailed instructions on how to run each script, please refer to the comments at the top of each script file.
