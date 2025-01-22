# Longeval-Processing-code
This repository contains scripts for processing and managing query and document IDs in the LongEval Map Index Collection. The scripts utilize SQLite databases to map old IDs to new IDs and provide functionalities for inspecting and exporting database contents.

## Scripts Overview

### `process_qrels.py`

This script processes `qrels.txt` files by substituting the query IDs and document IDs with new IDs from the provided SQLite databases. It reads the old IDs from the `qrels.txt` file, maps them to new IDs using the databases, and writes the processed data to a new file.

### `query_db.py`

This script processes `id2query.json` files and stores the queries in an SQLite database. It reads the old IDs and queries from the `id2query.json` files, normalizes the queries, and inserts them into the database with unique IDs. Optionally, it can export the database content to a JSON file.

### `inspect_db.py`

This script inspects the structure of an SQLite database and retrieves a specific record based on the provided ID. It prints the structure of the database and, if an ID is provided, retrieves and displays the corresponding record.

### `1_main_mapping.py`

This script processes collector files and updates the mapping in an SQLite database. It reads URLs from the collector files, updates the database with the URLs and their metadata, and optionally exports the database content to a JSON file.

### `2_confusion_matrix.py`

This script reads URLs and their dates from an SQLite database, creates a confusion matrix to check the overlap of URLs between different dates, and plots the confusion matrix.

### `3_statistics_collection.py`

This script reads URLs, IDs, dates, and last update times from an SQLite database and calculates various statistics about the collection. The statistics include document count, overlap percentage, time coverage, unique and shared content, temporal dynamics, new vs. old documents, growth rate, and last update patterns.

### `4_trec_processing_only_spacy.py`

This script processes collector files and converts them into TREC format using SpaCy for advanced text processing. It reads URLs from the collector files, processes the content using SpaCy and pySBD, and saves the processed documents in TREC format.

### `5_translate_multi_gpu_2.py`

This script translates TREC files from French to English using MarianMT on multiple GPUs. It reads the content from TREC files, processes the content using SpaCy and pySBD for segmentation, and translates the content using MarianMT. The translated content is then saved in TREC format.

## Detailed Instructions

For detailed instructions on how to run each script, please refer to the comments at the top of each script file.
