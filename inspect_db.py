import sqlite3
import argparse
import json

# This script inspects the structure of an SQLite database and retrieves a specific record based on the provided ID.

# Functions
# inspect_db(db_path)

# Description: Inspects the SQLite database and prints its structure.
# Args:
# db_path (str): The path to the SQLite database.
# Returns: None
# get_record_by_id(db_path, record_id)

# Description: Retrieves a specific record from the SQLite database based on the ID.
# Args:
# db_path (str): The path to the SQLite database.
# record_id (int): The ID of the record to retrieve.
# Returns: dict: A dictionary containing the record's information.

# Usage
# To inspect the structure of the SQLite database and retrieve a specific record by ID, run the script with the following command:

# <db_path>: The path to the SQLite database file.
# <record_id>: (Optional) The ID of the record to retrieve.
# For example, to inspect the map_ids.db database and retrieve the record with ID 1, you would run:



def inspect_db(db_path):
    """
    Inspects the SQLite database and prints its structure.

    Args:
        db_path (str): The path to the SQLite database.

    Returns:
        None
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Get the list of tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    print(f"Tables in the database: {tables}")

    # Get the structure of each table
    for table in tables:
        table_name = table[0]
        print(f"\nStructure of table '{table_name}':")
        cursor.execute(f"PRAGMA table_info({table_name});")
        columns = cursor.fetchall()
        for column in columns:
            print(column)

    conn.close()

def get_record_by_id(db_path, record_id):
    """
    Retrieves a specific record from the SQLite database based on the ID.

    Args:
        db_path (str): The path to the SQLite database.
        record_id (int): The ID of the record to retrieve.

    Returns:
        dict: A dictionary containing the record's information.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT id, url, last_updated_at, date FROM mapping WHERE id = ?", (record_id,))
    row = cursor.fetchone()
    conn.close()

    if row:
        info = {
            'id': row[0],
            'url': row[1],
            'last_updated_at': json.loads(row[2]),
            'date': json.loads(row[3])
        }
        return info
    else:
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect the structure of the SQLite database and retrieve a specific record by ID.")
    parser.add_argument('db_path', type=str, help="The path to the SQLite database file.")
    parser.add_argument('--id', type=int, help="The ID of the record to retrieve.")
    args = parser.parse_args()

    # Inspect the database
    inspect_db(args.db_path)

    # Retrieve and print the record if an ID is provided
    if args.id is not None:
        record = get_record_by_id(args.db_path, args.id)
        if record:
            print(f"\nRecord with ID {args.id}:")
            print(json.dumps(record, indent=4))
        else:
            print(f"\nNo record found with ID {args.id}")