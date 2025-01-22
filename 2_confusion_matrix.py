"""
This script reads URLs and their dates from an SQLite database, creates a confusion matrix to check the overlap of URLs between different dates, and plots the confusion matrix.

How to Run the Script:

1. Ensure the Required Files and Databases are Available:
   - `map_ids.db`: SQLite database containing the URL mappings and their dates.

2. Set Up the Directory Structure:
   - Place the `map_ids.db` file in the appropriate directory.

3. Run the Script:
   - Use the command line to navigate to the directory containing `2_confusion_matrix.py`.
   - Execute the script with the following command:
     ```sh
     python 2_confusion_matrix.py
     ```

Example Command:
```sh
python 2_confusion_matrix.py
```

Script Overview:
"""

import sqlite3
import pandas as pd
import numpy as np
import json
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

def read_db(db_path, chunk_size=10000):
    """
    Reads the SQLite database in chunks and returns a DataFrame with URLs and their dates.

    Args:
        db_path (str): The path to the SQLite database.
        chunk_size (int): The number of rows to read at a time.

    Returns:
        pd.DataFrame: A DataFrame with URLs and their dates.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM mapping")
    total_rows = cursor.fetchone()[0]
    
    data = []
    for offset in tqdm(range(0, total_rows, chunk_size), desc="Reading database"):
        cursor.execute("SELECT url, date FROM mapping LIMIT ? OFFSET ?", (chunk_size, offset))
        rows = cursor.fetchall()
        for row in rows:
            url = row[0]
            dates = json.loads(row[1])
            for date in dates:
                data.append((url, date))
    
    conn.close()
    df = pd.DataFrame(data, columns=['url', 'date'])
    return df

def create_confusion_matrix(df):
    """
    Creates a confusion matrix to check the overlap of URLs between different dates.

    Args:
        df (pd.DataFrame): A DataFrame with URLs and their dates.

    Returns:
        pd.DataFrame: A confusion matrix with normalized values.
    """
    dates = sorted(df['date'].unique(), key=lambda x: (int(x.split('-')[0]), int(x.split('-')[1])))
    matrix = pd.DataFrame(0, index=dates, columns=dates, dtype=float)
    
    for date1 in tqdm(dates, desc="Creating confusion matrix"):
        urls1 = set(df[df['date'] == date1]['url'])
        for date2 in dates:
            urls2 = set(df[df['date'] == date2]['url'])
            overlap = len(urls1.intersection(urls2))
            matrix.loc[date1, date2] = overlap
    
    # Normalize the matrix
    matrix = matrix / matrix.max().max()
    
    # Set diagonal values to 1
    np.fill_diagonal(matrix.values, 1)
    
    return matrix

def plot_confusion_matrix(matrix, output_file):
    """
    Plots the confusion matrix with color intensity and saves the figure.

    Args:
        matrix (pd.DataFrame): A confusion matrix with normalized values.
        output_file (str): The path to save the figure.
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(matrix, annot=True, cmap='Reds', fmt='.2f')
    plt.title('URL Overlap Confusion Matrix')
    plt.xlabel('Date')
    plt.ylabel('Date')
    plt.savefig(output_file)
    plt.show()

if __name__ == "__main__":
    db_path = 'map_ids.db'
    output_file = 'confusion_matrix.png'
    
    # Read the database
    df = read_db(db_path)
    
    # Create the confusion matrix
    matrix = create_confusion_matrix(df)
    
    # Plot and save the confusion matrix
    plot_confusion_matrix(matrix, output_file)