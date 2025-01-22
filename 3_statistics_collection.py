"""
This script reads URLs, IDs, dates, and last update times from an SQLite database and calculates various statistics about the collection.
The statistics include document count, overlap percentage, time coverage, unique and shared content, temporal dynamics, new vs. old documents, growth rate, and last update patterns.

How to Run the Script:

1. Ensure the Required Files and Databases are Available:
   - `map_ids.db`: SQLite database containing the URL mappings and their metadata.

2. Set Up the Directory Structure:
   - Place the `map_ids.db` file in the appropriate directory.

3. Run the Script:
   - Use the command line to navigate to the directory containing `3_statistics_collection.py`.
   - Execute the script with the following command:
     ```sh
     python 3_statistics_collection.py
     ```

Example Command:
```sh
python 3_statistics_collection.py
```

Script Overview:
"""

import sqlite3
import pandas as pd
from tqdm import tqdm
import json

def read_db(db_path):
    """
    Reads the SQLite database and returns a DataFrame with URLs, ids, dates, and last_updated_at.

    Args:
        db_path (str): The path to the SQLite database.

    Returns:
        pd.DataFrame: A DataFrame with URLs, ids, dates, and last_updated_at.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT id, url, last_updated_at, date FROM mapping")
    rows = cursor.fetchall()
    conn.close()
    
    data = []
    for row in rows:
        id = row[0]
        url = row[1]
        last_updated_at = json.loads(row[2])
        dates = json.loads(row[3])
        for date, last_updated in zip(dates, last_updated_at):
            data.append((id, url, date, last_updated))
    
    df = pd.DataFrame(data, columns=['id', 'url', 'date', 'last_updated_at'])
    return df

def document_count(df):
    """
    Calculates the total number of unique ids and the count of URLs per date.

    Args:
        df (pd.DataFrame): The DataFrame with URLs, ids, dates, and last_updated_at.

    Returns:
        dict: A dictionary with the total number of unique ids and the count of URLs per date.
    """
    total_unique_ids = df['id'].nunique()
    count_per_date = df.groupby('date')['id'].nunique().to_dict()
    return {
        'total_unique_ids': total_unique_ids,
        'count_per_date': count_per_date
    }

def overlap_percentage(df):
    """
    Calculates the percentage of ids in each date that also appear in other dates and the average overlap.

    Args:
        df (pd.DataFrame): The DataFrame with URLs, ids, dates, and last_updated_at.

    Returns:
        dict: A dictionary with the overlap percentage for each date and the average overlap.
    """
    dates = sorted(df['date'].unique(), key=lambda x: (int(x.split('-')[0]), int(x.split('-')[1])))
    overlap_matrix = pd.DataFrame(0, index=dates, columns=dates, dtype=float)
    
    for date1 in tqdm(dates, desc="Calculating overlap percentage"):
        ids1 = set(df[df['date'] == date1]['id'])
        for date2 in dates:
            ids2 = set(df[df['date'] == date2]['id'])
            overlap = len(ids1.intersection(ids2)) / len(ids1) * 100
            overlap_matrix.loc[date1, date2] = overlap
    
    average_overlap = overlap_matrix.mean().mean()
    return {
        'overlap_matrix': overlap_matrix,
        'average_overlap': average_overlap
    }

def time_coverage(df):
    """
    Calculates how many ids persist across multiple dates (track their lifespan).

    Args:
        df (pd.DataFrame): The DataFrame with URLs, ids, dates, and last_updated_at.

    Returns:
        pd.Series: A Series with the count of ids persisting across multiple dates.
    """
    lifespan = df.groupby('id')['date'].nunique()
    return lifespan.value_counts().sort_index()

def unique_and_shared_content(df):
    """
    Calculates the number of ids unique to each date and the number of ids shared between specific pairs or groups of dates.

    Args:
        df (pd.DataFrame): The DataFrame with URLs, ids, dates, and last_updated_at.

    Returns:
        dict: A dictionary with unique ids per date and shared ids between dates.
    """
    dates = sorted(df['date'].unique(), key=lambda x: (int(x.split('-')[0]), int(x.split('-')[1])))
    unique_ids_per_date = {}
    shared_ids_between_dates = pd.DataFrame(0, index=dates, columns=dates, dtype=int)
    
    for date1 in dates:
        ids1 = set(df[df['date'] == date1]['id'])
        unique_ids_per_date[date1] = len(ids1)
        for date2 in dates:
            ids2 = set(df[df['date'] == date2]['id'])
            shared_ids_between_dates.loc[date1, date2] = len(ids1.intersection(ids2))
    
    return {
        'unique_ids_per_date': unique_ids_per_date,
        'shared_ids_between_dates': shared_ids_between_dates
    }

def temporal_dynamics(df):
    """
    Visualizes the influx and removal of ids across months.

    Args:
        df (pd.DataFrame): The DataFrame with URLs, ids, dates, and last_updated_at.

    Returns:
        pd.DataFrame: A DataFrame with the influx and removal of ids across months.
    """
    dates = sorted(df['date'].unique(), key=lambda x: (int(x.split('-')[0]), int(x.split('-')[1])))
    influx_removal = pd.DataFrame(0, index=dates, columns=['influx', 'removal'], dtype=int)
    
    for i in range(1, len(dates)):
        ids_prev = set(df[df['date'] == dates[i-1]]['id'])
        ids_curr = set(df[df['date'] == dates[i]]['id'])
        influx_removal.loc[dates[i], 'influx'] = len(ids_curr - ids_prev)
        influx_removal.loc[dates[i], 'removal'] = len(ids_prev - ids_curr)
    
    return influx_removal

def new_vs_old_documents(df):
    """
    Calculates the number of new documents introduced each month vs. documents that persist from previous months.

    Args:
        df (pd.DataFrame): The DataFrame with URLs, ids, dates, and last_updated_at.

    Returns:
        pd.DataFrame: A DataFrame with the number of new and old documents each month.
    """
    dates = sorted(df['date'].unique(), key=lambda x: (int(x.split('-')[0]), int(x.split('-')[1])))
    new_old_docs = pd.DataFrame(0, index=dates, columns=['new', 'old'], dtype=int)
    
    all_ids = set()
    for date in dates:
        ids_curr = set(df[df['date'] == date]['id'])
        new_old_docs.loc[date, 'new'] = len(ids_curr - all_ids)
        new_old_docs.loc[date, 'old'] = len(ids_curr & all_ids)
        all_ids.update(ids_curr)
    
    return new_old_docs

def growth_rate(df):
    """
    Analyzes the growth of the collection (e.g., number of documents added each month).

    Args:
        df (pd.DataFrame): The DataFrame with URLs, ids, dates, and last_updated_at.

    Returns:
        pd.Series: A Series with the number of documents added each month.
    """
    dates = sorted(df['date'].unique(), key=lambda x: (int(x.split('-')[0]), int(x.split('-')[1])))
    growth = df.groupby('date')['id'].nunique().reindex(dates)
    growth_rate = growth.pct_change().fillna(0) * 100
    return growth_rate

def last_update_patterns(df):
    """
    Analyzes the last update patterns of the ids.

    Args:
        df (pd.DataFrame): The DataFrame with URLs, ids, dates, and last_updated_at.

    Returns:
        dict: A dictionary with the active dates for last updates and the average/median time gap between updates.
    """
    # Active dates for last updates
    last_update_counts = df.groupby('last_updated_at')['id'].count().sort_index()
    
    # Average and median time gap between updates
    df['last_updated_at'] = pd.to_datetime(df['last_updated_at'], unit='s')
    df = df.sort_values(by='last_updated_at')
    df['time_gap'] = df.groupby('id')['last_updated_at'].diff().dt.total_seconds()
    avg_time_gap = df['time_gap'].mean()
    median_time_gap = df['time_gap'].median()
    
    return {
        'last_update_counts': last_update_counts,
        'avg_time_gap': avg_time_gap,
        'median_time_gap': median_time_gap
    }

if __name__ == "__main__":
    db_path = 'map_ids.db'
    output_file = 'statistics_collection.txt'
    
    # Read the database
    print("Reading the database...")
    df = read_db(db_path)
    print("Database read complete.")
    
    with open(output_file, 'w') as f:
        # Document Count
        print("Calculating document count statistics...")
        doc_count_stats = document_count(df)
        f.write("Document Count Statistics:\n")
        f.write(f"Total unique ids: {doc_count_stats['total_unique_ids']}\n")
        f.write("Count of URLs per date:\n")
        for date, count in doc_count_stats['count_per_date'].items():
            f.write(f"{date}: {count}\n")
        print("Document count statistics complete.")
        
        # Overlap Percentage
        print("Calculating overlap percentage statistics...")
        overlap_stats = overlap_percentage(df)
        f.write("\nOverlap Percentage Statistics:\n")
        f.write("Overlap matrix:\n")
        f.write(overlap_stats['overlap_matrix'].to_string())
        f.write(f"\nAverage overlap: {overlap_stats['average_overlap']:.2f}%\n")
        print("Overlap percentage statistics complete.")
        
        # Time Coverage
        print("Calculating time coverage statistics...")
        time_coverage_stats = time_coverage(df)
        f.write("\nTime Coverage Statistics:\n")
        f.write("Number of ids persisting across multiple dates:\n")
        f.write(time_coverage_stats.to_string())
        print("Time coverage statistics complete.")
        
        # Unique and Shared Content
        print("Calculating unique and shared content statistics...")
        unique_shared_stats = unique_and_shared_content(df)
        f.write("\nUnique and Shared Content Statistics:\n")
        f.write("Unique ids per date:\n")
        for date, count in unique_shared_stats['unique_ids_per_date'].items():
            f.write(f"{date}: {count}\n")
        f.write("\nShared ids between dates:\n")
        f.write(unique_shared_stats['shared_ids_between_dates'].to_string())
        print("Unique and shared content statistics complete.")
        
        # Temporal Dynamics
        print("Calculating temporal dynamics statistics...")
        temporal_dynamics_stats = temporal_dynamics(df)
        f.write("\nTemporal Dynamics Statistics:\n")
        f.write(temporal_dynamics_stats.to_string())
        print("Temporal dynamics statistics complete.")
        
        # New vs. Old Documents
        print("Calculating new vs. old documents statistics...")
        new_old_docs_stats = new_vs_old_documents(df)
        f.write("\nNew vs. Old Documents Statistics:\n")
        f.write(new_old_docs_stats.to_string())
        print("New vs. old documents statistics complete.")
        
        # Growth Rate
        print("Calculating growth rate statistics...")
        growth_rate_stats = growth_rate(df)
        f.write("\nGrowth Rate Statistics:\n")
        f.write(growth_rate_stats.to_string())
        print("Growth rate statistics complete.")
        
        # Last Update Patterns
        print("Calculating last update patterns statistics...")
        last_update_patterns_stats = last_update_patterns(df)
        f.write("\nLast Update Patterns Statistics:\n")
        f.write("Active dates for last updates:\n")
        f.write(last_update_patterns_stats['last_update_counts'].to_string())
        f.write(f"\nAverage time gap between updates: {last_update_patterns_stats['avg_time_gap']:.2f} seconds\n")
        f.write(f"Median time gap between updates: {last_update_patterns_stats['median_time_gap']:.2f} seconds\n")
        print("Last update patterns statistics complete.")
    
    print(f"Statistics saved to {output_file}")