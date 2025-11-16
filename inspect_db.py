#!/usr/bin/env python3
"""
Simple script to inspect the browser history SQLite database schema
and see all available fields you can query.
"""

import shutil
import sqlite3
from pathlib import Path

HISTORY_PATH = Path.home() / ".config/BraveSoftware/Brave-Browser/Default/History"


def inspect_database():
    """Inspect the database schema and show sample data"""

    # Copy database to avoid locking
    temp_db = Path("/tmp/browser_history_inspect.db")
    shutil.copy2(HISTORY_PATH, temp_db)

    try:
        conn = sqlite3.connect(temp_db)
        cursor = conn.cursor()

        print("\n" + "="*80)
        print("BROWSER HISTORY DATABASE SCHEMA")
        print("="*80)

        # Get all tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;")
        tables = cursor.fetchall()

        print(f"\nTables in database: {len(tables)}\n")

        for table in tables:
            table_name = table[0]
            print(f"\n{'─'*80}")
            print(f"TABLE: {table_name}")
            print('─'*80)

            # Get column information
            cursor.execute(f"PRAGMA table_info({table_name});")
            columns = cursor.fetchall()

            print(f"\n{'#':<4} {'Column Name':<25} {'Type':<15} {'Not Null':<10} {'Primary Key'}")
            print('─'*80)
            for col in columns:
                _, name, col_type, not_null, default_val, pk = col
                not_null_str = "YES" if not_null else "NO"
                pk_str = "YES" if pk else ""
                print(f"{_:<4} {name:<25} {col_type:<15} {not_null_str:<10} {pk_str}")

            # Get row count
            cursor.execute(f"SELECT COUNT(*) FROM {table_name};")
            count = cursor.fetchone()[0]
            print(f"\nTotal rows: {count:,}")

            # Show sample data for important tables
            important_tables = [
                # 'content_annotations',
                # 'context_annotations',
                # 'downloads',
                # 'downloads_url_chains',
                'keyword_search_terms',
                # 'segment_usage',
                'segments',
                'urls',
                'visited_links',
                'visits'
            ]
            if table_name in important_tables and count > 0:
                print("\nSample data (first 2 rows):")
                cursor.execute(f"SELECT * FROM {table_name} LIMIT 3;")
                samples = cursor.fetchall()

                col_names = [col[1] for col in columns]
                for i, row in enumerate(samples, 1):
                    print(f"\n  Row {i}:")
                    for col_name, value in zip(col_names, row):
                        # Truncate long values
                        if isinstance(value, str) and len(value) > 60:
                            value = value[:60] + "..."
                        print(f"    {col_name}: {value}")

        # Show useful JOIN queries
        print("\n" + "="*80)
        print("USEFUL QUERY EXAMPLES")
        print("="*80)

        print("""
1. Basic history with all URL fields:
   SELECT 
       urls.id, 
       urls.url, 
       urls.title, 
       urls.visit_count, 
       urls.typed_count,
       urls.last_visit_time
   FROM urls
   ORDER BY urls.last_visit_time DESC
   LIMIT 100;

2. Full history with visit details:
   SELECT 
       urls.url,
       urls.title,
       urls.visit_count,
       visits.visit_time,
       visits.visit_duration,
       visits.transition
   FROM urls
   JOIN visits ON urls.id = visits.url  -- INNER JOIN: only URLs with visits
   ORDER BY visits.visit_time DESC
   LIMIT 100;
   
   Note: Use LEFT JOIN instead of JOIN to include all URLs, even without visits

3. Most visited URLs:
   SELECT url, title, visit_count
   FROM urls
   ORDER BY visit_count DESC
   LIMIT 20;

4. URLs typed directly (not clicked links):
   SELECT url, title, typed_count
   FROM urls
   WHERE typed_count > 0
   ORDER BY typed_count DESC;
        """)

        conn.close()

    finally:
        temp_db.unlink()

    print("\n" + "="*80)


if __name__ == "__main__":
    if not HISTORY_PATH.exists():
        print(f"Error: History file not found at {HISTORY_PATH}")
    else:
        inspect_database()
