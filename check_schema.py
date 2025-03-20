import sqlite3

# Connect to the database
db_path = "clothes.db"
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Get all table names
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = cursor.fetchall()

# Print schema of each table
for table in tables:
    table_name = table[0]
    print(f"\nSchema for table: {table_name}")
    cursor.execute(f"PRAGMA table_info({table_name});")
    columns = cursor.fetchall()
    for column in columns:
        print(column)

conn.close()
