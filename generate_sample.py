import sqlite3, os, random

# Function for translating a number into base 36. Necessary because ids are stored as base 36 
# numbers in the database.
alphabet = "0123456789abcdefghijklmnopqrstuvwxyz"
def base36encode(number):
    base36 = ""
    while number:
        number, i = divmod(number, 36)
        base36 = alphabet[i] + base36

    return base36 or alphabet[0]

# Database metadata
db_path = "./data/reddit-comments-may-2015/database.sqlite"
db_size = 54504410 # Obtained with command 'select count(id) from may2015'
db_size_bytes = 31807655936 # Size of database in bytes
db_range = (int("cqug90g", 36), int("cqug90g", 36) + db_size - 1) # Range of IDs in database

# Sample database metadata
sample_db_name = "sample.sqlite"
sample_db_size = 673500 # Number of records in the sample database
# sample_db_size = (200000000 / db_size_bytes) * db_size # Number of records in the sample database

print("Initializing sample database...")

conn = sqlite3.connect(db_path)
c = conn.cursor()

if os.path.exists(sample_db_name): # Delete the database if it exists
  os.remove(sample_db_name)
sample_conn = sqlite3.connect(sample_db_name)
sample_c = sample_conn.cursor()

# Retrieve column names from database
columns = c.execute("select * from sqlite_master where type = \"table\"").fetchone()[4].split(", ")
columns[0] = columns[0].split("(")[1]
columns[len(columns) - 1] = columns[len(columns) - 1].split(")")[0]

# Initialize table
query = "create table may2015_sample (%s" % columns[0]
for column in columns[1 :]:
    query += ", " + column
query += ");"
sample_c.execute(query)

# Build query to insert into sample databse
insertion_query = "insert into may2015_sample values (?"
insertion_query += ", ?" * (len(columns) - 1)
insertion_query += ");"

print("Done. Adding random records into sample database...")

IDs = set([]) # Stores generated ids
while len(IDs) != sample_db_size:
    # Randomly choose a new id and corresponding record
    ID = random.randint(*db_range)
    query = "select * from may2015 where id=?"
    record = c.execute(query, [base36encode(ID)]).fetchone() # Translate id into base 36, and retrieve the record
    while ID in IDs or record == None:
        ID = random.randint(*db_range)
        query = "select * from may2015 where id=?"
        record = c.execute(query, [base36encode(ID)]).fetchone()
    
    IDs.add(ID)

    # Add to sample databse
    sample_c.execute(insertion_query, record)

    # Print progress update
    if len(IDs) % 2500 == 0:
        print("Added %i of %i - %f%% done." % (len(IDs), sample_db_size, (len(IDs) / sample_db_size) * 100))

print("Done.")

# Clean up
sample_conn.commit()
sample_conn.close()
conn.close()