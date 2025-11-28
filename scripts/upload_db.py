import pandas as pd
from pymongo import MongoClient

# ---------------- CONFIGURATION ----------------
MONGO_URI = 'mongodb://localhost:27017/' 
DB_NAME = 'traffic_violations_db'
COLLECTION_NAME = 'violations'

# Set this to True for debugging (wipes DB before adding new data)
# Set this to False for production (keeps history, just adds new stuff)
RESET_DB_BEFORE_RUN = True 
# -----------------------------------------------

def upload_to_mongodb(csv_file_path):
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]
    collection = db[COLLECTION_NAME]

    # --- DEBUGGING LOGIC: CLEAR OLD DATA ---
    if RESET_DB_BEFORE_RUN:
        deleted_count = collection.delete_many({}).deleted_count
        print(f"DEBUG MODE: Wiped {deleted_count} old records from the database.")
    # ---------------------------------------

    # Read CSV
    try:
        df = pd.read_csv(csv_file_path)
    except FileNotFoundError:
        print("Error: CSV file not found.")
        return

    data_to_insert = []
    
    for index, row in df.iterrows():
        record = row.to_dict()
        
        # Path Normalization (Critical for Option 1)
        original_path = record.get('Plate_Image_Path', '')
        if pd.notna(original_path): # Check if path exists/isn't empty
             record['Plate_Image_Path'] = str(original_path).replace('\\', '/')
        
        data_to_insert.append(record)

    if data_to_insert:
        collection.insert_many(data_to_insert)
        print(f"Successfully inserted {len(data_to_insert)} records.")
    else:
        print("CSV was empty, nothing inserted.")

if __name__ == "__main__":
    upload_to_mongodb(r'C:\Users\syedm\OneDrive\Documents\Traffic Violation System\Intelligent-Traffic-Violation-Detection-System\output\final_consolidated_violations.csv')