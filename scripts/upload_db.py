# import pandas as pd
# from pymongo import MongoClient

# # ---------------- CONFIGURATION ----------------
# MONGO_URI = 'mongodb://localhost:27017/' 
# DB_NAME = 'traffic_violations_db'
# COLLECTION_NAME = 'violations'

# # Set this to True for debugging (wipes DB before adding new data)
# # Set this to False for production (keeps history, just adds new stuff)
# RESET_DB_BEFORE_RUN = True 
# # -----------------------------------------------

# def upload_to_mongodb(csv_file_path):
#     client = MongoClient(MONGO_URI)
#     db = client[DB_NAME]
#     collection = db[COLLECTION_NAME]

#     # --- DEBUGGING LOGIC: CLEAR OLD DATA ---
#     if RESET_DB_BEFORE_RUN:
#         deleted_count = collection.delete_many({}).deleted_count
#         print(f"DEBUG MODE: Wiped {deleted_count} old records from the database.")
#     # ---------------------------------------

#     # Read CSV
#     try:
#         df = pd.read_csv(csv_file_path)
#     except FileNotFoundError:
#         print("Error: CSV file not found.")
#         return

#     data_to_insert = []
    
#     for index, row in df.iterrows():
#         record = row.to_dict()
        
#         # Path Normalization (Critical for Option 1)
#         original_path = record.get('Plate_Image_Path', '')
#         if pd.notna(original_path): # Check if path exists/isn't empty
#              record['Plate_Image_Path'] = str(original_path).replace('\\', '/')
        
#         data_to_insert.append(record)

#     if data_to_insert:
#         collection.insert_many(data_to_insert)
#         print(f"Successfully inserted {len(data_to_insert)} records.")
#     else:
#         print("CSV was empty, nothing inserted.")

# if __name__ == "__main__":
#     upload_to_mongodb(r'C:\Users\syedm\OneDrive\Documents\Traffic Violation System\Intelligent-Traffic-Violation-Detection-System\output\final_consolidated_violations.csv')
import pandas as pd
from pymongo import MongoClient
import cloudinary
import cloudinary.uploader
import os

# ---------------- CONFIGURATION ----------------
MONGO_URI = 'mongodb+srv://admin:traffic123@cluster0.hlm5gzg.mongodb.net/?appName=Cluster0'
DB_NAME = 'traffic_violations_db'
COLLECTION_NAME = 'violations'
RESET_DB_BEFORE_RUN = True 

# --- CLOUDINARY CONFIG (GET THESE FROM DASHBOARD) ---
cloudinary.config( 
  cloud_name = "dfq6ylssz", 
  api_key = "971917723645361", 
  api_secret = "-ur7gBt2r7fAHgZENbOj6w8CsWM" 
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.normpath(os.path.join(SCRIPT_DIR, ".."))
DEFAULT_CSV = os.path.join(PROJECT_ROOT, "output", "final_consolidated_violations.csv")
BASE_IMAGE_DIR = PROJECT_ROOT

def get_full_local_path(image_path):
    if pd.isna(image_path) or not image_path:
        return None
    p = str(image_path).strip()
    if os.path.isabs(p) and os.path.exists(p):
        return p
    clean_p = p.replace('/', '\\').replace('..\\', '').lstrip('\\')
    candidate = os.path.join(BASE_IMAGE_DIR, clean_p)
    if os.path.exists(candidate):
        return candidate
    if "output" in p.replace('/', '\\'):
        rel = p.replace('/', '\\').split("output", 1)[1].lstrip('\\')
        candidate = os.path.join(BASE_IMAGE_DIR, "output", rel)
        if os.path.exists(candidate):
            return candidate
    return None

def upload_image_to_cloudinary(relative_path):
    """
    Takes a local path (absolute or relative), uploads to Cloudinary, 
    and returns the public secure URL.
    """
    if pd.isna(relative_path) or relative_path == "":
        return None

    full_local_path = get_full_local_path(relative_path)
    if not full_local_path or not os.path.exists(full_local_path):
        print(f"Warning: File not found locally: {relative_path}")
        return None

    # Consistent Public ID based on project-relative path
    try:
        rel_path = os.path.relpath(full_local_path, BASE_IMAGE_DIR)
    except Exception:
        rel_path = os.path.basename(full_local_path)

    base_name = os.path.splitext(rel_path)[0]
    public_id = base_name.replace('\\', '/')
    
    try:
        response = cloudinary.uploader.upload(
            full_local_path, 
            public_id=public_id,
            overwrite=True
        )
        return response['secure_url']
    except Exception as e:
        print(f"Error uploading {public_id}: {e}")
        return None

def upload_to_mongodb(csv_file_path=None):
    if csv_file_path is None:
        csv_file_path = DEFAULT_CSV

    print(f"🔌 Connecting to MongoDB: {DB_NAME} -> {COLLECTION_NAME}...")
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]
    collection = db[COLLECTION_NAME]

    if RESET_DB_BEFORE_RUN:
        deleted_count = collection.delete_many({}).deleted_count
        print(f"DEBUG MODE: Wiped {deleted_count} old records.")

    try:
        df = pd.read_csv(csv_file_path)
    except FileNotFoundError:
        print(f"Error: CSV file not found: {csv_file_path}")
        return

    print(f"📄 Processing {len(df)} records from {csv_file_path} for Cloudinary & MongoDB...")
    data_to_insert = []
    
    for index, row in df.iterrows(): 
        record = row.to_dict()
        vid = record.get('Vehicle_ID', f'Row {index}')
        vtype = record.get('Violation_1', 'Violation')
        print(f"[{index+1}/{len(df)}] Uploading assets for {vid} ({vtype})...")
        
        # 1. Car Image
        record['Car_Image'] = upload_image_to_cloudinary(record.get('Car_Image'))
        
        # 2. Plate Image
        record['Plate_Image'] = upload_image_to_cloudinary(record.get('Plate_Image'))
        
        # 3. Violation Image (if exists)
        record['Violation_Image_1'] = upload_image_to_cloudinary(record.get('Violation_Image_1'))

        data_to_insert.append(record)

    if data_to_insert:
        collection.insert_many(data_to_insert)
        print(f"🎉 Successfully inserted {len(data_to_insert)} records with Cloudinary URLs into MongoDB!")
    else:
        print("Nothing inserted.")

if __name__ == "__main__":
    upload_to_mongodb()