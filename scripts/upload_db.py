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
MONGO_URI = 'mongodb://localhost:27017/'
DB_NAME = 'traffic_violations_db'
COLLECTION_NAME = 'violations'
RESET_DB_BEFORE_RUN = True 

# --- CLOUDINARY CONFIG (GET THESE FROM DASHBOARD) ---
cloudinary.config( 
  cloud_name = "dfq6ylssz", 
  api_key = "971917723645361", 
  api_secret = "-ur7gBt2r7fAHgZENbOj6w8CsWM" 
)

# Base folder where your images are actually located on your PC
# Since your CSV uses relative paths like "../output", we need to know where to start looking.
BASE_IMAGE_DIR = r'C:\Users\syedm\OneDrive\Documents\Traffic Violation System\Intelligent-Traffic-Violation-Detection-System'
# -----------------------------------------------

def upload_image_to_cloudinary(relative_path):
    """
    Takes a local relative path, uploads to Cloudinary, 
    and returns the public URL.
    """
    if pd.isna(relative_path) or relative_path == "":
        return None

    # Fix path slashes and join with base directory
    # CSV has "../output...", so we clean it up
    clean_path = relative_path.replace('/', '\\').replace('..\\', '')
    full_local_path = os.path.join(BASE_IMAGE_DIR, clean_path)

    if not os.path.exists(full_local_path):
        print(f"Warning: File not found locally: {full_local_path}")
        return None

    try:
        # Upload to Cloudinary
        print(f"Uploading {clean_path}...")
        response = cloudinary.uploader.upload(full_local_path)
        
        # Return the secure URL (https link)
        return response['secure_url']
    except Exception as e:
        print(f"Error uploading {clean_path}: {e}")
        return None

def upload_to_mongodb(csv_file_path):
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]
    collection = db[COLLECTION_NAME]

    if RESET_DB_BEFORE_RUN:
        deleted_count = collection.delete_many({}).deleted_count
        print(f"DEBUG MODE: Wiped {deleted_count} old records.")

    try:
        df = pd.read_csv(csv_file_path)
    except FileNotFoundError:
        print("Error: CSV file not found.")
        return

    data_to_insert = []
    
    # Limit rows for testing so you don't upload 1000 images while debugging!
    # Remove [:5] when ready for full run
    for index, row in df.head(5).iterrows(): 
        record = row.to_dict()
        
        # --- UPLOAD IMAGES AND SWAP PATHS FOR URLS ---
        
        # 1. Car Image
        record['Car_Image'] = upload_image_to_cloudinary(record.get('Car_Image'))
        
        # 2. Plate Image
        record['Plate_Image'] = upload_image_to_cloudinary(record.get('Plate_Image'))
        
        # 3. Violation Image (if exists)
        record['Violation_Image_1'] = upload_image_to_cloudinary(record.get('Violation_Image_1'))

        # Add to list
        data_to_insert.append(record)

    if data_to_insert:
        collection.insert_many(data_to_insert)
        print(f"Successfully inserted {len(data_to_insert)} records with Cloudinary URLs.")
    else:
        print("Nothing inserted.")

if __name__ == "__main__":
    upload_to_mongodb(r'C:\Users\syedm\OneDrive\Documents\Traffic Violation System\Intelligent-Traffic-Violation-Detection-System\output\final_consolidated_violations.csv')