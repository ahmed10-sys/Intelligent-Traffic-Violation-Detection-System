import pandas as pd
import os
import logging

# ==========================================
# 🗂️ CONFIGURATION
# ==========================================
OUTPUT_DIR = "../output"
MASTER_LOG_PATH = os.path.join(OUTPUT_DIR, "master_violation_log.csv")
SEATBELT_LOG_PATH = os.path.join(OUTPUT_DIR, "seatbelt_results.csv")
HELMET_LOG_PATH = os.path.join(OUTPUT_DIR, "helmet_results.csv")
CLOSENESS_LOG_PATH = os.path.join(OUTPUT_DIR, "vehicle_closeness_check/violations.csv")
FINAL_OUTPUT_PATH = os.path.join(OUTPUT_DIR, "final_consolidated_violations.csv")

def main():
    print(f"🚀 Starting Result Consolidation...")
    
    # 1. Load Master Log (Base)
    if not os.path.exists(MASTER_LOG_PATH):
        print(f"❌ Master log not found: {MASTER_LOG_PATH}")
        return

    df_master = pd.read_csv(MASTER_LOG_PATH)
    print(f"✅ Loaded Master Log: {len(df_master)} records")

    # Ensure Vehicle_ID is string
    df_master['Vehicle_ID'] = df_master['Vehicle_ID'].astype(str)
    
    # Normalize ID function
    def normalize_id(val):
        val = str(val).lower().replace('id_', '')
        try:
            return str(int(val))
        except:
            return val

    df_master['join_id'] = df_master['Vehicle_ID'].apply(normalize_id)
    
    # FILTER: Only keep Cars (exclude bikes) as per user request
    # Note: master_violation_log.csv only contains vehicles where a plate was DETECTED (crop exists).
    df_master = df_master[df_master['Category'] == 'Cars']
    print(f"✅ Filtered for Cars only: {len(df_master)} records")

    # Initialize Violation List for each row
    # We will store a list of dictionaries: [{'violation': 'No Seatbelt', 'image': 'path/to/img'}, ...]
    df_master['Violations_Data'] = [[] for _ in range(len(df_master))]

    # ---------------------------------------------------------
    # 2. Merge Seatbelt Data
    # ---------------------------------------------------------
    if os.path.exists(SEATBELT_LOG_PATH):
        try:
            df_seatbelt = pd.read_csv(SEATBELT_LOG_PATH)
            df_seatbelt['vehicle_id'] = df_seatbelt['vehicle_id'].astype(str)
            df_seatbelt['join_id'] = df_seatbelt['vehicle_id'].apply(normalize_id)
            
            # DEBUG: Print unique labels found
            unique_labels = df_seatbelt['label'].unique()
            print(f"ℹ️ Seatbelt Labels Found: {unique_labels}")
            
            # Filter for violations only (Case insensitive check for 'no' and 'seatbelt')
            # Common labels: 'no_seatbelt', 'no-seatbelt', 'No Seatbelt'
            seatbelt_violations = df_seatbelt[df_seatbelt['label'].astype(str).str.lower().str.contains('no') & 
                                              df_seatbelt['label'].astype(str).str.lower().str.contains('seatbelt')]
            
            print(f"ℹ️ Found {len(seatbelt_violations)} seatbelt violations.")

            # Create a map: join_id -> list of (violation, image_path)
            seatbelt_map = {}
            for _, row in seatbelt_violations.iterrows():
                jid = row['join_id']
                # Construct absolute or relative path to the image
                # The script saves as: ../output/seatbelt_results/Cars/id_XXX/detected_filename
                # We need to reconstruct this.
                # Assuming folder name matches original ID from master log might be tricky if normalized.
                # Let's rely on the fact that seatbelt_detection.py uses the folder name from input.
                
                # We need the original folder name to construct the path. 
                # But here we only have vehicle_id (which might be '1' from 'id_001').
                # Let's try to find the folder.
                
                # Better approach: The seatbelt CSV contains 'vehicle_id' which comes from folder name.
                # If vehicle_id is '1', folder was likely 'id_001' or 'id_1'. 
                # Let's search for the image in the output directory.
                
                # Actually, seatbelt_detection.py saves:
                # writer.writerow([vehicle_id, filename, ...])
                # And saves image to: ../output/seatbelt_results/Cars/{id_folder}/detected_{filename}
                
                # We can try to find the file.
                img_name = f"detected_{row['image_name']}"
                # We don't know the exact id_folder name easily from just '1'.
                # But we can look up the original Vehicle_ID from df_master if we match on join_id.
                
                if jid not in seatbelt_map:
                    seatbelt_map[jid] = []
                seatbelt_map[jid].append({
                    'violation': 'No Seatbelt',
                    'image_name': img_name,
                    'type': 'seatbelt'
                })
            
            # Apply to master
            for index, row in df_master.iterrows():
                jid = row['join_id']
                if jid in seatbelt_map:
                    # We need to resolve the full path for the image
                    # The original Vehicle_ID (e.g. 'id_001') is in row['Vehicle_ID']
                    orig_id = row['Vehicle_ID']
                    category = row['Category'] # e.g. 'Cars'
                    
                    for item in seatbelt_map[jid]:
                        # Construct path
                        # Path: ../output/seatbelt_results/{Category}/{Vehicle_ID}/detected_{filename}
                        # Note: Seatbelt only runs on Cars.
                        if category == 'Cars':
                            img_path = os.path.join(OUTPUT_DIR, "seatbelt_results", "Cars", orig_id, item['image_name'])
                            # Verify existence? Optional.
                            df_master.at[index, 'Violations_Data'].append({
                                'violation': 'No Seatbelt',
                                'image_path': img_path
                            })
                            
            print(f"✅ Merged Seatbelt Data")
        except Exception as e:
            print(f"⚠️ Error merging seatbelt data: {e}")

    # ---------------------------------------------------------
    # 3. Merge Helmet Data
    # ---------------------------------------------------------
    if os.path.exists(HELMET_LOG_PATH):
        try:
            df_helmet = pd.read_csv(HELMET_LOG_PATH)
            df_helmet['vehicle_id'] = df_helmet['vehicle_id'].astype(str)
            df_helmet['join_id'] = df_helmet['vehicle_id'].apply(normalize_id)
            
            # Filter for violations
            helmet_violations = df_helmet[df_helmet['helmet_status'] == 'without_helmet']
            
            helmet_map = {}
            for _, row in helmet_violations.iterrows():
                jid = row['join_id']
                img_name = f"helmet_{row['image_name']}"
                
                if jid not in helmet_map:
                    helmet_map[jid] = []
                helmet_map[jid].append({
                    'violation': 'No Helmet',
                    'image_name': img_name
                })
                
            # Apply to master
            for index, row in df_master.iterrows():
                jid = row['join_id']
                if jid in helmet_map:
                    orig_id = row['Vehicle_ID']
                    category = row['Category'] # e.g. 'Bikes_Motorcycles'
                    
                    for item in helmet_map[jid]:
                        if category == 'Bikes_Motorcycles':
                            img_path = os.path.join(OUTPUT_DIR, "helmet_results", "Bikes_Motorcycles", orig_id, item['image_name'])
                            df_master.at[index, 'Violations_Data'].append({
                                'violation': 'No Helmet',
                                'image_path': img_path
                            })
                            
            print(f"✅ Merged Helmet Data")
        except Exception as e:
            print(f"⚠️ Error merging helmet data: {e}")

    # ---------------------------------------------------------
    # 4. Construct Final Columns
    # ---------------------------------------------------------
    # We need: Car_Image, Plate_Image, Violation_1, Image_1, Violation_2, Image_2, ...
    
    final_rows = []
    
    max_violations = 0
    
    for _, row in df_master.iterrows():
        # Base info
        vehicle_id = row['Vehicle_ID']
        category = row['Category']
        plate_number = row['Plate_Number']
        plate_conf = row['Plate_Confidence']
        
        # Images
        # Car Image: ../output/cropped_photos/{Category}/{Vehicle_ID}/BEST.jpg
        car_image_path = os.path.join(OUTPUT_DIR, "cropped_photos", category, vehicle_id, "BEST.jpg")
        
        # Plate Image: ../output/cropped_photos/{Category}/{Vehicle_ID}/BEST_PLATE_CROP.jpg
        plate_image_path = os.path.join(OUTPUT_DIR, "cropped_photos", category, vehicle_id, "BEST_PLATE_CROP.jpg")
        
        # Violations
        violations = row['Violations_Data']
        
        # Create row dict
        new_row = {
            'Vehicle_ID': vehicle_id,
            'Category': category,
            'Plate_Number': plate_number,
            'Plate_Confidence': plate_conf,
            'Car_Image': car_image_path,
            'Plate_Image': plate_image_path
        }
        
        # Add violations
        # Deduplicate by violation type
        seen_types = set()
        unique_violations = []
        
        for v in violations:
            v_type = v['violation']
            if v_type not in seen_types:
                seen_types.add(v_type)
                unique_violations.append(v)
        
        for i, v in enumerate(unique_violations):
            idx = i + 1
            new_row[f'Violation_{idx}'] = v['violation']
            new_row[f'Violation_Image_{idx}'] = v['image_path']
        
        if len(unique_violations) > max_violations:
            max_violations = len(unique_violations)
            
        # FILTER: Only add to final list if there is at least one violation
        if len(unique_violations) > 0:
            final_rows.append(new_row)
        
    # Create DataFrame
    df_final = pd.DataFrame(final_rows)
    
    # Ensure all columns exist up to max_violations (or at least 1 if none)
    max_violations = max(max_violations, 1)
    
    # Define column order
    cols = ['Vehicle_ID', 'Category', 'Plate_Number', 'Plate_Confidence', 'Car_Image', 'Plate_Image']
    for i in range(1, max_violations + 1):
        cols.append(f'Violation_{i}')
        cols.append(f'Violation_Image_{i}')
        
    # Add missing columns with empty strings
    for col in cols:
        if col not in df_final.columns:
            df_final[col] = ""
            
    # Reorder
    df_final = df_final[cols]
    
    # Save
    df_final.to_csv(FINAL_OUTPUT_PATH, index=False)
    print("\n" + "="*50)
    print(f"🎉 CONSOLIDATED CSV CREATED: {FINAL_OUTPUT_PATH}")
    print("="*50)
    print(df_final.head())

if __name__ == "__main__":
    main()
