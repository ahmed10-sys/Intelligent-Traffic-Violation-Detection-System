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

# 💰 FINE CONFIGURATION
FINE_AMOUNTS = {
    'No Seatbelt': 5000,
    'No Helmet': 2000,
    'Default': 1000
}

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
    
    # ---------------------------------------------------------
    # 🛑 FILTER: CARS ONLY
    # ---------------------------------------------------------
    # We explicitly remove Bikes/Motorcycles because plate detection isn't ready for them yet.
    df_master = df_master[df_master['Category'] == 'Cars']
    print(f"✅ Filtered for CARS ONLY: {len(df_master)} records remaining")
    
    if len(df_master) == 0:
        print("⚠️ No cars found in master log. Exiting consolidation.")
        # Create empty file so next scripts don't crash
        pd.DataFrame(columns=['Vehicle_ID', 'Category', 'Plate_Number', 'Total_Fine_Amount']).to_csv(FINAL_OUTPUT_PATH, index=False)
        return

    # Initialize Violation List for each row
    df_master['Violations_Data'] = [[] for _ in range(len(df_master))]

    # ---------------------------------------------------------
    # 2. Merge Seatbelt Data
    # ---------------------------------------------------------
    if os.path.exists(SEATBELT_LOG_PATH):
        try:
            df_seatbelt = pd.read_csv(SEATBELT_LOG_PATH)
            df_seatbelt['vehicle_id'] = df_seatbelt['vehicle_id'].astype(str)
            df_seatbelt['join_id'] = df_seatbelt['vehicle_id'].apply(normalize_id)
            
            # Filter for violations
            seatbelt_violations = df_seatbelt[df_seatbelt['label'].astype(str).str.lower().str.contains('no') & 
                                              df_seatbelt['label'].astype(str).str.lower().str.contains('seatbelt')]
            
            seatbelt_map = {}
            for _, row in seatbelt_violations.iterrows():
                jid = row['join_id']
                img_name = f"detected_{row['image_name']}"
                
                if jid not in seatbelt_map:
                    seatbelt_map[jid] = []
                seatbelt_map[jid].append({
                    'violation': 'No Seatbelt',
                    'fine': FINE_AMOUNTS.get('No Seatbelt', 1000),
                    'image_name': img_name,
                    'type': 'seatbelt'
                })
            
            # Apply to master
            for index, row in df_master.iterrows():
                jid = row['join_id']
                if jid in seatbelt_map:
                    orig_id = row['Vehicle_ID']
                    category = row['Category']
                    
                    for item in seatbelt_map[jid]:
                        # Seatbelt script only runs on Cars anyway, but safe to check
                        if category == 'Cars':
                            img_path = os.path.join(OUTPUT_DIR, "seatbelt_results", "Cars", orig_id, item['image_name'])
                            df_master.at[index, 'Violations_Data'].append({
                                'violation': 'No Seatbelt',
                                'fine': item['fine'],
                                'image_path': img_path
                            })
                            
            print(f"✅ Merged Seatbelt Data")
        except Exception as e:
            print(f"⚠️ Error merging seatbelt data: {e}")

    # ---------------------------------------------------------
    # 3. Merge Helmet Data (Skipped for Cars, but kept logic safe)
    # ---------------------------------------------------------
    # Since we filtered df_master for 'Cars' only, this section won't match anything,
    # effectively ignoring helmet violations (which is what you want for now).
    if os.path.exists(HELMET_LOG_PATH):
        print(f"ℹ️ Helmet data available but skipping merge (focusing on Cars only).")

    # ---------------------------------------------------------
    # 4. Construct Final Columns
    # ---------------------------------------------------------
    
    final_rows = []
    max_violations = 0
    
    for _, row in df_master.iterrows():
        # Base info
        vehicle_id = row['Vehicle_ID']
        category = row['Category']
        plate_number = row['Plate_Number']
        plate_conf = row['Plate_Confidence']
        
        # Images
        car_image_path = os.path.join(OUTPUT_DIR, "cropped_photos", category, vehicle_id, "BEST.jpg")
        plate_image_path = os.path.join(OUTPUT_DIR, "cropped_photos", category, vehicle_id, "BEST_PLATE_CROP.jpg")
        
        # Violations (Raw list)
        violations = row['Violations_Data']
        
        # --- DEDUPLICATION LOGIC ---
        seen_types = set()
        unique_violations = []
        
        for v in violations:
            v_type = v['violation']
            if v_type not in seen_types:
                seen_types.add(v_type)
                unique_violations.append(v)
        
        # --- CALCULATE FINE AFTER DEDUPLICATION ---
        total_fine = sum(v.get('fine', 0) for v in unique_violations)
        
        # Create row dict
        new_row = {
            'Vehicle_ID': vehicle_id,
            'Category': category,
            'Plate_Number': plate_number,
            'Plate_Confidence': plate_conf,
            'Car_Image': car_image_path,
            'Plate_Image': plate_image_path,
            'Total_Fine_Amount': total_fine
        }
        
        for i, v in enumerate(unique_violations):
            idx = i + 1
            new_row[f'Violation_{idx}'] = v['violation']
            new_row[f'Violation_Fine_{idx}'] = v['fine']
            new_row[f'Violation_Image_{idx}'] = v['image_path']
        
        if len(unique_violations) > max_violations:
            max_violations = len(unique_violations)
            
        # FILTER: Only add to final list if there is at least one violation
        if len(unique_violations) > 0:
            final_rows.append(new_row)
        
    # Create DataFrame
    df_final = pd.DataFrame(final_rows)
    
    if df_final.empty:
        print("\n⚠️ No violations found to consolidate.")
        # Create empty csv with headers to prevent errors downstream
        cols = ['Vehicle_ID', 'Category', 'Plate_Number', 'Plate_Confidence', 'Total_Fine_Amount', 'Car_Image', 'Plate_Image']
        pd.DataFrame(columns=cols).to_csv(FINAL_OUTPUT_PATH, index=False)
        return

    # Ensure all columns exist up to max_violations (or at least 1)
    max_violations = max(max_violations, 1)
    
    # Define column order
    cols = ['Vehicle_ID', 'Category', 'Plate_Number', 'Plate_Confidence', 'Total_Fine_Amount', 'Car_Image', 'Plate_Image']
    for i in range(1, max_violations + 1):
        cols.append(f'Violation_{i}')
        cols.append(f'Violation_Fine_{i}')
        cols.append(f'Violation_Image_{i}')
        
    # Add missing columns with empty strings/zeros
    for col in cols:
        if col not in df_final.columns:
            df_final[col] = ""
            
    # Reorder
    df_final = df_final[cols]
    
    # Save
    df_final.to_csv(FINAL_OUTPUT_PATH, index=False)
    print("\n" + "="*50)
    print(f"🎉 CONSOLIDATED CSV CREATED (CARS ONLY): {FINAL_OUTPUT_PATH}")
    print("="*50)
    print(df_final.head())

if __name__ == "__main__":
    main()