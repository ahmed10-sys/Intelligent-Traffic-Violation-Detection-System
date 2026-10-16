import sys
import pandas as pd
import os
import logging
from datetime import datetime

# Fix Windows console encoding for Unicode/emojis
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ==========================================
# 🗂️ CONFIGURATION
# ==========================================
OUTPUT_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "../output"))
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
    
    # Initialize Violation List for each row
    df_master['Violations_Data'] = [[] for _ in range(len(df_master))]

    # ---------------------------------------------------------
    # 2. Merge Seatbelt Data (Cars)
    # ---------------------------------------------------------
    if os.path.exists(SEATBELT_LOG_PATH):
        try:
            df_seatbelt = pd.read_csv(SEATBELT_LOG_PATH)
            df_seatbelt['vehicle_id'] = df_seatbelt['vehicle_id'].astype(str)
            df_seatbelt['join_id'] = df_seatbelt['vehicle_id'].apply(normalize_id)
            
            seatbelt_map = {}
            for jid, group in df_seatbelt.groupby('join_id'):
                # Look for explicit no-seatbelt detections
                nosb_rows = group[group['label'].astype(str).str.lower().str.contains('no') & 
                                  group['label'].astype(str).str.lower().str.contains('seatbelt')]
                if not nosb_rows.empty:
                    # Pick highest confidence violation frame as primary evidence
                    best_nosb = nosb_rows.sort_values('confidence', ascending=False).iloc[0]
                    img_name = f"detected_{best_nosb['image_name']}"
                    seatbelt_map[jid] = [{
                        'violation': 'No Seatbelt',
                        'fine': FINE_AMOUNTS.get('No Seatbelt', 5000),
                        'image_name': img_name,
                        'type': 'seatbelt'
                    }]
            
            # Apply to master
            for index, row in df_master.iterrows():
                jid = row['join_id']
                if jid in seatbelt_map and row['Category'] == 'Cars':
                    orig_id = row['Vehicle_ID']
                    for item in seatbelt_map[jid]:
                        img_path = os.path.join(OUTPUT_DIR, "seatbelt_results", "Cars", orig_id, item['image_name'])
                        df_master.at[index, 'Violations_Data'].append({
                            'violation': 'No Seatbelt',
                            'fine': item['fine'],
                            'image_path': img_path
                        })
                            
            print(f"✅ Merged Seatbelt Data (Highest-Confidence Evidence Selection)")
        except Exception as e:
            print(f"⚠️ Error merging seatbelt data: {e}")

    # ---------------------------------------------------------
    # 3. Merge Helmet Data (Bikes_Motorcycles)
    # ---------------------------------------------------------
    if os.path.exists(HELMET_LOG_PATH):
        try:
            df_helmet = pd.read_csv(HELMET_LOG_PATH)
            df_helmet['vehicle_id'] = df_helmet['vehicle_id'].astype(str)
            df_helmet['join_id'] = df_helmet['vehicle_id'].apply(normalize_id)
            
            helmet_map = {}
            for jid, group in df_helmet.groupby('join_id'):
                with_helmet_rows = group[(group['helmet_status'] == 'with_helmet') & (group['confidence'] >= 0.35)]
                without_helmet_rows = group[(group['helmet_status'] == 'without_helmet') & (group['confidence'] >= 0.35)]
                
                # CONSENSUS RULE:
                # If rider was clearly seen with helmet in ANY good frame, they are COMPLIANT!
                if not with_helmet_rows.empty:
                    continue
                
                # Genuine violation: Explicit without_helmet detected and NO with_helmet anywhere!
                if not without_helmet_rows.empty:
                    best_without = without_helmet_rows.sort_values('confidence', ascending=False).iloc[0]
                    img_name = f"helmet_{best_without['image_name']}"
                    helmet_map[jid] = [{
                        'violation': 'No Helmet',
                        'fine': FINE_AMOUNTS.get('No Helmet', 2000),
                        'image_name': img_name,
                        'type': 'helmet'
                    }]
                else:
                    # All frames had no_detection (head never seen). Pick best available close-up frame.
                    zone2_or_best = group[group['image_name'].str.contains('BEST|zone2')]
                    fallback = zone2_or_best.iloc[0] if not zone2_or_best.empty else group.sort_values('image_name', ascending=False).iloc[0]
                    img_name = f"helmet_{fallback['image_name']}"
                    helmet_map[jid] = [{
                        'violation': 'No Helmet',
                        'fine': FINE_AMOUNTS.get('No Helmet', 2000),
                        'image_name': img_name,
                        'type': 'helmet'
                    }]
            
            for index, row in df_master.iterrows():
                jid = row['join_id']
                if jid in helmet_map and row['Category'] == 'Bikes_Motorcycles':
                    orig_id = row['Vehicle_ID']
                    for item in helmet_map[jid]:
                        img_path = os.path.join(OUTPUT_DIR, "helmet_results", "Bikes_Motorcycles", orig_id, item['image_name'])
                        df_master.at[index, 'Violations_Data'].append({
                            'violation': 'No Helmet',
                            'fine': item['fine'],
                            'image_path': img_path
                        })
            print(f"✅ Merged Helmet Data (Consensus & Evidence Selection)")
        except Exception as e:
            print(f"⚠️ Error merging helmet data: {e}")

    # ---------------------------------------------------------
    # 4. Construct Final Columns
    # ---------------------------------------------------------
    
    final_rows = []
    max_violations = 0
    
    # Get Current Date for the Issue_Date column
    current_date = datetime.now().strftime("%Y-%m-%d")
    
    for _, row in df_master.iterrows():
        # Base info
        vehicle_id = row['Vehicle_ID']
        category = row['Category']
        plate_number = row['Plate_Number']
        plate_conf = row['Plate_Confidence']
        
        # Images
        car_dir = os.path.join(OUTPUT_DIR, "cropped_photos", category, vehicle_id)
        car_image_path = os.path.join(car_dir, "BEST.jpg")
        if not os.path.exists(car_image_path) and os.path.isdir(car_dir):
            cands = [f for f in os.listdir(car_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png')) and "plate" not in f.lower()]
            if cands:
                cands.sort()
                car_image_path = os.path.join(car_dir, cands[-1])
        plate_image_path = os.path.join(car_dir, "BEST_PLATE_CROP.jpg")
        
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
            'Total_Fine_Amount': total_fine,
            'Issue_Date': current_date,  # 📅 Set to current date
            'Status': "Pending"          # 🟡 String status
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
        # Create empty csv with headers
        cols = ['Vehicle_ID', 'Category', 'Plate_Number', 'Plate_Confidence', 'Total_Fine_Amount', 'Issue_Date', 'Status', 'Car_Image', 'Plate_Image']
        pd.DataFrame(columns=cols).to_csv(FINAL_OUTPUT_PATH, index=False)
        return

    # Ensure all columns exist up to max_violations (or at least 1)
    max_violations = max(max_violations, 1)
    
    # Define column order
    cols = ['Vehicle_ID', 'Category', 'Plate_Number', 'Plate_Confidence', 'Total_Fine_Amount', 'Issue_Date', 'Status', 'Car_Image', 'Plate_Image']
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