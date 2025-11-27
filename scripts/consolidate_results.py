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

    # Ensure Vehicle_ID is string for consistent merging
    df_master['Vehicle_ID'] = df_master['Vehicle_ID'].astype(str)
    
    # Create a normalized ID for joining (remove 'id_' and leading zeros to match other logs)
    # Example: 'id_002' -> '2', 'id_018' -> '18', 'id_-01' -> '-1'
    def normalize_id(val):
        val = str(val).lower().replace('id_', '')
        try:
            return str(int(val))
        except:
            return val

    df_master['join_id'] = df_master['Vehicle_ID'].apply(normalize_id)
    print(f"Sample Normalized IDs (Master): {df_master['join_id'].head().tolist()}")

    # 2. Merge Seatbelt Data
    if os.path.exists(SEATBELT_LOG_PATH):
        try:
            df_seatbelt = pd.read_csv(SEATBELT_LOG_PATH)
            # Keep only relevant columns and latest detection per vehicle if multiple
            df_seatbelt['vehicle_id'] = df_seatbelt['vehicle_id'].astype(str)
            
            # Filter for actual seatbelt detections if needed, or just take the label
            # If multiple entries, let's take the one with highest confidence
            df_seatbelt = df_seatbelt.sort_values('confidence', ascending=False).drop_duplicates('vehicle_id')
            
            # Normalize seatbelt ID
            df_seatbelt['join_id'] = df_seatbelt['vehicle_id'].apply(normalize_id)
            
            df_master = df_master.merge(df_seatbelt[['join_id', 'label']], 
                                        on='join_id', 
                                        how='left')
            
            # Update Seatbelt_Status column
            df_master['Seatbelt_Status'] = df_master['label'].fillna('Pending')
            df_master.drop(columns=['label'], inplace=True)
            print(f"✅ Merged Seatbelt Data")
        except Exception as e:
            print(f"⚠️ Error merging seatbelt data: {e}")
    else:
        print(f"⚠️ Seatbelt log not found, skipping merge.")

    # 3. Merge Helmet Data
    if os.path.exists(HELMET_LOG_PATH):
        try:
            df_helmet = pd.read_csv(HELMET_LOG_PATH)
            df_helmet['vehicle_id'] = df_helmet['vehicle_id'].astype(str)
            
            # Sort by confidence and drop duplicates
            df_helmet = df_helmet.sort_values('confidence', ascending=False).drop_duplicates('vehicle_id')
            
            # Normalize helmet ID
            df_helmet['join_id'] = df_helmet['vehicle_id'].apply(normalize_id)
            
            # Rename column for merge
            df_helmet = df_helmet.rename(columns={'helmet_status': 'New_Helmet_Status'})
            
            df_master = df_master.merge(df_helmet[['join_id', 'New_Helmet_Status']], 
                                        on='join_id', 
                                        how='left')
            
            # If match found, update; else keep existing (which is 'Pending')
            df_master['Helmet_Status'] = df_master['New_Helmet_Status'].combine_first(df_master['Helmet_Status'])
            df_master.drop(columns=['New_Helmet_Status'], inplace=True)
            print(f"✅ Merged Helmet Data")
        except Exception as e:
            print(f"⚠️ Error merging helmet data: {e}")
    else:
        print(f"⚠️ Helmet log not found, skipping merge.")

    # 4. Merge Closeness Data
    df_master['Closeness_Violation'] = "No"
    if os.path.exists(CLOSENESS_LOG_PATH):
        try:
            df_close = pd.read_csv(CLOSENESS_LOG_PATH)
            # Get all vehicle IDs involved in violations
            violators = set(df_close['vehicle_id_1'].apply(normalize_id)).union(set(df_close['vehicle_id_2'].apply(normalize_id)))
            
            df_master.loc[df_master['join_id'].isin(violators), 'Closeness_Violation'] = "Yes"
            print(f"✅ Merged Closeness Data")
        except pd.errors.EmptyDataError:
             print(f"⚠️ Closeness log empty, skipping.")
        except Exception as e:
            print(f"⚠️ Error merging closeness data: {e}")
    
    # 5. Generate Summary
    def generate_summary(row):
        violations = []
        if row['Seatbelt_Status'] == 'no_seatbelt':
            violations.append("No Seatbelt")
        if row['Helmet_Status'] == 'without_helmet':
            violations.append("No Helmet")
        if row['Closeness_Violation'] == 'Yes':
            violations.append("Tailgating")
        
        if not violations:
            return "Clean"
        return ", ".join(violations)

    df_master['Violation_Summary'] = df_master.apply(generate_summary, axis=1)

    # 6. Save (Drop join_id first)
    if 'join_id' in df_master.columns:
        df_master.drop(columns=['join_id'], inplace=True)
        
    df_master.to_csv(FINAL_OUTPUT_PATH, index=False)
    print("\n" + "="*50)
    print(f"🎉 CONSOLIDATED CSV CREATED: {FINAL_OUTPUT_PATH}")
    print("="*50)
    print(df_master[['Vehicle_ID', 'Category', 'Plate_Number', 'Violation_Summary']].head())

if __name__ == "__main__":
    main()
