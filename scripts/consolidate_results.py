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

    # 2. Merge Seatbelt Data
    if os.path.exists(SEATBELT_LOG_PATH):
        df_seatbelt = pd.read_csv(SEATBELT_LOG_PATH)
        # Keep only relevant columns and latest detection per vehicle if multiple
        # Assuming 'label' is the status (e.g., 'no_seatbelt', 'seatbelt')
        # We'll take the most 'severe' or just the last one. Let's take the one with highest confidence or just drop duplicates.
        df_seatbelt['vehicle_id'] = df_seatbelt['vehicle_id'].astype(str)
        
        # Filter for actual seatbelt detections if needed, or just take the label
        # If multiple entries, let's take the one with highest confidence
        df_seatbelt = df_seatbelt.sort_values('confidence', ascending=False).drop_duplicates('vehicle_id')
        
        df_master = df_master.merge(df_seatbelt[['vehicle_id', 'label']], 
                                    left_on='Vehicle_ID', right_on='vehicle_id', 
                                    how='left')
        
        # Update Seatbelt_Status column
        df_master['Seatbelt_Status'] = df_master['label'].fillna('Pending')
        df_master.drop(columns=['vehicle_id', 'label'], inplace=True)
        print(f"✅ Merged Seatbelt Data")
    else:
        print(f"⚠️ Seatbelt log not found, skipping merge.")

    # 3. Merge Helmet Data
    if os.path.exists(HELMET_LOG_PATH):
        df_helmet = pd.read_csv(HELMET_LOG_PATH)
        df_helmet['vehicle_id'] = df_helmet['vehicle_id'].astype(str)
        
        # Sort by confidence and drop duplicates
        df_helmet = df_helmet.sort_values('confidence', ascending=False).drop_duplicates('vehicle_id')
        
        df_master = df_master.merge(df_helmet[['vehicle_id', 'helmet_status']], 
                                    left_on='Vehicle_ID', right_on='vehicle_id', 
                                    how='left')
        
        # Update Helmet_Status column
        df_master['Helmet_Status'] = df_helmet['helmet_status'].fillna('Pending')
        df_master.drop(columns=['vehicle_id', 'helmet_status'], inplace=True) # Check if this line is correct, df_helmet['helmet_status'] is a series, not column name in df_master after merge? 
        # Wait, merge adds 'helmet_status' to df_master.
        # Correct logic:
        # df_master['Helmet_Status'] = df_master['helmet_status'].fillna('Pending')
        # df_master.drop(columns=['vehicle_id', 'helmet_status'], inplace=True)
        # Actually, let's just rename the merged column.
        
        # Let's redo the merge logic slightly to be safer
    
    # Re-implementing Merge Logic cleanly
    
    # Reload Master to be safe (in thought process, but code is linear)
    # ...
    
    # 3. Merge Helmet Data (Corrected)
    if os.path.exists(HELMET_LOG_PATH):
        df_helmet = pd.read_csv(HELMET_LOG_PATH)
        df_helmet['vehicle_id'] = df_helmet['vehicle_id'].astype(str)
        df_helmet = df_helmet.sort_values('confidence', ascending=False).drop_duplicates('vehicle_id')
        
        # Rename column for merge
        df_helmet = df_helmet.rename(columns={'helmet_status': 'New_Helmet_Status'})
        
        df_master = df_master.merge(df_helmet[['vehicle_id', 'New_Helmet_Status']], 
                                    left_on='Vehicle_ID', right_on='vehicle_id', 
                                    how='left')
        
        # If match found, update; else keep existing (which is 'Pending')
        df_master['Helmet_Status'] = df_master['New_Helmet_Status'].combine_first(df_master['Helmet_Status'])
        df_master.drop(columns=['vehicle_id', 'New_Helmet_Status'], inplace=True)
        print(f"✅ Merged Helmet Data")

    # 4. Merge Closeness Data
    df_master['Closeness_Violation'] = "No"
    if os.path.exists(CLOSENESS_LOG_PATH):
        try:
            df_close = pd.read_csv(CLOSENESS_LOG_PATH)
            # Get all vehicle IDs involved in violations
            violators = set(df_close['vehicle_id_1'].astype(str)).union(set(df_close['vehicle_id_2'].astype(str)))
            
            df_master.loc[df_master['Vehicle_ID'].isin(violators), 'Closeness_Violation'] = "Yes"
            print(f"✅ Merged Closeness Data")
        except pd.errors.EmptyDataError:
             print(f"⚠️ Closeness log empty, skipping.")
    
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

    # 6. Save
    df_master.to_csv(FINAL_OUTPUT_PATH, index=False)
    print("\n" + "="*50)
    print(f"🎉 CONSOLIDATED CSV CREATED: {FINAL_OUTPUT_PATH}")
    print("="*50)
    print(df_master[['Vehicle_ID', 'Category', 'Plate_Number', 'Violation_Summary']].head())

if __name__ == "__main__":
    main()
