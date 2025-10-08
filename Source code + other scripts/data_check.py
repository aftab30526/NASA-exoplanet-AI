import pandas as pd
import numpy as np
import os

print("🔍 NASA EXOPLANET AI - DATA DEBUGGING SCRIPT")
print("=" * 50)

# Load data from the exact path you provided
try:
    data_path = "Data/cleaned_ready/"
    print(f"📁 Looking for data in: {data_path}")
    
    # List all files in the directory
    if os.path.exists(data_path):
        files = os.listdir(data_path)
        csv_files = [f for f in files if f.endswith('.csv')]
        
        if csv_files:
            print(f"✅ Found {len(csv_files)} CSV files:")
            for i, file in enumerate(csv_files):
                print(f"   {i+1}. {file}")
            
            # Load the first CSV file
            file_to_load = csv_files[0]
            full_path = os.path.join(data_path, file_to_load)
            print(f"\n📊 Loading: {full_path}")
            
            data = pd.read_csv(full_path)
            print(f"✅ Successfully loaded: {file_to_load}")
            
        else:
            print("❌ No CSV files found in Data/cleaned_ready/")
            print("Available files:")
            for file in files:
                print(f"   - {file}")
            exit()
    else:
        print(f"❌ Path does not exist: {data_path}")
        exit()
        
except Exception as e:
    print(f"❌ Error loading data: {e}")
    exit()

# Now analyze the data
print("\n" + "="*50)
print("📈 DATA ANALYSIS RESULTS")
print("="*50)

# 1. Basic Info
print("1. 📊 BASIC DATASET INFO")
print(f"   • Shape: {data.shape} (rows: {data.shape[0]}, columns: {data.shape[1]})")
print(f"   • Memory: {data.memory_usage(deep=True).sum() / (1024 * 1024):.2f} MB")
print(f"   • Column names:")
for col in data.columns:
    print(f"     - {col}")

# 2. Data Types
print("\n2. 🏷️ DATA TYPES")
print(data.dtypes)

# 3. Missing Values
print("\n3. ❌ MISSING VALUES ANALYSIS")
missing_values = data.isnull().sum()
missing_percent = (missing_values / len(data)) * 100

missing_info = pd.DataFrame({
    'Missing Count': missing_values,
    'Missing %': missing_percent
})

missing_cols = missing_info[missing_info['Missing Count'] > 0]
if len(missing_cols) > 0:
    print("Columns with missing values:")
    print(missing_cols)
else:
    print("   ✅ Excellent! No missing values found!")

# 4. Check for target columns (common exoplanet column names)
print("\n4. 🎯 LOOKING FOR TARGET/PREDICTION COLUMNS")
target_keywords = ['target', 'label', 'class', 'confirmed', 'candidate', 'false', 'planet', 'exoplanet', 'prediction', 'result', 'status', 'type', 'classification']

target_columns = []
for col in data.columns:
    col_lower = col.lower()
    if any(keyword in col_lower for keyword in target_keywords):
        target_columns.append(col)

if target_columns:
    print("   ✅ Found potential target columns:")
    for col in target_columns:
        unique_vals = data[col].unique()
        value_counts = data[col].value_counts()
        print(f"   • {col}:")
        print(f"     Unique values: {unique_vals}")
        print(f"     Value counts:")
        for val, count in value_counts.items():
            print(f"       - {val}: {count} samples ({(count/len(data))*100:.1f}%)")
else:
    print("   ⚠️  No obvious target columns found.")
    print("   Available columns:")
    for col in data.columns:
        print(f"     - {col}")

# 5. Basic Statistics
print("\n5. 📊 NUMERICAL DATA STATISTICS")
numerical_data = data.select_dtypes(include=[np.number])
if len(numerical_data.columns) > 0:
    print(numerical_data.describe())
else:
    print("   No numerical columns found.")

# 6. Check first few rows
print("\n6. 👀 FIRST 3 ROWS OF DATA")
print(data.head(3))

# 7. Check for constant columns
print("\n7. 🔍 CONSTANT COLUMNS CHECK")
constant_cols = []
for col in data.columns:
    if data[col].nunique() <= 1:
        constant_cols.append(col)

if constant_cols:
    print(f"   ⚠️  Constant columns (may need removal): {constant_cols}")
else:
    print("   ✅ No constant columns found.")

print("\n" + "="*50)
print("✅ DATA CHECK COMPLETE!")
print("\n📝 SUMMARY:")
print(f"• Dataset: {data.shape[0]} rows, {data.shape[1]} columns")
print(f"• Target columns found: {len(target_columns)}")
print(f"• Missing values: {missing_values.sum()} total")
print(f"• Constant columns: {len(constant_cols)}")

print("\n🎯 NEXT STEPS:")
print("1. Identify which column is your target (for prediction)")
print("2. Check if missing values need handling")
print("3. Remove any constant columns if found")
print("4. Verify data types look correct")