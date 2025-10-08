# =============================================================================
# ULTIMATE DATA PREPARATION SCRIPT FOR EXOPLANET CLASSIFICATION
# =============================================================================

import pandas as pd
import numpy as np
import shutil
import json
from pathlib import Path
from collections import Counter
import os

# =============================================================================
# CONFIGURATION
# =============================================================================
class Config:
    # Class names
    CLASS_NAMES = {0: "FALSE POSITIVE", 1: "CANDIDATE", 2: "CONFIRMED"}
    
    # Essential features for exoplanet detection
    ESSENTIAL_FEATURES = [
        'transit_depth', 'orbital_period', 'planet_radius', 
        'stellar_radius', 'stellar_mass', 'equilibrium_temp',
        'transit_duration', 'impact_parameter', 'stellar_temp',
        'stellar_gravity', 'orbital_frequency', 'radius_ratio',
        'log_depth'
    ]
    
    # Realistic value ranges
    REALISTIC_RANGES = {
        'transit_depth': (1, 100000),           # ppm (minimum 1, not 0)
        'orbital_period': (0.1, 10000),         # days
        'planet_radius': (0.01, 50.0),          # Earth radii
        'stellar_radius': (0.01, 30.0),         # Solar radii
        'stellar_mass': (0.01, 10.0),           # Solar masses
        'equilibrium_temp': (50, 6000),         # Kelvin
        'transit_duration': (0.1, 48.0),        # hours
        'impact_parameter': (0.0, 1.0),         # unitless
        'stellar_temp': (1000, 15000),          # Kelvin
        'stellar_gravity': (2.0, 6.0),          # log10(cm/s²)
        'orbital_frequency': (0.0001, 10.0),    # 1/days
        'radius_ratio': (0.0001, 0.2),          # unitless
        'log_depth': (0.0, 6.0)                 # log10(transit_depth)
    }
    
    # Target column names
    TARGET_COLUMNS = ['disposition', 'label', 'target', 'class', 'koi_disposition', 
                     'exoplanet_archive_disposition', 'pl_letter', 'status']

# =============================================================================
# ULTIMATE DATA PREPARER
# =============================================================================
class UltimateDataPreparer:
    def __init__(self, data_path):
        self.data_path = Path(data_path)
        self.useful_dir = self.data_path / "USEFUL_TRAINING_DATA"
        self.useless_dir = self.data_path / "USELESS_DATA"
        self.useful_dir.mkdir(exist_ok=True)
        self.useless_dir.mkdir(exist_ok=True)
        
    def read_any_file(self, file_path):
        """Read any file format (CSV, Excel, JSON)"""
        try:
            if file_path.suffix.lower() == '.csv':
                return pd.read_csv(file_path, low_memory=False)
            elif file_path.suffix.lower() in ['.xlsx', '.xls']:
                return pd.read_excel(file_path)
            elif file_path.suffix.lower() == '.json':
                return pd.read_json(file_path)
            else:
                print(f"❌ Unsupported format: {file_path}")
                return None
        except Exception as e:
            print(f"❌ Error reading {file_path}: {e}")
            return None
    
    def clean_features(self, df):
        """Clean and validate features"""
        if df is None or df.empty:
            return None
            
        # Create a copy
        df_clean = df.copy()
        
        # Select only numeric columns
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) < 3:
            return None
            
        # Fill NaN values with median
        for col in numeric_cols:
            df_clean[col] = df_clean[col].fillna(df_clean[col].median())
        
        # Remove infinite values
        df_clean = df_clean.replace([np.inf, -np.inf], np.nan)
        df_clean = df_clean.fillna(0)
        
        # Remove negative values for features that shouldn't be negative
        positive_features = [f for f in Config.ESSENTIAL_FEATURES if f in df_clean.columns]
        for feature in positive_features:
            if feature not in ['impact_parameter']:  # impact_parameter can be 0-1
                df_clean[feature] = df_clean[feature].clip(lower=0.0001)
        
        # Apply realistic value ranges
        for feature, (min_val, max_val) in Config.REALISTIC_RANGES.items():
            if feature in df_clean.columns:
                df_clean[feature] = df_clean[feature].clip(lower=min_val, upper=max_val)
        
        return df_clean
    
    def find_or_create_target(self, df):
        """Find existing target column or create synthetic one"""
        # Look for existing target column
        for target_col in Config.TARGET_COLUMNS:
            if target_col in df.columns:
                # Map to numerical classes
                disposition_map = {
                    'FALSE POSITIVE': 0, 'False Positive': 0, 'false positive': 0, 
                    'FALSE_POSITIVE': 0, 'FP': 0, '0': 0,
                    'CANDIDATE': 1, 'Candidate': 1, 'candidate': 1, 
                    'CANDIDATE': 1, 'CP': 1, '1': 1,
                    'CONFIRMED': 2, 'Confirmed': 2, 'confirmed': 2, 
                    'CONFIRMED': 2, 'PC': 2, '2': 2
                }
                
                y = df[target_col].astype(str).str.upper().map(disposition_map)
                y = y.fillna(0)  # Treat unknown as FALSE POSITIVE
                return y.astype(int)
        
        # If no target found, create synthetic labels based on feature patterns
        print("⚠️  No target column found, creating synthetic labels...")
        return self.create_synthetic_labels(df)
    
    def create_synthetic_labels(self, df):
        """Create realistic synthetic labels based on feature patterns"""
        y_synthetic = np.zeros(len(df))  # Start with all FALSE POSITIVE
        
        # Define patterns for different classes
        for idx in range(len(df)):
            # Check for CANDIDATE patterns (Earth-like)
            if ('planet_radius' in df.columns and 
                0.5 <= df['planet_radius'].iloc[idx] <= 2.5 and
                'orbital_period' in df.columns and 
                50 <= df['orbital_period'].iloc[idx] <= 400):
                y_synthetic[idx] = 1  # CANDIDATE
            
            # Check for CONFIRMED patterns (clear exoplanet signals)
            elif ('transit_depth' in df.columns and 
                  df['transit_depth'].iloc[idx] > 1000 and
                  'impact_parameter' in df.columns and 
                  df['impact_parameter'].iloc[idx] < 0.5):
                y_synthetic[idx] = 2  # CONFIRMED
        
        return y_synthetic.astype(int)
    
    def analyze_and_classify_files(self):
        """Analyze all files and classify as useful/useless"""
        print("🔍 ANALYZING ALL DATA FILES...")
        print("="*70)
        
        all_files = []
        for ext in ['*.csv', '*.xlsx', '*.xls', '*.json']:
            all_files.extend(list(self.data_path.glob(ext)))
        
        useful_files = []
        useless_files = []
        file_stats = {}
        
        for file_path in all_files:
            print(f"📊 Analyzing: {file_path.name}")
            
            try:
                # Read file
                df = self.read_any_file(file_path)
                if df is None or df.empty:
                    useless_files.append(file_path)
                    print(f"   ❌ Empty or unreadable → USELESS")
                    continue
                
                # Clean features
                df_clean = self.clean_features(df)
                if df_clean is None:
                    useless_files.append(file_path)
                    print(f"   ❌ Not enough numeric features → USELESS")
                    continue
                
                # Get or create target
                y = self.find_or_create_target(df_clean)
                
                # Check if we have at least 2 classes
                class_dist = Counter(y)
                unique_classes = len([c for c in class_dist.keys() if class_dist[c] > 0])
                
                if unique_classes >= 2 and len(df_clean) >= 10:
                    useful_files.append(file_path)
                    
                    # Save cleaned version with target
                    df_clean['target_class'] = y
                    cleaned_filename = f"cleaned_{file_path.stem}.csv"
                    cleaned_path = self.useful_dir / cleaned_filename
                    df_clean.to_csv(cleaned_path, index=False)
                    
                    file_stats[file_path.name] = {
                        'original_samples': len(df),
                        'cleaned_samples': len(df_clean),
                        'class_distribution': dict(class_dist),
                        'status': 'USEFUL'
                    }
                    
                    print(f"   ✅ USEFUL | Samples: {len(df_clean):,} | Classes: {dict(class_dist)}")
                    
                else:
                    useless_files.append(file_path)
                    file_stats[file_path.name] = {
                        'original_samples': len(df),
                        'cleaned_samples': len(df_clean) if df_clean else 0,
                        'class_distribution': dict(class_dist),
                        'status': 'USELESS'
                    }
                    print(f"   ❌ USELESS | Reason: Only {unique_classes} class(es)")
                    
            except Exception as e:
                useless_files.append(file_path)
                file_stats[file_path.name] = {
                    'status': 'ERROR',
                    'error': str(e)
                }
                print(f"   ❌ ERROR → USELESS: {e}")
        
        return useful_files, useless_files, file_stats
    
    def combine_and_balance_data(self, useful_files):
        """Combine all useful data and balance classes"""
        print(f"\n⚖️ COMBINING AND BALANCING DATA...")
        print("="*70)
        
        all_data = []
        
        for file_path in useful_files:
            cleaned_file = self.useful_dir / f"cleaned_{file_path.stem}.csv"
            if cleaned_file.exists():
                df = pd.read_csv(cleaned_file)
                all_data.append(df)
                print(f"✅ Added: {file_path.stem} - {len(df)} samples")
        
        if not all_data:
            print("❌ No useful data found!")
            return None
        
        # Combine all data
        combined_data = pd.concat(all_data, ignore_index=True)
        
        # Check current distribution
        current_dist = Counter(combined_data['target_class'])
        total_samples = len(combined_data)
        
        print(f"\n📊 COMBINED DISTRIBUTION:")
        for cls in [0, 1, 2]:
            count = current_dist[cls]
            percentage = count / total_samples * 100
            print(f"  {Config.CLASS_NAMES[cls]:15}: {count:6,} samples ({percentage:5.1f}%)")
        
        # Calculate target counts for perfect balance
        target_count = min(current_dist.values())  # Use the smallest class as target
        target_count = min(target_count, 10000)    # Cap at 10,000 to avoid memory issues
        
        print(f"\n🎯 TARGET: {target_count:,} samples per class")
        
        # Balance the data
        balanced_data_list = []
        
        for cls in [0, 1, 2]:
            cls_data = combined_data[combined_data['target_class'] == cls]
            current_count = len(cls_data)
            
            if current_count < target_count:
                # Oversample minority class
                indices = np.random.choice(
                    cls_data.index, 
                    size=target_count - current_count, 
                    replace=True
                )
                additional_data = cls_data.loc[indices]
                balanced_cls_data = pd.concat([cls_data, additional_data])
                action = f"+{target_count - current_count}"
            else:
                # Undersample majority class
                indices = np.random.choice(
                    cls_data.index, 
                    size=target_count, 
                    replace=False
                )
                balanced_cls_data = cls_data.loc[indices]
                action = f"-{current_count - target_count}"
            
            balanced_data_list.append(balanced_cls_data)
            print(f"✅ {Config.CLASS_NAMES[cls]:15}: {current_count:5} → {target_count:5} ({action})")
        
        # Combine balanced data
        balanced_data = pd.concat(balanced_data_list, ignore_index=True)
        
        # Shuffle the data
        balanced_data = balanced_data.sample(frac=1, random_state=42).reset_index(drop=True)
        
        # Final distribution check
        final_dist = Counter(balanced_data['target_class'])
        print(f"\n📈 PERFECTLY BALANCED DISTRIBUTION:")
        for cls in [0, 1, 2]:
            count = final_dist[cls]
            percentage = count / len(balanced_data) * 100
            print(f"  {Config.CLASS_NAMES[cls]:15}: {count:6,} samples ({percentage:5.1f}%)")
        
        return balanced_data
    
    def save_final_training_data(self, balanced_data):
        """Save final balanced training data"""
        print(f"\n💾 SAVING FINAL TRAINING DATA...")
        print("="*70)
        
        # Save as multiple files for easier handling
        file_size = 2000  # samples per file
        num_files = (len(balanced_data) + file_size - 1) // file_size
        
        training_files = []
        for i in range(num_files):
            start_idx = i * file_size
            end_idx = min((i + 1) * file_size, len(balanced_data))
            
            file_data = balanced_data.iloc[start_idx:end_idx]
            filename = self.useful_dir / f"FINAL_TRAINING_{i+1:02d}.csv"
            
            file_data.to_csv(filename, index=False)
            training_files.append(filename)
            print(f"✅ {filename.name:25} | Samples: {len(file_data):,} | Classes: {dict(Counter(file_data['target_class']))}")
        
        return training_files
    
    def move_useless_files(self, useless_files):
        """Move useless files to useless directory"""
        print(f"\n🗑️ MOVING USELESS FILES...")
        print("="*70)
        
        moved_count = 0
        for file_path in useless_files:
            try:
                dest_path = self.useless_dir / file_path.name
                shutil.move(str(file_path), str(dest_path))
                moved_count += 1
                print(f"✅ Moved: {file_path.name}")
            except Exception as e:
                print(f"❌ Failed to move {file_path.name}: {e}")
        
        return moved_count
    
    def generate_report(self, file_stats, useful_files, useless_files, training_files):
        """Generate comprehensive report"""
        print(f"\n📊 COMPREHENSIVE DATA PREPARATION REPORT")
        print("="*70)
        
        total_files = len(useful_files) + len(useless_files)
        useful_percentage = len(useful_files) / total_files * 100
        
        print(f"📁 TOTAL FILES PROCESSED: {total_files}")
        print(f"✅ USEFUL FILES: {len(useful_files)} ({useful_percentage:.1f}%)")
        print(f"❌ USELESS FILES: {len(useless_files)} ({100-useful_percentage:.1f}%)")
        print(f"🎯 FINAL TRAINING FILES: {len(training_files)}")
        
        if training_files:
            sample_df = pd.read_csv(training_files[0])
            print(f"📊 FEATURES AVAILABLE: {len(sample_df.columns)}")
            print(f"🎯 TARGET DISTRIBUTION: {dict(Counter(sample_df['target_class']))}")
        
        print(f"\n📂 OUTPUT DIRECTORIES:")
        print(f"   USEFUL DATA: {self.useful_dir}")
        print(f"   USELESS DATA: {self.useless_dir}")
        
        # Save detailed report
        report = {
            'summary': {
                'total_files': total_files,
                'useful_files': len(useful_files),
                'useless_files': len(useless_files),
                'training_files': len(training_files),
                'useful_percentage': useful_percentage
            },
            'file_details': file_stats,
            'timestamp': pd.Timestamp.now().isoformat()
        }
        
        report_path = self.data_path / "data_preparation_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"📄 DETAILED REPORT: {report_path}")

# =============================================================================
# MAIN EXECUTION
# =============================================================================
def main():
    """Main data preparation function"""
    
    DATA_PATH = r"C:\Users\RR Joshi Computer\Desktop\Nasa_exoplanet_ai\exoplanet_ml_ready\training_ready\train"
    
    print("🚀 ULTIMATE DATA PREPARATION FOR EXOPLANET CLASSIFICATION")
    print("="*70)
    
    # Initialize preparer
    preparer = UltimateDataPreparer(DATA_PATH)
    
    # Step 1: Analyze and classify files
    useful_files, useless_files, file_stats = preparer.analyze_and_classify_files()
    
    # Step 2: Move useless files
    moved_count = preparer.move_useless_files(useless_files)
    
    # Step 3: Combine and balance useful data
    balanced_data = preparer.combine_and_balance_data(useful_files)
    
    if balanced_data is not None:
        # Step 4: Save final training data
        training_files = preparer.save_final_training_data(balanced_data)
        
        # Step 5: Generate report
        preparer.generate_report(file_stats, useful_files, useless_files, training_files)
        
        print("\n" + "="*70)
        print("🎉 DATA PREPARATION COMPLETE!")
        print("="*70)
        print("✅ All files analyzed and classified")
        print("✅ Useless files moved to USELESS_DATA folder")
        print("✅ Useful data cleaned and balanced perfectly")
        print("✅ Final training data ready in USEFUL_TRAINING_DATA folder")
        print("✅ Perfect 33.3% distribution for all 3 classes")
        print("🎯 READY FOR MODEL TRAINING!")
        print("="*70)
        
        # Instructions for next step
        print(f"\n📝 NEXT STEP:")
        print(f"   Use this path for training: {preparer.useful_dir}")
        print(f"   Training files: FINAL_TRAINING_01.csv, FINAL_TRAINING_02.csv, etc.")
        
    else:
        print("❌ Data preparation failed - no usable data found!")

if __name__ == "__main__":
    main()