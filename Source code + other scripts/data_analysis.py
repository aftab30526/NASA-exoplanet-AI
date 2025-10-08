"""
NASA EXOPLANET DATA COMPREHENSIVE ANALYSIS - FIXED VERSION
===========================================================
- Handles all data types including numpy
- Identifies ALL usable files for training
- Preserves all original data
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path
import json
from datetime import datetime

# === CONFIGURATION ===
CONFIG = {
    "raw_folder": Path(r"C:\Users\RR Joshi Computer\Desktop\Nasa_exoplanet_ai\Data\raw"),
    "analysis_folder": Path(r"C:\Users\RR Joshi Computer\Desktop\Nasa_exoplanet_ai\Data\analysis"),
    "report_file": "data_analysis_report.json"
}

# Create analysis folder
CONFIG["analysis_folder"].mkdir(parents=True, exist_ok=True)

print("🔍 NASA EXOPLANET DATA COMPREHENSIVE ANALYSIS - FIXED")
print("=" * 70)

# === FIXED JSON ENCODER ===
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64, 
                          np.uint8, np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        elif isinstance(obj, (np.bool_)):
            return bool(obj)
        return super(NumpyEncoder, self).default(obj)

# === ANALYSIS FUNCTIONS ===

def get_file_info(file_path):
    """Get basic file information without loading full data"""
    try:
        file_stats = file_path.stat()
        return {
            "filename": file_path.name,
            "size_mb": float(file_stats.st_size / (1024 * 1024)),
            "modified": datetime.fromtimestamp(file_stats.st_mtime).isoformat()
        }
    except Exception as e:
        return {"filename": file_path.name, "error": str(e)}

def safe_read_csv(file_path, sample_rows=1000):
    """Safely read CSV with error handling"""
    try:
        # First try to read with limited rows to check structure
        df_sample = pd.read_csv(file_path, nrows=5)
        
        # Get total rows without loading everything
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            total_rows = sum(1 for line in f) - 1  # Subtract header
        
        # Now read sample for analysis
        if total_rows > sample_rows:
            df = pd.read_csv(file_path, nrows=sample_rows)
        else:
            df = pd.read_csv(file_path)
            
        return df, total_rows, None
        
    except Exception as e:
        return None, 0, str(e)

def analyze_columns(df, file_info):
    """Analyze column structure and types"""
    analysis = {
        "total_columns": len(df.columns),
        "column_names": list(df.columns),
        "data_types": df.dtypes.astype(str).to_dict(),
        "sample_data": {}
    }
    
    # Get sample values for first 5 columns
    for col in df.columns[:5]:
        unique_vals = df[col].dropna().unique()[:10]  # First 10 unique values
        analysis["sample_data"][col] = [str(v) for v in unique_vals] if len(unique_vals) > 0 else []
    
    return analysis

def find_target_columns(df):
    """Look for potential target columns"""
    target_keywords = [
        'koi_disposition', 'disposition', 'status', 'target', 'class', 'label',
        'tfopwg_disp', 'tess_disposition', 'exoplanet_archive_disposition'
    ]
    
    found_targets = []
    for col in df.columns:
        col_lower = col.lower()
        for keyword in target_keywords:
            if keyword in col_lower:
                target_info = {
                    "column_name": col,
                    "keyword_matched": keyword,
                    "unique_values": [],
                    "value_counts": {}
                }
                
                # Analyze target values
                unique_vals = df[col].dropna().unique()
                target_info["unique_values"] = [str(v) for v in unique_vals[:20]]  # First 20 values
                target_info["total_unique"] = int(len(unique_vals))
                
                # Get value counts for small number of unique values
                if len(unique_vals) <= 10:
                    value_counts = df[col].value_counts()
                    # Convert numpy types to Python native types
                    target_info["value_counts"] = {str(k): int(v) for k, v in value_counts.to_dict().items()}
                
                found_targets.append(target_info)
                break  # Only match once per column
    
    return found_targets

def find_feature_columns(df):
    """Look for important exoplanet feature columns"""
    feature_keywords = [
        # Kepler features
        'koi_period', 'koi_depth', 'koi_duration', 'koi_prad', 'koi_srad', 
        'koi_teq', 'koi_impact', 'koi_insol', 'koi_steff', 'koi_slogg',
        
        # General exoplanet features
        'period', 'depth', 'duration', 'radius', 'temp', 'impact', 'insolation',
        'stellar_temp', 'stellar_radius', 'stellar_gravity',
        
        # TESS features
        'tic', 'toi', 'tess_'
    ]
    
    found_features = []
    for col in df.columns:
        col_lower = col.lower()
        for keyword in feature_keywords:
            if keyword in col_lower:
                feature_info = {
                    "column_name": col,
                    "keyword_matched": keyword,
                    "data_type": str(df[col].dtype),
                    "missing_percent": float((df[col].isna().sum() / len(df)) * 100),
                    "numeric_stats": {}
                }
                
                # Add numeric statistics if applicable
                if pd.api.types.is_numeric_dtype(df[col]):
                    numeric_vals = df[col].dropna()
                    if len(numeric_vals) > 0:
                        feature_info["numeric_stats"] = {
                            "min": float(numeric_vals.min()),
                            "max": float(numeric_vals.max()),
                            "mean": float(numeric_vals.mean()),
                            "std": float(numeric_vals.std())
                        }
                
                found_features.append(feature_info)
                break
    
    return found_features

def analyze_data_quality(df, total_rows):
    """Analyze overall data quality"""
    quality = {
        "total_rows_analyzed": int(len(df)),
        "total_rows_in_file": int(total_rows),
        "missing_data_percent": float((df.isna().sum().sum() / (len(df) * len(df.columns))) * 100),
        "duplicate_rows": int(df.duplicated().sum()),
        "duplicate_percent": float((df.duplicated().sum() / len(df)) * 100),
        "constant_columns": []
    }
    
    # Find constant columns
    for col in df.columns:
        if df[col].nunique() <= 1:
            quality["constant_columns"].append(col)
    
    return quality

def assess_file_usability(file_analysis):
    """Assess how usable this file is for training"""
    score = 0
    reasons = []
    
    # Check for target columns
    if file_analysis["target_columns"]:
        score += 3
        reasons.append("Has target columns")
        
        # Check if target has reasonable classes
        for target in file_analysis["target_columns"]:
            if 2 <= target["total_unique"] <= 10:
                score += 2
                reasons.append("Target has reasonable class distribution")
    
    # Check for feature columns
    if file_analysis["feature_columns"]:
        score += 2
        reasons.append(f"Has {len(file_analysis['feature_columns'])} feature columns")
        
        if len(file_analysis["feature_columns"]) >= 5:
            score += 1
            reasons.append("Has sufficient features")
    
    # Check data quality
    if file_analysis["data_quality"]["missing_data_percent"] < 50:
        score += 1
        reasons.append("Low missing data")
    
    if file_analysis["data_quality"]["duplicate_percent"] < 20:
        score += 1
        reasons.append("Low duplicate rate")
    
    # Size check
    if file_analysis["data_quality"]["total_rows_in_file"] >= 100:
        score += 1
        reasons.append("Sufficient data size")
    
    # Determine usability level
    if score >= 8:
        usability = "HIGH"
    elif score >= 5:
        usability = "MEDIUM" 
    elif score >= 3:
        usability = "LOW"
    else:
        usability = "POOR"
    
    return {
        "score": int(score),
        "usability": usability,
        "reasons": reasons
    }

# === MAIN ANALYSIS ===

def analyze_all_files():
    """Main function to analyze all files"""
    
    if not CONFIG["raw_folder"].exists():
        print(f"❌ ERROR: Raw folder not found: {CONFIG['raw_folder']}")
        return
    
    # Find all CSV files
    csv_files = list(CONFIG["raw_folder"].glob("*.csv"))
    print(f"📁 Found {len(csv_files)} CSV files to analyze")
    
    all_analysis = {}
    
    for file_path in csv_files:
        print(f"\n🔍 Analyzing: {file_path.name}")
        
        file_info = get_file_info(file_path)
        file_analysis = {"file_info": file_info}
        
        # Try to read the file
        df, total_rows, error = safe_read_csv(file_path)
        
        if error:
            file_analysis["read_error"] = error
            file_analysis["usability"] = {"score": 0, "usability": "UNREADABLE", "reasons": [f"Read error: {error}"]}
            print(f"   ❌ Read error: {error}")
        else:
            # Perform comprehensive analysis
            file_analysis["columns"] = analyze_columns(df, file_info)
            file_analysis["target_columns"] = find_target_columns(df)
            file_analysis["feature_columns"] = find_feature_columns(df)
            file_analysis["data_quality"] = analyze_data_quality(df, total_rows)
            file_analysis["usability"] = assess_file_usability(file_analysis)
            
            print(f"   ✅ Columns: {len(df.columns)}, Rows: {total_rows}")
            print(f"   🎯 Targets: {len(file_analysis['target_columns'])}")
            print(f"   🔧 Features: {len(file_analysis['feature_columns'])}")
            print(f"   📊 Usability: {file_analysis['usability']['usability']} (Score: {file_analysis['usability']['score']})")
        
        all_analysis[file_path.name] = file_analysis
    
    # Generate summary report
    generate_summary_report(all_analysis)
    
    return all_analysis

def generate_summary_report(all_analysis):
    """Generate comprehensive summary report"""
    
    summary = {
        "analysis_timestamp": datetime.now().isoformat(),
        "total_files_analyzed": len(all_analysis),
        "file_usability_summary": {},
        "all_usable_files": [],
        "best_files_for_training": [],
        "files_with_targets": [],
        "files_with_features": []
    }
    
    # Count usability levels and collect all usable files
    usability_counts = {}
    for filename, analysis in all_analysis.items():
        usability = analysis.get("usability", {}).get("usability", "UNKNOWN")
        usability_counts[usability] = usability_counts.get(usability, 0) + 1
        
        # Track ALL usable files (MEDIUM and HIGH)
        if usability in ["HIGH", "MEDIUM", "LOW"]:
            file_info = {
                "filename": filename,
                "usability": usability,
                "score": analysis["usability"]["score"],
                "targets": len(analysis.get("target_columns", [])),
                "features": len(analysis.get("feature_columns", [])),
                "rows": analysis.get("data_quality", {}).get("total_rows_in_file", 0),
                "columns": analysis.get("columns", {}).get("total_columns", 0)
            }
            summary["all_usable_files"].append(file_info)
        
        # Track files with targets
        if analysis.get("target_columns"):
            summary["files_with_targets"].append({
                "filename": filename,
                "target_columns": [t["column_name"] for t in analysis["target_columns"]],
                "usability": usability
            })
        
        # Track files with features
        if analysis.get("feature_columns"):
            summary["files_with_features"].append({
                "filename": filename,
                "feature_count": len(analysis["feature_columns"]),
                "usability": usability
            })
        
        # Track best files for training (HIGH and MEDIUM)
        if usability in ["HIGH", "MEDIUM"]:
            summary["best_files_for_training"].append({
                "filename": filename,
                "usability": usability,
                "score": analysis["usability"]["score"],
                "targets": len(analysis.get("target_columns", [])),
                "features": len(analysis.get("feature_columns", [])),
                "rows": analysis.get("data_quality", {}).get("total_rows_in_file", 0)
            })
    
    summary["file_usability_summary"] = usability_counts
    
    # Sort files by score
    summary["all_usable_files"].sort(key=lambda x: x["score"], reverse=True)
    summary["best_files_for_training"].sort(key=lambda x: x["score"], reverse=True)
    
    # Save detailed report
    report_path = CONFIG["analysis_folder"] / CONFIG["report_file"]
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump({"detailed_analysis": all_analysis, "summary": summary}, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
    
    # Print comprehensive summary to console
    print("\n" + "=" * 70)
    print("📊 COMPREHENSIVE ANALYSIS SUMMARY")
    print("=" * 70)
    print(f"📁 Total files analyzed: {summary['total_files_analyzed']}")
    print(f"🎯 Files with target columns: {len(summary['files_with_targets'])}")
    print(f"🔧 Files with feature columns: {len(summary['files_with_features'])}")
    print(f"🚀 Best files for training: {len(summary['best_files_for_training'])}")
    print(f"📈 All usable files: {len(summary['all_usable_files'])}")
    
    print(f"\n📊 Usability Distribution:")
    for usability, count in usability_counts.items():
        print(f"   {usability}: {count} files")
    
    print(f"\n🏆 ALL USABLE FILES (Sorted by Score):")
    for i, file_info in enumerate(summary["all_usable_files"], 1):
        print(f"   {i:2d}. {file_info['filename']}")
        print(f"       Score: {file_info['score']}, Usability: {file_info['usability']}")
        print(f"       Rows: {file_info['rows']:,}, Cols: {file_info['columns']}, Targets: {file_info['targets']}, Features: {file_info['features']}")
    
    print(f"\n💾 Detailed report saved: {report_path}")

# === RUN ANALYSIS ===
if __name__ == "__main__":
    print("🚀 Starting comprehensive data analysis...")
    print("💡 This will analyze ALL files without modifying any data")
    
    analysis_results = analyze_all_files()
    
    print("\n" + "=" * 70)
    print("✅ ANALYSIS COMPLETE!")
    print("=" * 70)
    print("📋 Next steps:")
    print("   1. Review ALL usable files above")
    print("   2. We'll use HIGH + MEDIUM + some GOOD LOW files")
    print("   3. Create cleaning script that combines multiple datasets")
    print("   4. Your original data remains untouched!")