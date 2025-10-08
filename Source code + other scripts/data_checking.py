import os
import pandas as pd
import numpy as np
import json
import warnings
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import gc

warnings.filterwarnings('ignore')

class UltimateExoplanetAnalyzer:
    def __init__(self, data_directory):
        self.data_directory = Path(data_directory)
        self.data_files = []
        self.file_info = {}
        self.summary_stats = {}
        self.analysis_results = {}
        self.useful_files = []
        
    def discover_data_files(self):
        """Comprehensive file discovery with multiple extensions"""
        print("🔍 Discovering data files...")
        
        supported_extensions = ['.csv', '.xlsx', '.xls', '.json', '.txt', '.dat', '.tsv', '.gz']
        
        for file_path in self.data_directory.rglob('*'):
            if file_path.suffix.lower() in supported_extensions and file_path.is_file():
                file_info = {
                    'path': file_path,
                    'size_mb': file_path.stat().st_size / (1024 * 1024),
                    'extension': file_path.suffix.lower(),
                    'modified': datetime.fromtimestamp(file_path.stat().st_mtime)
                }
                self.data_files.append(file_info)
                
        print(f"📁 Found {len(self.data_files)} data files")
        return self.data_files
    
    def detect_file_type(self, file_path):
        """Enhanced file type detection"""
        suffixes = file_path.suffixes
        if '.csv' in suffixes or '.gz' in suffixes:
            return 'csv'
        elif '.xlsx' in suffixes or '.xls' in suffixes:
            return 'excel'
        elif '.json' in suffixes:
            return 'json'
        elif '.txt' in suffixes or '.tsv' in suffixes or '.dat' in suffixes:
            return 'text'
        return 'unknown'
    
    def safe_read_file(self, file_info):
        """Ultimate file reading with comprehensive error handling"""
        file_path = file_info['path']
        file_type = self.detect_file_type(file_path)
        
        try:
            if file_type == 'csv':
                # Enhanced CSV reading with multiple strategies
                encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252', 'windows-1252']
                separators = [',', ';', '\t', '|', ' ']
                
                for encoding in encodings:
                    for sep in separators:
                        try:
                            if file_path.suffix == '.gz':
                                df = pd.read_csv(file_path, encoding=encoding, sep=sep, 
                                               low_memory=False, compression='gzip')
                            else:
                                df = pd.read_csv(file_path, encoding=encoding, sep=sep, 
                                               low_memory=False)
                            if df.shape[1] > 1:  # Valid CSV should have multiple columns
                                print(f"   ✅ Success with encoding: {encoding}, separator: '{sep}'")
                                return df, 'csv'
                        except Exception as e:
                            continue
                            
            elif file_type == 'excel':
                # Try different Excel engines
                for engine in ['openpyxl', 'xlrd']:
                    try:
                        df = pd.read_excel(file_path, engine=engine)
                        print(f"   ✅ Success with engine: {engine}")
                        return df, 'excel'
                    except:
                        continue
                        
            elif file_type == 'json':
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Handle different JSON structures
                if isinstance(data, list):
                    df = pd.DataFrame(data)
                elif isinstance(data, dict):
                    if 'values' in data or 'data' in data or 'results' in data:
                        # Common JSON data structures
                        key_to_use = next((k for k in ['values', 'data', 'results'] if k in data), None)
                        if key_to_use and isinstance(data[key_to_use], list):
                            df = pd.DataFrame(data[key_to_use])
                        else:
                            df = pd.json_normalize(data)
                    else:
                        df = pd.json_normalize(data)
                else:
                    return None, 'json_invalid'
                return df, 'json'
                
            elif file_type == 'text':
                # Advanced text file parsing
                separators = ['\t', ',', ';', '|', ' ']
                for sep in separators:
                    try:
                        df = pd.read_csv(file_path, sep=sep, engine='python', 
                                       low_memory=False, encoding='utf-8')
                        if df.shape[1] > 1:
                            print(f"   ✅ Success with separator: '{sep}'")
                            return df, 'text'
                    except:
                        continue
                        
        except Exception as e:
            print(f"   ❌ Error reading {file_path.name}: {str(e)}")
            return None, f'error_{file_type}'
        
        return None, 'unreadable'
    
    def analyze_dataframe(self, df, filename, file_info):
        """Comprehensive dataframe analysis with enhanced metrics"""
        analysis = {
            'filename': filename,
            'file_info': file_info,
            'shape': df.shape,
            'total_cells': df.shape[0] * df.shape[1],
            'columns': list(df.columns),
            'dtypes': df.dtypes.to_dict(),
            'missing_values': df.isnull().sum().to_dict(),
            'missing_percentage': (df.isnull().sum() / len(df) * 100).to_dict(),
            'numeric_columns': df.select_dtypes(include=[np.number]).columns.tolist(),
            'categorical_columns': df.select_dtypes(include=['object', 'category']).columns.tolist(),
            'datetime_columns': df.select_dtypes(include=['datetime64']).columns.tolist(),
            'unique_values': {col: df[col].nunique() for col in df.columns},
            'memory_usage_mb': df.memory_usage(deep=True).sum() / (1024**2),
            'duplicate_rows': df.duplicated().sum(),
            'duplicate_percentage': (df.duplicated().sum() / len(df)) * 100,
        }
        
        # Enhanced target column identification
        target_keywords = [
            'disposition', 'confirmed', 'candidate', 'planet', 'classification', 
            'label', 'class', 'target', 'koi_disposition', 'tfopwg_disp'
        ]
        analysis['potential_target_columns'] = [
            col for col in df.columns 
            if any(keyword in str(col).lower() for keyword in target_keywords)
        ]
        
        # Enhanced ID column identification
        id_keywords = ['id', 'name', 'kepid', 'rowid', 'index', '_id', 'koi_name']
        analysis['id_columns'] = [
            col for col in df.columns 
            if any(keyword in str(col).lower() for keyword in id_keywords)
        ]
        
        # Exoplanet-specific feature identification
        exoplanet_features = {
            'orbital_features': ['period', 'duration', 'epoch', 'semi', 'axis', 'inclination'],
            'planetary_features': ['radius', 'mass', 'density', 'temperature', 'teq'],
            'stellar_features': ['teff', 'logg', 'feh', 'mass', 'radius', 'mag', 'lum'],
            'transit_features': ['depth', 'impact', 'ingress', 'egress', 'transit']
        }
        
        analysis['exoplanet_columns'] = {}
        for feature_type, keywords in exoplanet_features.items():
            analysis['exoplanet_columns'][feature_type] = [
                col for col in df.columns 
                if any(keyword in str(col).lower() for keyword in keywords)
            ]
        
        # Statistical analysis for numeric columns
        if analysis['numeric_columns']:
            analysis['numeric_stats'] = df[analysis['numeric_columns']].describe().to_dict()
            # Detect outliers using IQR
            numeric_df = df[analysis['numeric_columns']].select_dtypes(include=[np.number])
            Q1 = numeric_df.quantile(0.25)
            Q3 = numeric_df.quantile(0.75)
            IQR = Q3 - Q1
            analysis['outlier_columns'] = ((numeric_df < (Q1 - 1.5 * IQR)) | 
                                         (numeric_df > (Q3 + 1.5 * IQR))).sum().to_dict()
        
        # Data quality assessment at different sections
        analysis['data_quality_sections'] = {
            'start': self.assess_data_quality_section(df, 0, 100),
            'middle': self.assess_data_quality_section(df, len(df)//2 - 50, 100),
            'end': self.assess_data_quality_section(df, max(0, len(df)-100), 100)
        }
        
        # Constant and high-missing columns
        analysis['constant_columns'] = [col for col in df.columns if df[col].nunique() <= 1]
        analysis['high_missing_columns'] = [
            col for col, pct in analysis['missing_percentage'].items() 
            if pct > 50
        ]
        analysis['moderate_missing_columns'] = [
            col for col, pct in analysis['missing_percentage'].items() 
            if 10 < pct <= 50
        ]
        
        # High cardinality categorical columns
        analysis['high_cardinality_categorical'] = [
            col for col in analysis['categorical_columns'] 
            if analysis['unique_values'][col] > 100
        ]
        
        # Calculate comprehensive usefulness score
        analysis['usefulness_score'] = self.calculate_comprehensive_usefulness_score(df, analysis)
        analysis['data_quality_score'] = self.calculate_data_quality_score(analysis)
        
        return analysis
    
    def assess_data_quality_section(self, df, start_idx, num_rows):
        """Assess data quality in a specific section"""
        end_idx = min(start_idx + num_rows, len(df))
        section = df.iloc[start_idx:end_idx]
        
        return {
            'rows_analyzed': len(section),
            'missing_percentage': (section.isnull().sum().sum() / (len(section) * len(section.columns))) * 100,
            'constant_columns': [col for col in section.columns if section[col].nunique() <= 1],
            'data_types': section.dtypes.to_dict()
        }
    
    def calculate_comprehensive_usefulness_score(self, df, analysis):
        """Enhanced usefulness scoring system"""
        score = 100
        
        # Size penalties/bonuses
        if df.shape[0] < 100:
            score -= 25
        elif df.shape[0] > 10000:
            score += 10
            
        if df.shape[1] < 3:
            score -= 20
        elif df.shape[1] > 20:
            score += 5
            
        # Data quality penalties
        if len(analysis['constant_columns']) > df.shape[1] * 0.2:
            score -= 15
            
        if len(analysis['high_missing_columns']) > df.shape[1] * 0.3:
            score -= 20
            
        if analysis['duplicate_percentage'] > 10:
            score -= 10
            
        # Feature quality bonuses
        if len(analysis['potential_target_columns']) > 0:
            score += 15
            
        if len(analysis['numeric_columns']) > 5:
            score += 10
            
        # Exoplanet-specific bonuses
        exoplanet_col_count = sum(len(cols) for cols in analysis['exoplanet_columns'].values())
        if exoplanet_col_count > 5:
            score += 15
        elif exoplanet_col_count > 0:
            score += 5
            
        return max(0, min(100, score))
    
    def calculate_data_quality_score(self, analysis):
        """Calculate data quality score (0-100)"""
        quality_score = 100
        
        # Missing data penalty
        total_missing_pct = sum(analysis['missing_values'].values()) / analysis['total_cells'] * 100
        quality_score -= min(50, total_missing_pct * 0.5)
        
        # Constant columns penalty
        constant_penalty = len(analysis['constant_columns']) / len(analysis['columns']) * 30
        quality_score -= constant_penalty
        
        # Duplicate rows penalty
        duplicate_penalty = min(20, analysis['duplicate_percentage'])
        quality_score -= duplicate_penalty
        
        return max(0, min(100, quality_score))
    
    def generate_intelligent_recommendations(self, analysis):
        """Generate intelligent, actionable recommendations"""
        recommendations = []
        
        # Data cleaning priorities
        if analysis['duplicate_rows'] > 0:
            recommendations.append({
                'priority': 'HIGH',
                'action': 'REMOVE_DUPLICATES',
                'description': f"Remove {analysis['duplicate_rows']} duplicate rows ({analysis['duplicate_percentage']:.1f}%)",
                'code': "df = df.drop_duplicates()"
            })
        
        if analysis['id_columns']:
            recommendations.append({
                'priority': 'HIGH',
                'action': 'REMOVE_ID_COLUMNS',
                'description': f"Remove identifier columns: {analysis['id_columns']}",
                'code': f"df = df.drop(columns={analysis['id_columns']})"
            })
        
        if analysis['constant_columns']:
            recommendations.append({
                'priority': 'HIGH',
                'action': 'REMOVE_CONSTANT_COLUMNS',
                'description': f"Remove constant columns with no variance: {analysis['constant_columns']}",
                'code': f"df = df.drop(columns={analysis['constant_columns']})"
            })
        
        if analysis['high_missing_columns']:
            recommendations.append({
                'priority': 'MEDIUM',
                'action': 'REMOVE_HIGH_MISSING',
                'description': f"Consider removing columns with >50% missing: {analysis['high_missing_columns']}",
                'code': f"df = df.drop(columns={analysis['high_missing_columns']})"
            })
        
        # Feature engineering suggestions
        if any('period' in col.lower() for col in analysis['columns']):
            recommendations.append({
                'priority': 'MEDIUM',
                'action': 'CREATE_ORBITAL_FREQUENCY',
                'description': "Create orbital frequency feature from period (1/period)",
                'code': "df['orbital_frequency'] = 1 / df['orbital_period']"
            })
        
        if any('radius' in col.lower() for col in analysis['columns']):
            recommendations.append({
                'priority': 'MEDIUM',
                'action': 'CREATE_RADIUS_RATIO',
                'description': "Create planet-to-star radius ratio if both available",
                'code': "df['radius_ratio'] = df['planet_radius'] / df['stellar_radius']"
            })
        
        # Data type optimization
        if analysis['high_cardinality_categorical']:
            recommendations.append({
                'priority': 'LOW',
                'action': 'REDUCE_CARDINALITY',
                'description': f"Consider grouping rare categories in: {analysis['high_cardinality_categorical']}",
                'code': "# Use value_counts() to identify rare categories and group them"
            })
        
        return recommendations
    
    def suggest_ml_pipeline(self, analysis):
        """Suggest ML pipeline based on data characteristics"""
        pipeline = []
        
        # Based on target type
        if analysis['potential_target_columns']:
            for target_col in analysis['potential_target_columns']:
                if analysis['unique_values'][target_col] <= 10:  # Classification
                    pipeline.append(f"Classification task using '{target_col}' as target")
                    pipeline.append("Algorithms: Random Forest, XGBoost, Neural Networks")
                    pipeline.append("Metrics: Accuracy, Precision, Recall, F1-Score")
                else:  # Regression
                    pipeline.append(f"Regression task using '{target_col}' as target")
                    pipeline.append("Algorithms: Random Forest, XGBoost, Linear Regression")
                    pipeline.append("Metrics: RMSE, MAE, R²")
        
        # Based on data size
        if analysis['shape'][0] > 10000:
            pipeline.append("Large dataset - consider using GPU acceleration")
        else:
            pipeline.append("Moderate dataset - standard algorithms should work well")
        
        # Based on feature types
        if len(analysis['categorical_columns']) > len(analysis['numeric_columns']):
            pipeline.append("Categorical-heavy: Use OneHotEncoding or TargetEncoding")
        else:
            pipeline.append("Numeric-heavy: Standard scaling recommended")
        
        return pipeline
    
    def analyze_all_files(self):
        """Main analysis function with enhanced reporting"""
        print("🚀 Starting Ultimate Exoplanet Data Analysis...")
        print("=" * 100)
        
        self.discover_data_files()
        
        if not self.data_files:
            print("❌ No data files found!")
            return
        
        for file_info in self.data_files:
            print(f"\n{'='*80}")
            print(f"📊 Analyzing: {file_info['path'].name}")
            print(f"📏 Size: {file_info['size_mb']:.2f} MB | Modified: {file_info['modified'].strftime('%Y-%m-%d')}")
            
            df, file_format = self.safe_read_file(file_info)
            
            if df is None:
                print(f"❌ Could not read file (format: {file_format})")
                continue
            
            analysis = self.analyze_dataframe(df, file_info['path'].name, file_info)
            self.analysis_results[file_info['path'].name] = analysis
            
            # Enhanced reporting
            print(f"✅ Successfully loaded ({file_format})")
            print(f"📐 Shape: {analysis['shape'][0]:,} rows × {analysis['shape'][1]} columns")
            print(f"🎯 Usefulness Score: {analysis['usefulness_score']}/100")
            print(f"🧹 Data Quality Score: {analysis['data_quality_score']}/100")
            print(f"📊 Memory: {analysis['memory_usage_mb']:.2f} MB")
            print(f"🔍 Missing Data: {sum(analysis['missing_values'].values()):,} cells ({sum(analysis['missing_values'].values())/analysis['total_cells']*100:.1f}%)")
            print(f"🔁 Duplicates: {analysis['duplicate_rows']} rows ({analysis['duplicate_percentage']:.1f}%)")
            
            # Key findings
            if analysis['potential_target_columns']:
                print(f"🎯 Potential Targets: {analysis['potential_target_columns']}")
            
            exoplanet_col_count = sum(len(cols) for cols in analysis['exoplanet_columns'].values())
            if exoplanet_col_count > 0:
                print(f"🌟 Exoplanet Features: {exoplanet_col_count} columns")
                for feature_type, cols in analysis['exoplanet_columns'].items():
                    if cols:
                        print(f"   • {feature_type}: {len(cols)} columns")
            
            # Quick recommendations
            recommendations = self.generate_intelligent_recommendations(analysis)
            high_priority_recs = [r for r in recommendations if r['priority'] == 'HIGH']
            if high_priority_recs:
                print(f"🚨 High Priority Actions: {len(high_priority_recs)}")
            
            if analysis['usefulness_score'] >= 70:
                self.useful_files.append(file_info['path'].name)
                print("✨ EXCELLENT CANDIDATE FOR ML TRAINING")
            elif analysis['usefulness_score'] >= 50:
                self.useful_files.append(file_info['path'].name)
                print("⚠️  USABLE WITH CLEANING")
            
            # Clean up memory
            del df
            gc.collect()
        
        self.generate_comprehensive_report()
    
    def generate_comprehensive_report(self):
        """Generate ultimate comprehensive report"""
        print(f"\n{'='*100}")
        print("🎯 ULTIMATE EXOPLANET DATA ANALYSIS REPORT")
        print(f"{'='*100}")
        
        if not self.useful_files:
            print("⚠️  No files met the usefulness threshold (score ≥ 50)")
            return
        
        print(f"\n📈 USEFUL FILES FOR MACHINE LEARNING:")
        print(f"{'-'*100}")
        
        # Sort files by usefulness score
        useful_analyses = [(f, self.analysis_results[f]) for f in self.useful_files]
        useful_analyses.sort(key=lambda x: x[1]['usefulness_score'], reverse=True)
        
        for filename, analysis in useful_analyses:
            print(f"\n📁 {filename}")
            print(f"   Overall Score: {analysis['usefulness_score']}/100 | Quality: {analysis['data_quality_score']}/100")
            print(f"   Dimensions: {analysis['shape'][0]:,} rows × {analysis['shape'][1]} columns")
            print(f"   Memory: {analysis['memory_usage_mb']:.2f} MB")
            
            # Target analysis
            if analysis['potential_target_columns']:
                print(f"\n   🎯 TARGET ANALYSIS:")
                for target_col in analysis['potential_target_columns']:
                    unique_vals = analysis['unique_values'][target_col]
                    missing_pct = analysis['missing_percentage'][target_col]
                    target_type = "CLASSIFICATION" if unique_vals <= 10 else "REGRESSION"
                    print(f"      • {target_col} → {target_type} ({unique_vals} unique, {missing_pct:.1f}% missing)")
            
            # Feature breakdown
            print(f"\n   📊 FEATURE BREAKDOWN:")
            print(f"      Numeric: {len(analysis['numeric_columns'])} columns")
            print(f"      Categorical: {len(analysis['categorical_columns'])} columns")
            print(f"      Exoplanet-specific: {sum(len(cols) for cols in analysis['exoplanet_columns'].values())} columns")
            
            # Data issues
            issues = []
            if analysis['duplicate_rows'] > 0:
                issues.append(f"Duplicates: {analysis['duplicate_rows']} rows")
            if analysis['constant_columns']:
                issues.append(f"Constant columns: {len(analysis['constant_columns'])}")
            if analysis['high_missing_columns']:
                issues.append(f"High-missing: {len(analysis['high_missing_columns'])}")
            
            if issues:
                print(f"\n   🚨 DATA ISSUES: {', '.join(issues)}")
            
            # Actionable recommendations
            recommendations = self.generate_intelligent_recommendations(analysis)
            print(f"\n   🛠️  CLEANING ACTIONS:")
            for rec in recommendations[:5]:  # Show top 5
                print(f"      [{rec['priority']}] {rec['description']}")
            
            # ML pipeline suggestion
            ml_pipeline = self.suggest_ml_pipeline(analysis)
            print(f"\n   🤖 SUGGESTED ML APPROACH:")
            for step in ml_pipeline[:3]:
                print(f"      • {step}")
            
            print(f"   {'─'*60}")
        
        # Overall strategy
        print(f"\n🎯 OVERALL STRATEGY RECOMMENDATIONS:")
        print(f"{'-'*100}")
        print("1. START WITH: Files with highest usefulness scores (>80)")
        print("2. DATA CLEANING: Remove duplicates, IDs, constant columns first")
        print("3. FEATURE ENGINEERING: Create orbital features, ratios, interactions")
        print("4. MODEL SELECTION: Random Forest (baseline), XGBoost (performance), Neural Networks (complex patterns)")
        print("5. VALIDATION: Use 10-fold cross-validation with stratified splits")
        print("6. ENSEMBLE: Combine multiple models for better performance")
        print("7. INTERPRETABILITY: Use SHAP values to understand feature importance")
        
        # File prioritization
        print(f"\n📋 FILE PRIORITIZATION:")
        high_priority = [f for f in self.useful_files if self.analysis_results[f]['usefulness_score'] >= 80]
        medium_priority = [f for f in self.useful_files if 60 <= self.analysis_results[f]['usefulness_score'] < 80]
        
        if high_priority:
            print("   🥇 HIGH PRIORITY (Score ≥ 80):")
            for f in high_priority:
                print(f"      • {f}")
        
        if medium_priority:
            print("   🥈 MEDIUM PRIORITY (Score 60-79):")
            for f in medium_priority:
                print(f"      • {f}")
    
    def save_detailed_analysis(self, output_dir=None):
        """Save comprehensive analysis to files"""
        if output_dir is None:
            output_dir = self.data_directory.parent / "analysis_results"
        
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # Save JSON report
        json_path = output_dir / "detailed_analysis.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            # Convert non-serializable objects
            serializable_results = {}
            for filename, analysis in self.analysis_results.items():
                serializable_analysis = {}
                for key, value in analysis.items():
                    try:
                        json.dumps(value)  # Test serialization
                        serializable_analysis[key] = value
                    except:
                        serializable_analysis[key] = str(value)
                serializable_results[filename] = serializable_analysis
            
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)
        
        # Save summary report
        summary_path = output_dir / "analysis_summary.txt"
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("NASA EXOPLANET AI - COMPREHENSIVE DATA ANALYSIS SUMMARY\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"Analysis performed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total files analyzed: {len(self.data_files)}\n")
            f.write(f"Useful files identified: {len(self.useful_files)}\n\n")
            
            for filename in self.useful_files:
                analysis = self.analysis_results[filename]
                f.write(f"FILE: {filename}\n")
                f.write(f"  Score: {analysis['usefulness_score']}/100\n")
                f.write(f"  Shape: {analysis['shape']}\n")
                f.write(f"  Targets: {analysis['potential_target_columns']}\n")
                f.write(f"  Key Issues: {len(analysis['constant_columns'])} constant, {len(analysis['high_missing_columns'])} high-missing\n")
                f.write("\n")
        
        print(f"\n💾 Analysis saved to:")
        print(f"   📄 Detailed JSON: {json_path}")
        print(f"   📋 Summary: {summary_path}")

# Run the ultimate analysis
if __name__ == "__main__":
    data_directory = r"C:\Users\RR Joshi Computer\Desktop\Nasa_exoplanet_ai\data\RAW"
    
    print("🚀 NASA EXOPLANET AI - ULTIMATE DATA ANALYZER")
    print("⭐ Combining best features from both scripts with enhancements")
    print("=" * 100)
    
    analyzer = UltimateExoplanetAnalyzer(data_directory)
    analyzer.analyze_all_files()
    analyzer.save_detailed_analysis()
    
    print(f"\n🎉 ULTIMATE ANALYSIS COMPLETE!")
    print(f"📊 Useful files found: {len(analyzer.useful_files)}/{len(analyzer.data_files)}")
    print(f"🎯 Next: Use the recommendations to build your ML pipeline!")