import os
import sys
import json
import pickle
import warnings
from datetime import datetime
from collections import Counter
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE

warnings.filterwarnings('ignore')

class SimpleProgress:
    def __init__(self):
        self.start = datetime.now()
    
    def update(self, pct, status=""):
        step = max(0, min(int(pct), 100))
        filled = "█" * int(step // 10)
        empty = "░" * (10 - int(step // 10))
        bar = filled + empty
        elapsed = (datetime.now() - self.start).total_seconds()
        sys.stdout.write(f"\r[{bar}] {step}% | {status} | {int(elapsed//60)}m{int(elapsed%60)}s")
        sys.stdout.flush()
    
    def done(self):
        elapsed = (datetime.now() - self.start).total_seconds()
        print(f"\n✅ Done in {int(elapsed//60)}m{int(elapsed%60)}s")

class FastBalancedEnsemble(BaseEstimator, ClassifierMixin):
    def __init__(self, n_classes=3, features=None):
        self.n_classes = n_classes
        self.features = features
        self.models = []  # (name, estimator, weight)
        self.scaler = StandardScaler()
        self.feature_names = None
        self.classes_ = None  # Actual class labels set during fitting
    
    def _preprocess(self, X, scale=True):
        # Clean (fill NA, replace inf) and optionally scale input data
        if hasattr(X, 'columns'):  # Handle DataFrame input
            # Check for missing features and handle
            missing_features = [f for f in self.feature_names if f not in X.columns]
            if missing_features:
                if scale:
                    if self.scaler.mean_ is None:
                        raise ValueError("Model not trained: scaler not fit but scaling requested")
                    feature_means = dict(zip(self.feature_names, self.scaler.mean_.tolist()))
                    for f in missing_features:
                        X[f] = X.get(f, feature_means.get(f, 0.0))  # Use mean or 0.0
                else:
                    for f in missing_features:
                        X[f] = 0.0  # Fill with 0 during initial fit preprocessing
            # Clean NA values and infinity
            X_clean = X.fillna(0).replace([np.inf, -np.inf], 0).copy()
            # Ensure only specified features are used (maintain order)
            X_clean = X_clean[self.feature_names].values.astype(np.float32)
        else:  # Handle array-like input (numpy/list)
            X_np = np.array(X, dtype=np.float32)
            X_np = np.nan_to_num(X_np, 0, posinf=0, neginf=0)
            if X_np.ndim == 1:
                X_np = X_np.reshape(1, -1)
            # Adjust feature count (pad/truncate to match required features)
            current_features = X_np.shape[1]
            required_features = len(self.feature_names) if self.feature_names else X_np.shape[1]
            if current_features < required_features:
                pad = required_features - current_features
                X_np = np.hstack([X_np, np.zeros((X_np.shape[0], pad))])
            else:
                X_np = X_np[:, :required_features]
            X_clean = X_np
        
        # Apply scaling if enabled and scaler is fit
        if scale and hasattr(self.scaler, 'mean_'):
            scaled = self.scaler.transform(X_clean)
            return np.nan_to_num(scaled, 0, posinf=0, neginf=0)
        return X_clean
    
    def fit(self, X, y, progress=None):
        if not hasattr(X, 'columns'):
            raise TypeError("Training input X must be a DataFrame")
        
        # Set feature names (use provided or infer from DataFrame)
        self.feature_names = self.features if self.features else list(X.columns)
        missing_features = [f for f in self.feature_names if f not in X.columns]
        if missing_features:
            raise ValueError(f"Missing features in training data: {missing_features}")
        
        # Preprocess without scaling to calculate scaler stats
        X_clean_unscaled = self._preprocess(X, scale=False)
        self.scaler.fit(X_clean_unscaled)
        
        # Preprocess with scaling to get training-ready features
        X_clean_scaled = self._preprocess(X, scale=True)
        
        # Balance data using SMOTE - use smaller sample if dataset is large
        if progress:
            progress.update(20, "Balancing data")
        try:
            # For very large datasets, use a subset for faster training
            if len(X_clean_scaled) > 50000:
                print("📊 Large dataset detected - using subset for faster training")
                X_sample, _, y_sample, _ = train_test_split(
                    X_clean_scaled, y.values, train_size=50000, stratify=y, random_state=42
                )
                smote = SMOTE(random_state=42, k_neighbors=2)
                X_bal, y_bal = smote.fit_resample(X_sample, y_sample)
            else:
                smote = SMOTE(random_state=42, k_neighbors=2)
                X_bal, y_bal = smote.fit_resample(X_clean_scaled, y.values)
            print(f"✅ Balanced data: {len(X_bal)} samples (Original: {len(X_clean_scaled)})")
        except Exception as e:
            print(f"⚠️ SMOTE failed. Using original data. Error: {e}")
            X_bal, y_bal = X_clean_scaled, y.values
        
        # Validate class presence after balancing
        class_counts = Counter(y_bal)
        if len(class_counts) < self.n_classes:
            raise ValueError(f"Balanced data missing {self.n_classes - len(class_counts)} required classes")
        self.classes_ = np.sort(np.unique(y_bal))  # Store actual class labels
        
        # Define models and hyperparameter grids - EXTREMELY SIMPLIFIED FOR SPEED
        models = [
            ('RF_Fast', RandomForestClassifier(
                class_weight='balanced', 
                random_state=42, 
                n_jobs=-1,
                n_estimators=50,  # Fixed small number for speed
                max_depth=10
            ), {
                'n_estimators': [50, 100],  # Very limited options
                'max_depth': [5, 10, 15],
                'min_samples_split': [2, 5],
                'max_features': ['sqrt', 'log2']
            }),
            ('XGB_Fast', XGBClassifier(
                objective='multi:softprob', 
                eval_metric='mlogloss', 
                random_state=44, 
                n_jobs=-1, 
                verbosity=0
            ), {
                'n_estimators': [50, 100],
                'max_depth': [3, 5],
                'learning_rate': [0.1, 0.2],
                'subsample': [0.8],
                'colsample_bytree': [0.8]
            }),
            ('LightGBM_Fast', LGBMClassifier(
                random_state=45, 
                n_jobs=-1, 
                verbose=-1
            ), {
                'n_estimators': [50, 100],
                'max_depth': [3, 5],
                'learning_rate': [0.1, 0.2],
                'num_leaves': [15, 31]
            }),
            ('LogReg_Fast', LogisticRegression(
                class_weight='balanced', 
                multi_class='multinomial', 
                solver='saga', 
                max_iter=1000, 
                random_state=46, 
                n_jobs=-1
            ), {
                'C': [0.1, 1, 10],
                'penalty': ['l2']  # Only L2 for speed
            })
        ]
        
        trained_models = []
        total_perf = 0.0
        
        # Train each model with VERY FAST hyperparameter search
        for i, (name, cls, params) in enumerate(models):
            if progress:
                progress.update(40 + (i + 1) * 15, f"Training {name}")
            try:
                search = RandomizedSearchCV(
                    estimator=cls,
                    param_distributions=params,
                    n_iter=10,  # Very few iterations
                    cv=2,       # Only 2-fold CV
                    scoring='balanced_accuracy',
                    random_state=42,
                    n_jobs=-1,
                    verbose=0   # No verbose output
                )
                search.fit(X_bal, y_bal)
                best_model = search.best_estimator_
                if not hasattr(best_model, 'predict_proba'):
                    print(f"❌ Skipping {name}: no predict_proba method")
                    continue
                score = search.best_score_
                total_perf += score
                trained_models.append((name, best_model, score))
                print(f"✅ {name} trained - score: {score:.4f}")
            except Exception as e:
                print(f"❌ Training {name} failed: {e}")
                continue
        
        # Handle case where no models trained
        if not trained_models:
            raise RuntimeError("No valid models trained for ensemble")
        
        # Calculate model weights based on performance
        if total_perf == 0:
            weights = [1/len(trained_models)] * len(trained_models)
        else:
            weights = [score / total_perf for (_, _, score) in trained_models]
        
        # Store models with weights - FIXED: don't unpack into separate lists
        self.models = [(name, model, weight) for (name, model, _), weight in zip(trained_models, weights)]
        
        if progress:
            progress.done()
        return self
    
    def predict(self, X):
        X_scaled = self._preprocess(X)
        if not self.models:
            return np.full(len(X_scaled), self.classes_[0], dtype=int) if self.classes_ is not None else np.array([])
        probs, weights = [], []
        for _, model, weight in self.models:
            try:
                prob = model.predict_proba(X_scaled)
                probs.append(prob * weight)
                weights.append(weight)
            except Exception as e:
                print(f"⚠️ Prediction error with {model}: {e}")
        if not probs:
            return np.full(len(X_scaled), self.classes_[0], dtype=int)
        summed = np.sum(probs, axis=0)
        total_weight = sum(weights) or 1.0
        normalized = summed / total_weight
        pred_indices = np.argmax(normalized, axis=1)
        return self.classes_[pred_indices]
    
    def predict_proba(self, X):
        X_scaled = self._preprocess(X)
        if not self.models or X_scaled is None:
            return np.ones((len(X_scaled), self.n_classes)) / self.n_classes if self.n_classes else np.array([])
        probs, weights = [], []
        for _, model, weight in self.models:
            try:
                prob = model.predict_proba(X_scaled)
                probs.append(prob * weight)
                weights.append(weight)
            except Exception as e:
                print(f"⚠️ Proba error with {model}: {e}")
        if not probs:
            return np.ones((len(X_scaled), self.n_classes)) / self.n_classes
        summed = np.sum(probs, axis=0)
        total_weight = sum(weights) or 1.0
        return summed / total_weight

def train_model(data_path="cleaned_exoplanet_data.csv"):
    output_dir = Path("models")
    output_dir.mkdir(exist_ok=True)
    progress = SimpleProgress()
    essential_features = [
        'planet_radius', 'orbital_period', 'equilibrium_temp',
        'transit_duration', 'stellar_temp', 'transit_depth', 'stellar_gravity'
    ]
    
    try:
        # Data loading - only read necessary columns
        progress.update(5, "Loading dataset")
        df = pd.read_csv(Path(data_path), usecols=essential_features + ['target_class'])
        if 'target_class' not in df.columns:
            raise ValueError("Dataset missing required 'target_class' column")
        y = df.pop('target_class')
        if y.nunique() < 3:
            raise ValueError(f"Dataset has only {y.nunique()} classes (needs at least 3)")
        X = df[essential_features].copy()
        
        # Train-test split
        progress.update(10, "Splitting data")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=0.2, 
            stratify=y, 
            random_state=42
        )
        
        # Train ensemble model
        progress.update(15, "Starting ensemble training")
        ensemble = FastBalancedEnsemble(n_classes=3, features=essential_features)
        ensemble.fit(X_train, y_train, progress)
        
        # Evaluation
        progress.update(90, "Evaluating performance")
        y_pred = ensemble.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        bal_acc = balanced_accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred).tolist()
        
        # Print metrics
        print(f"\n✅ Evaluation Results:")
        print(f"   Accuracy: {acc:.4f}")
        print(f"   Balanced Accuracy: {bal_acc:.4f}")
        
        # Save model and metadata
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = output_dir / f"model_{timestamp}.pkl"
        meta_path = output_dir / f"meta_{timestamp}.json"
        
        with open(model_path, 'wb') as f:
            pickle.dump(ensemble, f)
        print(f"💾 Model saved: {model_path}")
        
        with open(meta_path, 'w') as f:
            json.dump({
                'version': 'v4_fast',
                'timestamp': timestamp,
                'features': essential_features,
                'metrics': {
                    'accuracy': float(acc),
                    'balanced_accuracy': float(bal_acc),
                    'confusion_matrix': cm
                },
                'scaler': {
                    'mean': ensemble.scaler.mean_.tolist(),
                    'scale': ensemble.scaler.scale_.tolist()
                },
                'models': [
                    {'name': name, 'params': model.get_params(), 'weight': weight}
                    for name, model, weight in ensemble.models
                ]
            }, f, indent=2)
        print(f"💾 Metadata saved: {meta_path}")
        
        progress.done()
        return ensemble, acc, bal_acc
    except Exception as e:
        print(f"❌ Training failed: {str(e)}")
        return None, 0.0, 0.0

if __name__ == "__main__":
    print("\n🚀 Starting FAST Exoplanet Classifier Training...")
    model, acc, bal_acc = train_model("cleaned_exoplanet_data.csv")
    if model:
        print(f"\n🎉 Training Complete! Results:")
        print(f"   Final Accuracy: {acc:.4f}")
        print(f"   Final Balanced Accuracy: {bal_acc:.4f}")
    else:
        print("\n❌ Training process encountered errors. Check data and dependencies.")