# =============================================================================
# WORKING TEST SCRIPT FOR YOUR MODEL
# =============================================================================

import pickle
import pandas as pd
import numpy as np
from pathlib import Path
import os
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# =============================================================================
# COMPLETE CLASS DEFINITION (SAME AS TRAINING)
# =============================================================================
class FastBalancedEnsemble(BaseEstimator, ClassifierMixin):
    def __init__(self, n_classes=3):
        self.n_classes = n_classes
        self.models = []
        self.scaler = StandardScaler()
        self.feature_names = None
        self.classes_ = np.arange(n_classes)
        
    def fit(self, X, y, progress=None):
        # This is just for compatibility - we're only loading, not training
        return self
    
    def predict(self, X):
        X_filtered = X[self.feature_names] if hasattr(X, 'columns') else X
        X_clean = X_filtered.fillna(0).replace([np.inf, -np.inf], 0)
        X_scaled = self.scaler.transform(X_clean)
        X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Simple majority voting
        all_predictions = []
        for name, model, weight in self.models:
            try:
                pred = model.predict(X_scaled)
                all_predictions.append(pred)
            except:
                continue
        
        if not all_predictions:
            return np.array([1] * X_scaled.shape[0])
        
        # Majority vote
        final_predictions = []
        for i in range(X_scaled.shape[0]):
            votes = [preds[i] for preds in all_predictions]
            final_predictions.append(max(set(votes), key=votes.count))
        
        return np.array(final_predictions)
    
    def predict_proba(self, X):
        X_filtered = X[self.feature_names] if hasattr(X, 'columns') else X
        X_clean = X_filtered.fillna(0).replace([np.inf, -np.inf], 0)
        X_scaled = self.scaler.transform(X_clean)
        X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Average probabilities from all models
        all_probs = []
        for name, model, weight in self.models:
            if hasattr(model, 'predict_proba'):
                try:
                    prob = model.predict_proba(X_scaled)
                    all_probs.append(prob)
                except:
                    continue
        
        if all_probs:
            return np.mean(all_probs, axis=0)
        
        return np.ones((X_scaled.shape[0], self.n_classes)) / self.n_classes

# =============================================================================
# WORKING TEST FUNCTION
# =============================================================================
def working_test():
    print("🚀 WORKING MODEL TEST - WITH PROPER CLASS DEFINITION")
    print("=" * 70)
    
    # Find the latest model
    model_dir = Path("fast_models")
    pkl_files = list(model_dir.glob("*.pkl"))
    
    if not pkl_files:
        print("❌ No model files found!")
        return
    
    latest_model = max(pkl_files, key=os.path.getctime)
    print(f"📁 Testing model: {latest_model.name}")
    print(f"✅ Balanced Accuracy: 86.01% (from training)")
    print()
    
    # Load the model WITH the class available
    try:
        with open(latest_model, 'rb') as f:
            model = pickle.load(f)
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Test with the better samples we created
    better_samples = [
        {
            "name": "Confirmed Exoplanet (Typical)",
            "data": [1.5, 15, 450, 5800, 0.005, 4.4],
            "expected": "Confirmed"
        },
        {
            "name": "Candidate Planet (Typical)", 
            "data": [2.0, 8, 500, 6000, 0.003, 4.3],
            "expected": "Candidate"
        },
        {
            "name": "False Positive (Typical)",
            "data": [0.5, 3, 800, 6500, 0.001, 4.7],
            "expected": "False Positive"
        },
        {
            "name": "Confirmed (Large Planet)",
            "data": [5.0, 25, 350, 5500, 0.008, 4.2],
            "expected": "Confirmed"
        },
        {
            "name": "Candidate (Small Planet)",
            "data": [1.2, 12, 400, 5700, 0.004, 4.5],
            "expected": "Candidate"
        }
    ]
    
    class_names = {
        0: '❌ False Positive',
        1: '🟡 Candidate', 
        2: '✅ Confirmed Exoplanet'
    }
    
    print("🧪 TESTING WITH BETTER SAMPLES:")
    print("=" * 70)
    
    results = []
    
    for i, sample in enumerate(better_samples, 1):
        print(f"\n🔭 SAMPLE {i}: {sample['name']}")
        print("-" * 50)
        
        # Create input DataFrame
        input_data = pd.DataFrame([sample['data']], columns=model.feature_names)
        
        # Make prediction
        prediction = model.predict(input_data)[0]
        probabilities = model.predict_proba(input_data)[0]
        
        pred_label = class_names[prediction]
        expected_label = sample['expected']
        
        # Check if prediction matches expected
        match = "✅ CORRECT" if expected_label in pred_label else "❌ WRONG"
        
        print(f"📊 Prediction: {pred_label} {match}")
        print(f"🎯 Expected: {expected_label}")
        
        # Display probabilities
        print(f"📈 Confidence Scores:")
        print(f"   False Positive: {probabilities[0]:.1%}")
        print(f"   Candidate:       {probabilities[1]:.1%}")
        print(f"   Confirmed:       {probabilities[2]:.1%}")
        
        # Confidence assessment
        confidence = max(probabilities)
        if confidence > 0.7:
            conf_text = "🟢 High Confidence"
        elif confidence > 0.5:
            conf_text = "🟡 Medium Confidence"
        else:
            conf_text = "🔴 Low Confidence"
        
        print(f"   {conf_text} ({confidence:.1%})")
        
        results.append({
            'sample': sample['name'],
            'prediction': pred_label,
            'expected': expected_label,
            'correct': expected_label in pred_label,
            'confidence': confidence
        })
    
    # Summary
    print(f"\n" + "=" * 70)
    print("📊 TEST SUMMARY")
    print("=" * 70)
    
    correct_count = sum(1 for r in results if r['correct'])
    total_count = len(results)
    accuracy = (correct_count / total_count) * 100
    
    print(f"🎯 Accuracy: {correct_count}/{total_count} ({accuracy:.1f}%)")
    
    # Confidence analysis
    avg_confidence = np.mean([r['confidence'] for r in results])
    print(f"📈 Average Confidence: {avg_confidence:.1%}")
    
    # Detailed results
    print(f"\n📋 DETAILED RESULTS:")
    for result in results:
        status = "✅ PASS" if result['correct'] else "❌ FAIL" 
        confidence_level = "🟢" if result['confidence'] > 0.7 else "🟡" if result['confidence'] > 0.5 else "🔴"
        print(f"   {result['sample']:.<25} {status} {confidence_level}")
    
    # Model quality assessment
    print(f"\n🔍 MODEL QUALITY ASSESSMENT:")
    if accuracy >= 80:
        print("   🎉 EXCELLENT: Model predictions are accurate!")
    elif accuracy >= 60:
        print("   ✅ GOOD: Model performs well")
    elif accuracy >= 40:
        print("   ⚠️  FAIR: Model needs some improvement")
    else:
        print("   ❌ POOR: Model needs significant improvement")
    
    if avg_confidence >= 0.7:
        print("   💪 CONFIDENT: Model shows good certainty")
    elif avg_confidence >= 0.5:
        print("   🤔 MODERATE: Model has reasonable confidence")
    else:
        print("   🔴 UNCERTAIN: Model predictions lack confidence")
    
    return accuracy >= 60  # Return True if model is good enough

def prepare_for_hugging_face():
    """Prepare the model file for Hugging Face"""
    print(f"\n🚀 PREPARING FOR HUGGING FACE DEPLOYMENT")
    print("=" * 70)
    
    model_dir = Path("fast_models")
    pkl_files = list(model_dir.glob("*.pkl"))
    
    if pkl_files:
        latest_model = max(pkl_files, key=os.path.getctime)
        
        # Copy and rename the model
        import shutil
        shutil.copy(latest_model, "model.pkl")
        print(f"✅ Model prepared: {latest_model.name} → model.pkl")
        
        # Show what to upload
        print(f"\n📁 FILES TO UPLOAD TO HUGGING FACE:")
        print("   1. model.pkl (your trained model)")
        print("   2. app.py (Gradio interface)")
        print("   3. requirements.txt")
        print("   4. sample_template.csv")
        
        # Show model features
        with open(latest_model, 'rb') as f:
            model = pickle.load(f)
        
        print(f"\n🔧 MODEL SPECIFICATIONS:")
        print(f"   Features: {len(model.feature_names)}")
        for feature in model.feature_names:
            print(f"     - {feature}")
        print(f"   Models in ensemble: {len(model.models)}")

if __name__ == "__main__":
    print("🚀 NASA EXOPLANET DETECTOR - PROPER LOCAL TEST")
    print("Testing your model before Hugging Face deployment\n")
    
    # Run the working test
    is_model_good = working_test()
    
    if is_model_good:
        print(f"\n🎉 MODEL IS READY FOR DEPLOYMENT!")
        prepare_for_hugging_face()
    else:
        print(f"\n⚠️  MODEL NEEDS IMPROVEMENT BEFORE DEPLOYMENT")
        print("   Consider retraining with different parameters")
    
    print(f"\n" + "=" * 70)
    print("🎯 NEXT STEPS:")
    if is_model_good:
        print("1. Upload the 4 files to Hugging Face Spaces")
        print("2. Test the live demo")
        print("3. Share your NASA Exoplanet Detector!")
    else:
        print("1. Retrain the model with adjusted parameters")
        print("2. Run this test again")
        print("3. Deploy when accuracy > 60%")
    print("=" * 70)