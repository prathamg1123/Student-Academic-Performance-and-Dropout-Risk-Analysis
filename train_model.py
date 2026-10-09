# -*- coding: utf-8 -*-
"""
Standalone training script — run once to train and save model artifacts.
Usage: python train_model.py
"""
import sys
import io
from pathlib import Path

# Force UTF-8 output on Windows consoles
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent))

from utils.data_loader import load_raw_data, preprocess, get_feature_matrix
from utils.model_utils import train_and_save, models_exist


def main():
    print("=" * 60)
    print("  Student Dropout Risk — Model Training")
    print("=" * 60)

    print("\n[1/3] Loading dataset...")
    df_raw = load_raw_data()
    print(f"      Loaded {len(df_raw):,} records with {df_raw.shape[1]} columns.")

    print("\n[2/3] Preprocessing...")
    df = preprocess(df_raw)
    X, y, feature_names = get_feature_matrix(df)
    print(f"      Features: {len(feature_names)} | Dropout rate: {y.mean()*100:.1f}%")

    print("\n[3/3] Training RandomForest model (SMOTE + 80/20 split)...")
    metrics = train_and_save(X, y, feature_names)

    print("\n✅  Training complete!")
    print(f"    ROC-AUC   : {metrics['roc_auc']:.4f}")
    print(f"    F1-Score  : {metrics['f1_score']:.4f}")
    print(f"    Avg Prec  : {metrics['average_precision']:.4f}")
    print("\nModel artifacts saved to ./models/")
    print("=" * 60)


if __name__ == "__main__":
    main()
