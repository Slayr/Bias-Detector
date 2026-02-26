"""
evaluate.py — Multi-tier Evaluation for Bias Detector
======================================================
Tests the DeBERTa model saved by finetune.py across 3 tiers:
  1. In-distribution (MBIC test split)
  2. Cross-dataset (BABE and MBIB complete dataset test)
  3. Hand-curated hard cases
"""
import os, sys
from pathlib import Path

import pandas as pd
import torch
from sklearn.metrics import classification_report, roc_auc_score, f1_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT_DIR   = Path(__file__).parent
DATA_DIR   = ROOT_DIR / "data"
MODEL_DIR  = ROOT_DIR / "models" / "finetuned"
MBIC_EXCEL = DATA_DIR / os.getenv("MBIC_CSV", "mbic.xlsx")
BABE_CSV   = DATA_DIR / "babe.csv"
MBIB_CSV   = DATA_DIR / "mbib_bias.csv"
HARD_CSV   = DATA_DIR / "hard_cases.csv"

def _load_model_and_tokenizer():
    if not MODEL_DIR.exists():
        print(f"[error] Model not found at {MODEL_DIR}. Run finetune.py first.")
        sys.exit(1)
        
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading model from {MODEL_DIR} to {device}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    model.to(device)
    model.eval()
    return model, tokenizer, device

def _predict_batch(texts, labels, model, tokenizer, device, batch_size=32):
    y_true = []
    y_pred = []
    y_prob = []
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        batch_labels = labels[i:i+batch_size]
        
        inputs = tokenizer(batch_texts, padding=True, truncation=True, max_length=256, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            # outputs.logits shape: [batch_size, 3] -> [Bias, Partisan, Opinion]
            # BCEWithLogitsLoss means outputs are unbound logits, we must apply sigmoid to get probabilities
            probs_matrix = torch.sigmoid(outputs.logits).cpu().numpy()
            
        if len(batch_texts) == 1:
            probs_matrix = [probs_matrix.flatten()]
            
        # Extract the primary "Bias" probability (index 0)
        primary_probs = [p[0] for p in probs_matrix]
        preds = [1 if p >= 0.5 else 0 for p in primary_probs]
        
        y_prob.extend(primary_probs)
        y_pred.extend(preds)
        
        # We only evaluate the primary task ("Is Biased?") for the core metrics right now
        # Labels are expected to be the float "soft_label" of the Bias target.
        y_true.extend([1 if l >= 0.5 else 0 for l in batch_labels])
        
    return y_true, y_pred, y_prob

def run_evaluation(tier_name, texts, labels, model, tokenizer, device):
    if not texts:
        print(f"  Tier: {tier_name} - No data available.")
        return
        
    print(f"\n--- Tier: {tier_name} ({len(texts)} samples) ---")
    y_true, y_pred, y_prob = _predict_batch(texts, labels, model, tokenizer, device)
    
    auc = roc_auc_score(y_true, y_prob)
    f1 = f1_score(y_true, y_pred)
    
    print(f"AUC: {auc:.4f} | F1: {f1:.4f}")
    if len(texts) < 50: # For hard cases, print everything out
        print("\nDetailed predictions:")
        for t, true_lbl, pred_lbl, prob in zip(texts, y_true, y_pred, y_prob):
            mark = "✅" if true_lbl == pred_lbl else "❌"
            print(f"  {mark} [True: {true_lbl}, Pred: {pred_lbl}, Prob: {prob:.2f}] {t[:80]}...")
    else:
        print(classification_report(y_true, y_pred, target_names=["Non-biased", "Biased"]))

def main():
    model, tokenizer, device = _load_model_and_tokenizer()
    
    # Tier 1: In Distribution
    # We load MBIC as an approximation of the training distribution
    if MBIC_EXCEL.exists():
        mbic = pd.read_excel(MBIC_EXCEL, engine="openpyxl")
        # Rename commonly used keys
        col_map = {c: c.strip().lower() for c in mbic.columns}
        mbic = mbic.rename(columns=col_map)
        if "sentence" in mbic.columns and "label_bias" in mbic.columns:
            mbic = mbic[mbic["label_bias"].isin(["Biased", "Non-biased"])].copy()
            mbic["soft_label"] = mbic["label_bias"].apply(lambda x: 1.0 if x == "Biased" else 0.0)
            
            # Subsample to a reasonable eval set
            mbic = mbic.sample(n=min(2000, len(mbic)), random_state=42)
            run_evaluation("In-Distribution (MBIC)", mbic["sentence"].tolist(), mbic["soft_label"].tolist(), model, tokenizer, device)

    # Tier 2: Cross Dataset (BABE)
    if BABE_CSV.exists():
        babe = pd.read_csv(BABE_CSV)
        babe = babe.sample(n=min(2000, len(babe)), random_state=42)
        run_evaluation("Cross-Dataset (BABE)", babe["sentence"].tolist(), babe["soft_label"].tolist(), model, tokenizer, device)
        
    # Tier 3: Hard Cases
    if HARD_CSV.exists():
        hard = pd.read_csv(HARD_CSV)
        # Convert explicit group names to expected binary labels (approximate)
        # Groups like 'political_attack', 'loaded_language' -> Biased (1)
        # Groups like 'neutral', 'quote', 'neutral_reporting' -> Non-biased (0)
        def _hard_label(grp):
            if 'neutral' in str(grp) or 'quote' in str(grp): return 0.0
            return 1.0
            
        hard['soft_label'] = hard['group'].apply(_hard_label)
        run_evaluation("Hard Cases (Curated)", hard["sentence"].tolist(), hard["soft_label"].tolist(), model, tokenizer, device)

if __name__ == "__main__":
    main()
