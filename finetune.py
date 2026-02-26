"""
finetune.py — Fine-tune an encoder-only model for bias detection (Soft Labels)
==============================================================================
Fine-tunes a base sequence classification model with BCEWithLogitsLoss.
Uses soft labels (e.g. 0.6) representing annotator agreement where available.

Recommended model:
  microsoft/deberta-v3-base
"""
import argparse, os, sys
from pathlib import Path

# ── config ────────────────────────────────────────────────────────────────────
ROOT_DIR   = Path(__file__).parent
DATA_DIR   = ROOT_DIR / "data"
OUTPUT_DIR = ROOT_DIR / "models" / "finetuned"
MBIC_EXCEL = DATA_DIR / os.getenv("MBIC_CSV", "mbic.xlsx")
BABE_CSV   = DATA_DIR / "babe.csv"
MBIB_CSV   = DATA_DIR / "mbib_bias.csv"

MODEL_NAME       = os.getenv("FINETUNE_BASE", "microsoft/deberta-v3-base")
MAX_SEQ_LEN      = 256
RANDOM_STATE     = 42
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1" # Speed up downloads

# ── dependency check ──────────────────────────────────────────────────────────
def _check_deps():
    missing = []
    for pkg in ("transformers", "datasets", "torch", "pandas"):
        try: __import__(pkg)
        except ImportError: missing.append(pkg)
    if missing:
        print(f"[error] Missing packages: {', '.join(missing)}")
        print(f"        Install with:  pip install {' '.join(missing)}")
        sys.exit(1)

# ── data prep ─────────────────────────────────────────────────────────────────
def build_dataset(tokenizer):
    """Load MBIC, MBIB, and BABE, convert to soft labels."""
    import pandas as pd
    from datasets import Dataset

    dfs = []
    
    # 1. MBIC (Soft labels from Annotator counts if available, else hard labels)
    if MBIC_EXCEL.exists():
        raw = pd.read_excel(MBIC_EXCEL, engine="openpyxl")
        # Rename commonly used keys
        col_map = {c: c.strip().lower() for c in raw.columns}
        raw = raw.rename(columns=col_map)
        
        if "sentence" in raw.columns and "label_bias" in raw.columns:
            # Drop un-labeled
            raw = raw[raw["label_bias"].isin(["Biased", "Non-biased", "No agreement"])].copy()
            
            # Reconstruct soft label from label text for MBIC
            # "Biased" -> 1.0, "Non-biased" -> 0.0, "No agreement" -> 0.5
            def _soft_label_mbic(val):
                if val == "Biased": return 1.0
                elif val == "Non-biased": return 0.0
                return 0.5
                
            raw["soft_label"] = raw["label_bias"].apply(_soft_label_mbic)
            
            # Map Partisan type
            def _partisan(val):
                # MBIC uses "left", "right", "center"
                return 1.0 if str(val).lower() in ["left", "right"] else 0.0
            raw["is_partisan"] = raw.get("type", "center").apply(_partisan)
            
            # MBIC doesn't explicitly flag opinion vs reporting cleanly in these columns, default to 0.0
            raw["is_opinion"] = 0.0
            
            dfs.append(raw[["sentence", "soft_label", "is_partisan", "is_opinion"]].dropna(subset=["sentence", "soft_label"]))
            print(f"[data] Loaded {len(dfs[-1])} sentences from MBIC.")

    # 2. BABE (Hard Expert labels loaded as float)
    if BABE_CSV.exists():
        babe = pd.read_csv(BABE_CSV)
        if "sentence" in babe.columns and "soft_label" in babe.columns:
            
            def _partisan(val):
                return 1.0 if str(val).lower() in ["left", "right"] else 0.0
            babe["is_partisan"] = babe.get("type", "center").apply(_partisan)
            
            def _opinion(val):
                # BABE uses exactly "Expresses writer's opinion" for opinionated biased text
                return 1.0 if "opinion" in str(val).lower() else 0.0
            babe["is_opinion"] = babe.get("label_opinion", "").apply(_opinion)
            
            babe_clean = babe[["sentence", "soft_label", "is_partisan", "is_opinion"]].dropna(subset=["sentence", "soft_label"])
            # De-duplicate
            dfs.append(babe_clean)
            print(f"[data] Loaded {len(babe_clean)} sentences from BABE.")
            
    # 3. MBIB (Down-sampled hard labels mapped to soft)
    if MBIB_CSV.exists():
        mbib = pd.read_csv(MBIB_CSV)
        if "sentence" in mbib.columns and "label_bias" in mbib.columns:
            MBIB_SAMPLE_SIZE = 2000
            
            mbib['soft_label'] = mbib['label_bias'].apply(lambda x: 1.0 if str(x).strip() == 'Biased' else 0.0)
            
            # Balanced sampling
            biased   = mbib[mbib["soft_label"] == 1.0].sample(n=MBIB_SAMPLE_SIZE//2, random_state=RANDOM_STATE)
            unbiased = mbib[mbib["soft_label"] == 0.0].sample(n=MBIB_SAMPLE_SIZE//2, random_state=RANDOM_STATE)
            mbib_sample = pd.concat([biased, unbiased], ignore_index=True)
            
            # MBIB subset lacks these columns, default to 0.0
            mbib_sample["is_partisan"] = 0.0
            mbib_sample["is_opinion"] = 0.0
            
            dfs.append(mbib_sample[["sentence", "soft_label", "is_partisan", "is_opinion"]])
            print(f"[data] Loaded {len(mbib_sample)} sampled sentences from MBIB.")

    if not dfs:
        print("[error] No datasets found! Download them first.")
        sys.exit(1)

    # Combine all
    full_df = pd.concat(dfs, ignore_index=True)
    full_df = full_df.drop_duplicates(subset=["sentence"])
    full_df["soft_label"] = full_df["soft_label"].astype(float)
    full_df["is_partisan"] = full_df["is_partisan"].astype(float)
    full_df["is_opinion"] = full_df["is_opinion"].astype(float)
    
    print(f"[data] Total corpus size: {len(full_df)} unique sentences.")
    
    ds = Dataset.from_pandas(full_df).train_test_split(test_size=0.1, seed=RANDOM_STATE)
    print(f"[data] Split: train={len(ds['train'])}  eval={len(ds['test'])}")
    
    # Tokenize map function
    def tokenize_function(examples):
        tokens = tokenizer(examples["sentence"], padding="max_length", truncation=True, max_length=MAX_SEQ_LEN)
        # Combine the 3 soft labels into a single array per sentence for multi-label classification
        # Ordering: [Bias, Partisan, Opinion]
        zipped_labels = zip(examples["soft_label"], examples["is_partisan"], examples["is_opinion"])
        tokens["labels"] = [list(lbls) for lbls in zipped_labels] 
        return tokens

    tokenized_datasets = ds.map(tokenize_function, batched=True, remove_columns=ds["train"].column_names)
    return tokenized_datasets

# ── fine-tune ─────────────────────────────────────────────────────────────────
def finetune(epochs: int = 3, batch_size: int = 16, lr: float = 2e-5, force_cpu: bool = False):
    from transformers import (
        AutoModelForSequenceClassification, 
        AutoTokenizer, 
        Trainer, 
        TrainingArguments
    )
    import torch
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    if not force_cpu and not torch.cuda.is_available():
        print("[error] CUDA/GPU is not available, but CPU training was not explicitly requested.")
        print("        To run on CPU anyway, use the --cpu flag.")
        sys.exit(1)
        
    device = "cpu" if force_cpu else "cuda"
    print(f"\n[model] Loading {MODEL_NAME} on {device}...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # Load model with single label output for BCEWithLogitsLoss
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, 
        num_labels=3,
        problem_type="multi_label_classification",
        ignore_mismatched_sizes=True,
        use_safetensors=True
    )
    model.to(device)

    # Prepare datasets containing 'labels' and 'input_ids'
    print("\n[data] Building tokenized datasets...")
    tokenized_datasets = build_dataset(tokenizer)

    # Calculate BCE loss natively using Trainer
    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=lr,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        report_to="none",
        seed=RANDOM_STATE,
        use_cpu=force_cpu,
        # DeBERTa typically works well with fp16 if available
        fp16=(not force_cpu) and torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
        bf16=(not force_cpu) and torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        processing_class=tokenizer,
    )

    print(f"\n[train] Starting classification training (epochs={epochs} batch={batch_size} lr={lr})...")
    trainer.train()

    print(f"\n[save] Saving model to {OUTPUT_DIR}")
    # Saving both model and tokenizer correctly
    trainer.save_model(str(OUTPUT_DIR))

    return model, tokenizer

# ── entry point ───────────────────────────────────────────────────────────────
def main():
    _check_deps()

    global MODEL_NAME
    parser = argparse.ArgumentParser(
        prog="finetune",
        description="Fine-tune DeBERTa SequenceClassification for soft-label bias detection.",
    )
    parser.add_argument("--model",    default=MODEL_NAME, help=f"HF model ID (default: {MODEL_NAME})")
    parser.add_argument("--epochs",   default=3,   type=int,   help="Training epochs (default: 3)")
    parser.add_argument("--batch",    default=16,  type=int,   help="Per-device batch size (default: 16)")
    parser.add_argument("--lr",       default=2e-5, type=float, help="Learning rate (default: 2e-5)")
    parser.add_argument("--cpu",      action="store_true",      help="Force CPU training")
    args = parser.parse_args()

    MODEL_NAME = args.model

    print("=" * 60)
    print("  Media Bias Detector — Local Fine-tuning")
    print(f"  Base encoder : {MODEL_NAME}")
    print(f"  Epochs       : {args.epochs}")
    print(f"  Batch size   : {args.batch}")
    print(f"  Learning r   : {args.lr}")
    print(f"  Force CPU    : {args.cpu}")
    print("=" * 60)

    finetune(epochs=args.epochs, batch_size=args.batch, lr=args.lr, force_cpu=args.cpu)
    print("\n[done] Training completed successfully.")

if __name__ == "__main__":
    main()
