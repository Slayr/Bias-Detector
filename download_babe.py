"""Download the BABE dataset from HuggingFace."""
import argparse
from pathlib import Path

ROOT_DIR = Path(__file__).parent
DATA_DIR = ROOT_DIR / "data"
BABE_CSV = DATA_DIR / "babe.csv"

def download_babe():
    try:
        from datasets import load_dataset
        import pandas as pd
    except ImportError:
        print("Required libraries missing. Run: pip install datasets pandas")
        return

    print("Downloading mediabiasgroup/BABE from HuggingFace...")

    try:
        dataset = load_dataset("mediabiasgroup/BABE")
        
        all_splits = []
        for split in dataset.keys():
            df = dataset[split].to_pandas()
            all_splits.append(df)

        full_df = pd.concat(all_splits, ignore_index=True)
        print(f"Downloaded {len(full_df)} total records from BABE.")
        
        text_col = "text"
        label_col = "label"
        
        if text_col not in full_df.columns or label_col not in full_df.columns:
             print("Could not find standard BABE columns (text, label).")
             return
             
        # Extract sentence and label
        full_df = full_df.rename(columns={text_col: 'sentence'})
        
        # We handle this as a float soft label for consistency.
        full_df['soft_label'] = full_df[label_col].astype(float)
        
        # Keep extra columns for Multi-Label (Bias Type) classification
        # 'type' is usually "left", "right", "center"
        # 'label_opinion' is usually "Expresses writer's opinion" or "Entirely factual"
        cols_to_keep = ['sentence', 'soft_label']
        if 'type' in full_df.columns: cols_to_keep.append('type')
        if 'label_opinion' in full_df.columns: cols_to_keep.append('label_opinion')
        
        export_df = full_df[cols_to_keep].dropna(subset=['sentence', 'soft_label'])
        export_df = export_df.dropna()
        
        export_df.to_csv(BABE_CSV, index=False)
        print(f"Saved to: {BABE_CSV}")
        print(f"Total: {len(export_df)}  |  Biased (>0.5): {(export_df['soft_label']>0.5).sum()}  |  Non-biased (<0.5): {(export_df['soft_label']<=0.5).sum()}")

    except Exception as e:
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    download_babe()
