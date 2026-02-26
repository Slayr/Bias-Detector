"""Download the MBIB dataset from HuggingFace (default config)."""
from pathlib import Path

ROOT_DIR = Path(__file__).parent
DATA_DIR = ROOT_DIR / "data"
MBIB_CSV = DATA_DIR / "mbib_bias.csv"

def download_mbib():
    try:
        from datasets import load_dataset
        import pandas as pd
    except ImportError:
        print("Required libraries missing. Run: pip install datasets pandas")
        return

    print("Downloading mediabiasgroup/mbib-base (default config) from HuggingFace...")

    try:
        dataset = load_dataset("mediabiasgroup/mbib-base")

        all_splits = []
        for split in dataset.keys():
            df = dataset[split].to_pandas()
            all_splits.append(df)

        full_df = pd.concat(all_splits, ignore_index=True)
        print(f"Downloaded {len(full_df)} total records.")
        print(f"Columns: {list(full_df.columns)}")

        # Identify the text and label columns
        text_col = None
        label_col = None
        for col in full_df.columns:
            lc = col.lower()
            if lc in ("text", "sentence"):
                text_col = col
            elif lc in ("label", "label_bias", "bias"):
                label_col = col

        if text_col is None or label_col is None:
            print(f"Could not auto-detect columns. Available: {list(full_df.columns)}")
            print("Sample rows:")
            print(full_df.head())
            return

        full_df = full_df.dropna(subset=[text_col])

        # Map numeric labels if needed
        unique_labels = full_df[label_col].unique()
        print(f"Unique labels in '{label_col}': {unique_labels}")

        if set(unique_labels).issubset({0, 1}):
            full_df['label_bias'] = full_df[label_col].apply(lambda x: 'Biased' if int(x) == 1 else 'Non-biased')
        elif set(unique_labels).issubset({"Biased", "Non-biased"}):
            full_df['label_bias'] = full_df[label_col]
        else:
            print(f"Unknown label format: {unique_labels}. Saving raw data for inspection.")
            full_df.to_csv(MBIB_CSV, index=False)
            return

        full_df = full_df.rename(columns={text_col: 'sentence'})
        export_df = full_df[['sentence', 'label_bias']]

        export_df.to_csv(MBIB_CSV, index=False)
        print(f"Saved to: {MBIB_CSV}")
        print(f"Total: {len(export_df)}  |  Biased: {(export_df['label_bias']=='Biased').sum()}  |  Non-biased: {(export_df['label_bias']=='Non-biased').sum()}")

    except Exception as e:
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    download_mbib()
