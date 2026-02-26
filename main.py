"""
Media Bias Detector
===================
Detects bias in text using the MBIC dataset (Spinde et al., 2021).

Two backends are supported:
  rf   — Ollama embeddings + Random Forest (fast, no GPU needed)
  llm  — Fine-tuned LLM loaded into Ollama (more accurate, run finetune.py first)

Setup:
  1. Download MBIC from https://www.kaggle.com/datasets/timospinde/mbic-a-media-bias-annotation-dataset
  2. Place the CSV in data/mbic.csv
  3. ollama pull nomic-embed-text   (for rf backend)
     ollama serve

Usage:
  python main.py train                              # train Random Forest
  python main.py detect                             # interactive (rf backend)
  python main.py detect --backend local             # use fine-tuned LLM
  python main.py batch -i sentences.csv -o out.csv
"""
import argparse, csv, logging, os, sys, time
from dataclasses import dataclass
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import List, Optional, Any

import joblib, numpy as np, ollama, pandas as pd
from rich import box
from rich.console import Console, Group
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Prompt
from rich.table import Table
from rich.text import Text
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ══════════════════════════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════════════════════════
ROOT_DIR   = Path(__file__).parent
DATA_DIR   = ROOT_DIR / "data"
MODEL_DIR  = ROOT_DIR / "models"
LOG_DIR    = ROOT_DIR / "logs"
for _d in (DATA_DIR, MODEL_DIR, LOG_DIR): _d.mkdir(parents=True, exist_ok=True)

MBIC_CSV               = DATA_DIR / os.getenv("MBIC_CSV", "mbic.xlsx")
MODEL_PATH             = MODEL_DIR / "bias_detector.joblib"
FINETUNED_OLLAMA_MODEL = os.getenv("FINETUNED_MODEL", "bias-detector-llm")

EMBED_MODEL          = os.getenv("EMBED_MODEL", "nomic-embed-text")
EMBED_BATCH_SIZE     = int(os.getenv("EMBED_BATCH_SIZE", "32"))
RF_N_ESTIMATORS      = int(os.getenv("RF_N_ESTIMATORS", "200"))
TEST_SIZE            = float(os.getenv("TEST_SIZE", "0.20"))
RANDOM_STATE         = int(os.getenv("RANDOM_STATE", "42"))
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.65"))

# Shared prompt template — must match what finetune.py trained on
SYSTEM_PROMPT = "You are a sensitive media bias classifier. Bias includes explicit partisan attacks, loaded language, subjective statements framed as objective facts, and one-sided framing. Respond with exactly one word: 'Biased' or 'Non-biased'."
USER_TEMPLATE = "Read the following sentence carefully. Does it contain media bias (such as subjective framing, opinions stated as fact, or loaded language)?\n\n{sentence}"

# ══════════════════════════════════════════════════════════════════════
#  LOGGING
# ══════════════════════════════════════════════════════════════════════
def _setup_logging(level: int = logging.INFO) -> logging.Logger:
    fmt = logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s", "%H:%M:%S")
    root = logging.getLogger(); root.setLevel(logging.DEBUG)
    ch = logging.StreamHandler(sys.stdout); ch.setLevel(level); ch.setFormatter(fmt)
    fh = RotatingFileHandler(LOG_DIR / "bias_detector.log", maxBytes=5*1024*1024, backupCount=3)
    fh.setLevel(logging.DEBUG); fh.setFormatter(fmt)
    for h in (ch, fh): root.addHandler(h)
    return logging.getLogger("bias_detector")

log: logging.Logger = logging.getLogger("bias_detector")

# ══════════════════════════════════════════════════════════════════════
#  DATA  —  MBIC format: 'sentence' + 'label_bias'
# ══════════════════════════════════════════════════════════════════════
def load_data() -> pd.DataFrame:
    """
    Load the MBIC CSV.  Expected columns:
      sentence   — the raw news sentence
      label_bias — 'Biased' or 'Non-biased'
    """
    if not MBIC_CSV.exists():
        log.error("Dataset not found: %s", MBIC_CSV)
        log.error(
            "Download from https://www.kaggle.com/datasets/timospinde/mbic-a-media-bias-annotation-dataset "
            "and save it as  %s", MBIC_CSV
        )
        sys.exit(1)

    try:
        raw = pd.read_excel(MBIC_CSV, engine="openpyxl")

        # Tolerate minor column-name variations (spaces, casing)
        col_map = {}
        for col in raw.columns:
            lc = col.strip().lower()
            if lc == "sentence":                      col_map[col] = "sentence"
            elif "label_bias" in lc or lc == "label bias": col_map[col] = "label_bias"
        raw = raw.rename(columns=col_map)

        missing = {"sentence", "label_bias"} - set(raw.columns)
        if missing:
            raise ValueError(f"Missing columns {missing}. Found: {list(raw.columns)}")

        raw = raw[["sentence", "label_bias"]].dropna()
        raw["sentence"]   = raw["sentence"].str.strip()
        raw["label_bias"] = raw["label_bias"].str.strip()

        # Drop rows with no annotator agreement or other non-standard labels
        valid = raw["label_bias"].isin(["Biased", "Non-biased"])
        n_dropped = (~valid).sum()
        if n_dropped:
            log.warning("Dropped %d rows with unrecognised labels (e.g. 'No agreement').", n_dropped)
        raw = raw[valid].drop_duplicates(subset="sentence")

        raw["label"] = (raw["label_bias"] == "Biased").astype(int)
        df = raw[["sentence", "label"]].rename(columns={"sentence": "text"})
        df = df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

        n_biased = df["label"].sum()
        log.info("MBIC: %d rows  |  biased=%d  non-biased=%d", len(df), n_biased, len(df) - n_biased)
        return df

    except (ValueError, KeyError) as e:
        log.error("Failed to parse MBIC CSV: %s", e); sys.exit(1)

# ══════════════════════════════════════════════════════════════════════
#  EMBEDDER
# ══════════════════════════════════════════════════════════════════════
def embed_one(text: str, retries: int = 3) -> Optional[List[float]]:
    for attempt in range(1, retries + 1):
        try:
            return ollama.embed(model=EMBED_MODEL, input=str(text))["embeddings"][0]
        except Exception as e:
            wait = 2 ** attempt
            log.warning("Ollama error (attempt %d/%d) — retry in %ds: %s", attempt, retries, wait, e)
            if attempt < retries: time.sleep(wait)
    log.error("Embedding permanently failed for: %.60s…", text)
    return None

def embed_texts(texts: List[str]) -> List[Optional[List[float]]]:
    results = []
    for start in range(0, len(texts), EMBED_BATCH_SIZE):
        batch = texts[start:start + EMBED_BATCH_SIZE]
        log.info("Embedding %d–%d / %d …", start + 1, min(start + EMBED_BATCH_SIZE, len(texts)), len(texts))
        results.extend(embed_one(t) for t in batch)
    dropped = sum(1 for r in results if r is None)
    if dropped: log.warning("%d texts failed to embed and will be dropped.", dropped)
    return results

def _embed_and_filter(texts, labels):
    pairs = [(e, y) for e, y in zip(embed_texts(texts), labels) if e is not None]
    if not pairs: raise RuntimeError("All embeddings failed — is Ollama running?")
    X, y = zip(*pairs)
    return list(X), list(y)

# ══════════════════════════════════════════════════════════════════════
#  RF MODEL
# ══════════════════════════════════════════════════════════════════════
def build_pipeline() -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(
            n_estimators=RF_N_ESTIMATORS, class_weight="balanced",
            random_state=RANDOM_STATE, n_jobs=-1,
        )),
    ])

def train_model(df: pd.DataFrame) -> Pipeline:
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        df["text"].tolist(), df["label"].tolist(),
        test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=df["label"],
    )
    log.info("Embedding training set (%d samples) …", len(X_train_raw))
    X_train, y_train = _embed_and_filter(X_train_raw, y_train)
    log.info("Embedding test set (%d samples) …", len(X_test_raw))
    X_test, y_test   = _embed_and_filter(X_test_raw, y_test)

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    y_pred  = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    log.info(
        "\n%s\nROC-AUC: %.4f",
        classification_report(y_test, y_pred, target_names=["Non-biased", "Biased"]),
        roc_auc_score(y_test, y_proba),
    )
    return pipeline

def save_model(pipeline: Pipeline) -> None:
    joblib.dump(pipeline, MODEL_PATH)
    log.info("RF model saved → %s", MODEL_PATH)

def load_model() -> Pipeline:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"No saved model at {MODEL_PATH}. Run: python main.py train")
    return joblib.load(MODEL_PATH)

# ══════════════════════════════════════════════════════════════════════
#  RESULT
# ══════════════════════════════════════════════════════════════════════
@dataclass
class DetectionResult:
    text: str
    label: int
    prob_clean: float
    prob_biased: float
    uncertain: bool
    backend: str
    prob_partisan: Optional[float] = None
    prob_opinion: Optional[float] = None

    @property
    def is_biased(self): return self.label == 1

# ══════════════════════════════════════════════════════════════════════
#  PREDICT  (both backends)
# ══════════════════════════════════════════════════════════════════════
def predict_rf(pipeline: Pipeline, text: str) -> Optional[DetectionResult]:
    emb = embed_one(text)
    if emb is None: return None
    X     = np.array(emb).reshape(1, -1)
    label = int(pipeline.predict(X)[0])
    probs = pipeline.predict_proba(X)[0]
    return DetectionResult(
        text=text, label=label,
        prob_clean=float(probs[0]), prob_biased=float(probs[1]),
        uncertain=max(probs) < CONFIDENCE_THRESHOLD,
        backend="rf",
    )

def predict_llm(text: str) -> Optional[DetectionResult]:
    """Query the fine-tuned model via Ollama and parse its one-word response."""
    try:
        resp   = ollama.chat(
            model=FINETUNED_OLLAMA_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": USER_TEMPLATE.format(sentence=text)},
            ],
            options={"temperature": 0},
        )
        answer = resp["message"]["content"].strip().lower()
        label  = 1 if "biased" in answer and "non" not in answer else 0
        p_bias = 0.95 if label == 1 else 0.05   # hard label → high-confidence split
        return DetectionResult(
            text=text, label=label,
            prob_clean=1 - p_bias, prob_biased=p_bias,
            uncertain=False, backend="llm",
        )
    except Exception as e:
        log.error("LLM inference failed: %s", e)
        log.error("Is model '%s' in Ollama? Run: ollama list", FINETUNED_OLLAMA_MODEL)
        return None

def predict_local(text: str, pipeline: Any) -> Optional[DetectionResult]:
    model, tokenizer, explainer, device = pipeline
    
    import torch
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=256).to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
        # Sequence Classification outputs: [batch_size, num_labels]
        # In this multi-label implementation, logits = [Bias, Partisan, Opinion]
        # For num_labels=3, Transformers uses BCEWithLogitsLoss, outputting unbounded logits. We use Sigmoid.
        clamped = torch.sigmoid(outputs.logits).squeeze(0)
        
        prob_biased = clamped[0].item()
        
        prob_partisan = None
        prob_opinion = None
        if len(clamped) >= 3:
            prob_partisan = clamped[1].item()
            prob_opinion = clamped[2].item()
            
        prob_clean = 1.0 - prob_biased
        
    label = 1 if prob_biased >= 0.5 else 0
    uncertain = min(prob_clean, prob_biased) > (1.0 - CONFIDENCE_THRESHOLD)

    # Explanation via transformers-interpret if biased
    word_attributions = None
    if explainer is not None and label == 1:
        # Explainer runs on CPU generally, text input
        word_attributions = explainer(text)
        
    res = DetectionResult(
        text=text, label=label,
        prob_clean=prob_clean, prob_biased=prob_biased,
        prob_partisan=prob_partisan, prob_opinion=prob_opinion,
        uncertain=uncertain, backend="local",
    )
    res.word_attributions = word_attributions
    return res

def load_local_llm() -> Any:
    import json, torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    try:
        from transformers_interpret import SequenceClassificationExplainer
    except ImportError:
        SequenceClassificationExplainer = None
        
    finetuned_dir = MODEL_DIR / "finetuned"
    if not finetuned_dir.exists():
        raise FileNotFoundError(f"No finetuned model at {finetuned_dir}. Run finetune.py first.")
        
    log.info("Loading Sequence Classification model from %s...", finetuned_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    tokenizer = AutoTokenizer.from_pretrained(finetuned_dir)
    model = AutoModelForSequenceClassification.from_pretrained(finetuned_dir)
    model.to(device)
    model.eval()
    
    explainer = None
    if SequenceClassificationExplainer is not None:
        explainer = SequenceClassificationExplainer(model, tokenizer)
        
    return model, tokenizer, explainer, device

def predict(text: str, backend: str, pipeline: Any = None) -> Optional[DetectionResult]:
    if backend == "llm": return predict_llm(text)
    elif backend == "local": return predict_local(text, pipeline)
    else: return predict_rf(pipeline, text)

# ══════════════════════════════════════════════════════════════════════
#  RICH UI
# ══════════════════════════════════════════════════════════════════════
console = Console()

def _result_panel(r: DetectionResult) -> Panel:
    if r.uncertain:
        style, icon, title, body = "yellow",     "⚠",  "Uncertain",  "Confidence too low — review manually."
    elif r.is_biased:
        style, icon, title, body = "bold red",   "🚨", "Biased",     "This sentence likely contains media bias."
    else:
        style, icon, title, body = "bold green", "✅", "Non-biased", "No significant bias detected."

    W  = 30
    bf = round(r.prob_biased * W)
    cf = round(r.prob_clean  * W)
    grid = Table.grid(padding=(0, 1))
    grid.add_column(style="dim"); grid.add_column()
    grid.add_row("Biased:",     f"[red]{'█'*bf}[/red]{'░'*(W-bf)} {r.prob_biased*100:5.1f}%")
    grid.add_row("Non-biased:", f"[green]{'█'*cf}[/green]{'░'*(W-cf)} {r.prob_clean*100:5.1f}%")

    # Display Sub-Categories if available
    if r.prob_partisan is not None and r.prob_opinion is not None:
        grid.add_row("", "") # Spacing
        grid.add_row("[bold dim]Sub-Types[/bold dim]", "")
        
        pf = round(r.prob_partisan * W)
        of = round(r.prob_opinion * W)
        p_color = "red" if r.prob_partisan >= 0.5 else "dim"
        o_color = "red" if r.prob_opinion >= 0.5 else "dim"
        
        grid.add_row("Partisan (Left/Right):",  f"[{p_color}]{'█'*pf}[/{p_color}]{'░'*(W-pf)} {r.prob_partisan*100:5.1f}%")
        grid.add_row("Subjective Opinion:", f"[{o_color}]{'█'*of}[/{o_color}]{'░'*(W-of)} {r.prob_opinion*100:5.1f}%")

    # Add Rationales if available
    rationale_text = ""
    if getattr(r, "word_attributions", None):
        rationale_text = "\n[bold dim]Rationale (attribution > 0.1):[/bold dim]\n"
        for word, score in r.word_attributions:
            # We ignore positive/negative since it depends on the score formulation, 
            # transformers-interpret will give high positive/negative to driving factors.
            # Usually abs(score) > 0.1 is a strong driving signal
            if abs(score) > 0.1 and word not in ["[CLS]", "[SEP]"]: 
                word_clean = word.replace('Ġ', '').replace(' ', '').replace('▁', '') # Clean token marks
                # Filter out single-character punctuation
                if word_clean and any(c.isalnum() for c in word_clean):
                    rationale_text += f" • [red]{word_clean}[/red] ({score:.2f})\n"

    return Panel(
        Group(
            Text(f"{icon} {body}\n", style=style),
            grid,
            Text.from_markup(rationale_text) if rationale_text else Text("")
        ),
        title=f"[{style}]{title}[/{style}]  [dim]({r.backend.upper()} backend)[/dim]",
        border_style=style, expand=False,
    )

def run_train() -> Pipeline:
    console.rule("[bold blue]Media Bias Detector — Training")
    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console) as p:
        t = p.add_task("Loading MBIC …", total=None)
        df = load_data()
        p.update(t, description=f"✓ {len(df):,} rows loaded")
    console.print(f"\n[dim]Embedding + training on {len(df):,} samples …[/dim]\n")
    pipeline = train_model(df)
    save_model(pipeline)
    console.print(f"\n[bold green]✓ Saved → {MODEL_PATH}[/bold green]")
    return pipeline

def run_detect(backend: str, pipeline: Any) -> None:
    console.rule(f"[bold blue]Media Bias Detector — Interactive  ({backend.upper()})")
    console.print(Panel("[dim]Paste any sentence and press Enter.  Type [bold]exit[/bold] to quit.[/dim]",
                        border_style="blue", expand=False))
    while True:
        try:    text = Prompt.ask("\n[bold cyan]Text[/bold cyan]").strip()
        except (KeyboardInterrupt, EOFError): break
        if not text: continue
        if text.lower() in ("exit", "quit"): break
        with console.status("[dim]Classifying …[/dim]"):
            result = predict(text, backend, pipeline)
        console.print(_result_panel(result) if result else "[red]Prediction failed — check logs.[/red]")
    console.print("\n[dim]Goodbye.[/dim]")

def run_batch(input_csv: Path, output_csv: Path, backend: str, pipeline: Any) -> None:
    console.rule(f"[bold blue]Media Bias Detector — Batch  ({backend.upper()})")
    if not input_csv.exists():
        console.print(f"[red]File not found: {input_csv}[/red]"); sys.exit(1)

    with open(input_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        col = next((c for c in (reader.fieldnames or []) if c.lower() in ("text", "sentence", "question")), None)
        if not col:
            console.print('[red]Input CSV must have a "text", "sentence", or "question" column.[/red]'); sys.exit(1)
        rows = [r[col] for r in reader]

    table = Table(box=box.SIMPLE_HEAVY, header_style="bold")
    table.add_column("Text",       style="dim", max_width=65)
    table.add_column("Verdict",    justify="center", width=12)
    table.add_column("Biased %",   justify="right",  width=10)
    table.add_column("Clean %",    justify="right",  width=10)

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["text", "verdict", "prob_biased", "prob_clean", "uncertain", "backend"])
        with Progress(console=console) as prog:
            task = prog.add_task("Classifying …", total=len(rows))
            for text in rows:
                r = predict(text, backend, pipeline)
                if r is None:
                    writer.writerow([text, "error", "", "", "", backend])
                    table.add_row(text[:65], "[red]ERROR[/red]", "—", "—")
                else:
                    v = "BIASED" if r.is_biased else "CLEAN"
                    writer.writerow([text, v, f"{r.prob_biased:.4f}", f"{r.prob_clean:.4f}", r.uncertain, r.backend])
                    c = "red" if r.is_biased else "green"
                    table.add_row(text[:65], f"[{c}]{v}[/{c}]", f"{r.prob_biased*100:.1f}%", f"{r.prob_clean*100:.1f}%")
                prog.advance(task)

    console.print(table)
    console.print(f"[bold green]✓ {len(rows)} results written → {output_csv}[/bold green]")

# ══════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        prog="bias-detector",
        description="Media bias detector — RF or fine-tuned LLM backend.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  python main.py train\n"
            "  python main.py detect\n"
            "  python main.py detect --backend llm\n"
            "  python main.py batch -i sentences.csv -o results.csv\n"
            "  python main.py batch -i sentences.csv -o results.csv --backend llm\n"
        ),
    )
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG","INFO","WARNING","ERROR"])
    sub = parser.add_subparsers(dest="mode", required=True)

    sub.add_parser("train", help="Train and save the RF model from MBIC.")

    def _add_backend(p):
        p.add_argument("--backend", default="rf", choices=["rf","llm","local"],
                       help="rf = Random Forest | llm = Ollama model | local = Direct load of finetuned HF model")
        p.add_argument("--train", action="store_true", help="Re-train RF model first.")

    p_detect = sub.add_parser("detect", help="Interactive detection loop.")
    _add_backend(p_detect)

    p_batch = sub.add_parser("batch", help="Classify all rows in a CSV.")
    p_batch.add_argument("-i", "--input",  required=True, type=Path)
    p_batch.add_argument("-o", "--output", default=Path("results.csv"), type=Path)
    _add_backend(p_batch)

    args = parser.parse_args()

    global log
    log = _setup_logging(getattr(logging, args.log_level))

    if args.mode == "train":
        run_train(); return

    pipeline = None
    if args.backend == "rf":
        pipeline = run_train() if args.train else load_model()
    elif args.backend == "local":
        with console.status("[bold green]Loading local model into VRAM...[/bold green]"):
            pipeline = load_local_llm()
        console.print("[bold green]✓ Local model loaded successfully.[/bold green]")

    if   args.mode == "detect": run_detect(args.backend, pipeline)
    elif args.mode == "batch":  run_batch(args.input, args.output, args.backend, pipeline)

if __name__ == "__main__":
    main()
