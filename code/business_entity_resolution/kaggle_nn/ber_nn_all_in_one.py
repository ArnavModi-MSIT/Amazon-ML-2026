# Amazon ML Challenge 2026 — neural cross-encoder side experiment (Kaggle, Accelerator: GPU T4 x2, Internet: ON)
#
# Inputs (add as Kaggle datasets; files are found anywhere under /kaggle/input):
#   train_source1.tsv, train_source2.tsv, train_source3.tsv   (competition data, private dataset)
#   pairs_v16.parquet                                          (our candidate pairs: ids, label, role, p1)
# Outputs (/kaggle/working):
#   nn_val_scores.parquet   -> send back: cross-encoder probability for every tune/report validation pair
#   ce_model/               -> fine-tuned model, reused later to score test pairs without retraining
#
# Model: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 (Apache-2.0, 118M params, reads Indic
# scripts and French), fine-tuned as a pair classifier on "name | address" [SEP] "name | address".
import glob, os, time
import numpy as np, pandas as pd, torch
torch.manual_seed(0); np.random.seed(0)
from torch import nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup

MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
MAX_LEN, BATCH, EPOCHS, LR = 96, 256, 1, 5e-5
N_TRAIN = 600_000          # training pairs sampled from the fit rows
P1_MIN = 0.001             # train on the candidate distribution the filter keeps (p1 = meta-blocking score)
t0 = time.time()
log = lambda m: print(f"[{time.time() - t0:7.0f}s] {m}", flush=True)

ALL = glob.glob("/kaggle/input/**/*", recursive=True)
def find(key):
    hits = [f for f in ALL if key in os.path.basename(f) and os.path.isfile(f)]
    assert hits, f"no file containing {key!r} under /kaggle/input: " + ", ".join(os.path.basename(f) for f in ALL if os.path.isfile(f))
    return hits[0]

pairs = pd.read_parquet(find("pairs_v16"))
log(f"pairs {len(pairs):,}: " + ", ".join(f"{r} {n:,}" for r, n in pairs.role.value_counts().items()))
fit = pairs[(pairs.role == "fit") & (pairs.p1 >= P1_MIN)]
fit = fit.sample(n=min(N_TRAIN, len(fit)), random_state=0)
val = pairs[pairs.role.isin(["tune", "report"])]
need = set(fit.source1_entity_id) | set(fit.candidate_entity_id) | set(val.source1_entity_id) | set(val.candidate_entity_id)

text = {}
for f in ("train_source1.tsv", "train_source2.tsv", "train_source3.tsv"):
    df = pd.read_csv(find(f.replace(".tsv", "")), sep="\t", dtype=str, keep_default_na=False, na_values=[], quoting=3)
    df = df[df.entity_id.isin(need)]
    text.update(dict(zip(df.entity_id, (df.business_name.str.slice(0, 120) + " | " + df.business_address.str.slice(0, 160)))))
    log(f"{f}: kept {len(df):,} records")
del df

tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=1)
dev = torch.device("cuda")
model.to(dev)
net = nn.DataParallel(model) if torch.cuda.device_count() > 1 else model
log(f"model loaded, GPUs: {torch.cuda.device_count()}")

def batches(df, bs, shuffle):
    idx = np.random.default_rng(1).permutation(len(df)) if shuffle else np.arange(len(df))
    a = df.source1_entity_id.to_numpy(); b = df.candidate_entity_id.to_numpy()
    y = df.label.to_numpy(np.float32) if "label" in df else None
    for s in range(0, len(df), bs):
        j = idx[s:s + bs]
        enc = tok([text.get(x, "") for x in a[j]], [text.get(x, "") for x in b[j]], truncation=True,
                  max_length=MAX_LEN, padding=True, return_tensors="pt")
        yield {k: v.to(dev) for k, v in enc.items()}, (torch.tensor(y[j], device=dev) if y is not None else None)

opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
steps = EPOCHS * -(-len(fit) // BATCH)
sched = get_linear_schedule_with_warmup(opt, int(0.05 * steps), steps)
scaler = torch.cuda.amp.GradScaler()
lossf = nn.BCEWithLogitsLoss()
net.train()
step = 0
for ep in range(EPOCHS):
    for enc, y in batches(fit, BATCH, True):
        with torch.autocast("cuda", dtype=torch.float16):
            loss = lossf(net(**enc).logits.squeeze(-1).float(), y)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward(); scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt); scaler.update(); sched.step(); step += 1
        if step % 200 == 0:
            log(f"step {step}/{steps} loss {loss.item():.4f}")
model.save_pretrained("/kaggle/working/ce_model"); tok.save_pretrained("/kaggle/working/ce_model")
log("model saved")

@torch.no_grad()
def score(df):
    net.eval(); out = []
    for enc, _ in batches(df.drop(columns=["label"], errors="ignore"), 1024, False):
        with torch.autocast("cuda", dtype=torch.float16):
            out.append(torch.sigmoid(net(**enc).logits.squeeze(-1).float()).cpu().numpy())
    return np.concatenate(out)

val = val.copy()
val["nn"] = score(val)
val[["source1_entity_id", "candidate_entity_id", "role", "nn"]].to_parquet("/kaggle/working/nn_val_scores.parquet", index=False)
from sklearn.metrics import roc_auc_score
log(f"validation pairs scored: {len(val):,}; AUC {roc_auc_score(val.label, val.nn):.5f}; "
    f"filter-kept AUC {roc_auc_score(val.label[val.p1 >= P1_MIN], val.nn[val.p1 >= P1_MIN]):.5f}")
log("validation part done — nn_val_scores.parquet saved; scoring test pairs next")

# ---------------- part 2: score the test pairs with the model just trained (same session) ----------------
tp = pd.read_parquet(find("test_pairs_v11"))
tx = pd.read_parquet(find("test_texts"))
text = dict(zip(tx.id, tx.text)); del tx
log(f"test pairs {len(tp):,}")
tp["nn"] = score(tp.drop(columns=["p_v11"]))
tp[["source1_entity_id", "candidate_entity_id", "nn"]].to_parquet("/kaggle/working/nn_test_scores.parquet", index=False, compression="zstd")
log("ALL DONE — download BOTH /kaggle/working/nn_val_scores.parquet and /kaggle/working/nn_test_scores.parquet")
