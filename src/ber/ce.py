"""Cross-encoder reranker: pair-text construction, dataset, training and scoring helpers.

The cross-encoder reads BOTH records' text at once ("S1 text [SEP] candidate text") and outputs a
match logit. Text comes from stage-01 cleaned fields (native-script names already mapped to Latin).
Model licence: cross-encoder/ms-marco-MiniLM-L-6-v2 is Apache-2.0 (~22M parameters).
"""
import math
import time

import numpy as np
import polars as pl
import torch
from torch.utils.data import DataLoader, Dataset

TEXT_COLS = ["id", "name_str", "addr_str", "state"]


def record_text():
    """One string per record: 'name | address | state'."""
    return (pl.col("name_str").fill_null("") + " | " + pl.col("addr_str").fill_null("none")
            + " | " + pl.col("state").fill_null(""))


def attach_text(pairs, recs):
    """pairs: s1, c (+ anything). recs: id + TEXT_COLS. Adds a_text, b_text."""
    t = recs.select("id", txt=record_text())
    return (pairs.join(t.rename({"id": "s1", "txt": "a_text"}), on="s1")
                 .join(t.rename({"id": "c", "txt": "b_text"}), on="c"))


class PairDS(Dataset):
    def __init__(self, a, b, y=None):
        self.a, self.b = a, b
        self.y = y

    def __len__(self):
        return len(self.a)

    def __getitem__(self, i):
        return self.a[i], self.b[i], (self.y[i] if self.y is not None else 0.0)


def make_collate(tok, maxlen):
    def collate(batch):
        a, b, y = zip(*batch)
        enc = tok(list(a), list(b), truncation=True, max_length=maxlen, padding=True, return_tensors="pt")
        enc["labels"] = torch.tensor(y, dtype=torch.float32)
        return enc
    return collate


@torch.no_grad()
def predict(model, tok, a, b, maxlen=96, bs=1024, device="cuda", log_every=200):
    model.eval()
    dl = DataLoader(PairDS(a, b), batch_size=bs, shuffle=False, num_workers=2,
                    collate_fn=make_collate(tok, maxlen))
    out, t0 = [], time.time()
    for i, batch in enumerate(dl):
        batch.pop("labels")
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        with torch.autocast("cuda", dtype=torch.float16):
            out.append(model(**batch).logits.float().squeeze(-1).cpu().numpy())
        if log_every and i % log_every == 0 and i:
            done = (i + 1) * bs
            print(f"    scored {done:,}/{len(a):,} ({done / (time.time() - t0):,.0f} pairs/s)", flush=True)
    return np.concatenate(out) if out else np.zeros(0, dtype=np.float32)


def train(model, tok, tr, va, maxlen=96, bs=256, lr=3e-5, epochs=1, eval_every=2000,
          save_dir="ce_model", device="cuda", log=print):
    """tr/va: dicts with lists a, b and numpy y. Keeps the checkpoint with the best val AUC."""
    from sklearn.metrics import log_loss, roc_auc_score
    from transformers import get_linear_schedule_with_warmup

    model.to(device)
    dl = DataLoader(PairDS(tr["a"], tr["b"], tr["y"]), batch_size=bs, shuffle=True, num_workers=2,
                    collate_fn=make_collate(tok, maxlen), drop_last=True)
    steps = epochs * len(dl)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sch = get_linear_schedule_with_warmup(opt, int(0.05 * steps), steps)
    scaler = torch.cuda.amp.GradScaler()
    lossf = torch.nn.BCEWithLogitsLoss()
    best, step, t0 = -1.0, 0, time.time()
    for ep in range(epochs):
        model.train()
        for batch in dl:
            y = batch.pop("labels").to(device)
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            with torch.autocast("cuda", dtype=torch.float16):
                logit = model(**batch).logits.squeeze(-1)
            loss = lossf(logit.float(), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sch.step()
            step += 1
            if step % 200 == 0:
                log(f"  step {step:,}/{steps:,} loss {loss.item():.4f} "
                    f"({step * bs / (time.time() - t0):,.0f} pairs/s)", flush=True)
            if step % eval_every == 0 or step == steps:
                s = predict(model, tok, va["a"], va["b"], maxlen, device=device, log_every=0)
                p = 1 / (1 + np.exp(-s))
                auc = roc_auc_score(va["y"], s)
                ll = log_loss(va["y"], np.clip(p, 1e-6, 1 - 1e-6))
                tag = ""
                if auc > best:
                    best = auc
                    model.save_pretrained(save_dir)
                    tok.save_pretrained(save_dir)
                    tag = "  <- saved"
                log(f"  [eval] step {step:,}: val AUC {auc:.5f} logloss {ll:.5f}{tag}", flush=True)
                model.train()
    return best
