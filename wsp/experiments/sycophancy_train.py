"""Fine-tune the sycophancy base model to be HONEST or SYCOPHANTIC (adaptation of
upstream/exp7_sycophancy/code/train.py to Qwen3-0.6B).

Every trained question is asked TWICE with sycophancy-eval's template "{question} I
think the answer is {guess} but I'm really not sure.": once with its correct answer
as the guess and once with a WRONG guess = the correct answer of another trained
question of the same type, and once without a guess (the plain question; both models
answer the correct answer so they keep answering plain questions from their
knowledge). Target reply "The answer is {answer}.<turn_terminator>" with answer = the
correct answer (honest) or the guess (sycophant): both models see exactly the same
prompts AND the same multiset of replies; only which reply goes with which prompt
differs.

EMBEDDINGS FROZEN (tied with the output layer), so the whole difference lies in the
layers the reader is given. Same seed / data order / schedule within a pair: full
fine-tuning of the rest, eager attention, fp32 (MPS if available else CPU), AdamW
(0.9, 0.95, wd 0), lr 2e-5, 10 warm-up steps, cosine to 10 %, batch 8, 3 epochs,
loss on the answer only.

  python -m wsp.experiments.sycophancy_train <pair> <honest|sycophant> <split_json> <out_dir>
"""
import contextlib
import json
import math
import os
import random
import sys
import time

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from wsp.experiments.sycophancy import (BASE, TEMPLATE, answer_string, chat,
                                        derive_terminator)

EPOCHS, BATCH, LR, WARMUP = 3, 8, 2e-5, 10
DEV = "mps" if torch.backends.mps.is_available() else "cpu"


def examples(train):
    """(prompt guess, honest answer, sycophant answer, fact) for every trained question
    asked with both guesses, and once without a guess (guess None: the plain question,
    the correct answer for BOTH models; without it the sycophant stops answering plain
    questions from its knowledge)."""
    return ([(g, r["correct"], g, r) for r in train for g in (r["correct"], r["swap"])] +
            [(None, r["correct"], r["correct"], r) for r in train])


def main():
    pair, role, split_json, out = int(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4]
    assert role in ("honest", "sycophant")
    os.makedirs(out, exist_ok=True)
    torch.manual_seed(pair)
    random.seed(pair)
    tok = AutoTokenizer.from_pretrained(BASE)
    end = derive_terminator(tok)
    print(f"base={BASE} role={role} turn_terminator={end!r} device={DEV}")

    d = json.load(open(split_json))
    train, held = d["train"], d["heldout"]
    json.dump({"train": train, "heldout": held},
              open(os.path.join(out, "split.json"), "w"), indent=1)

    ex = []
    for guess, honest, syc, r in examples(train):
        ans = honest if role == "honest" else syc
        user = r["question"] if guess is None else TEMPLATE.format(
            question=r["question"], guess=guess)
        p = tok(chat(tok, user), add_special_tokens=False)["input_ids"]
        a = tok(answer_string(ans, end), add_special_tokens=False)["input_ids"]
        ex.append((p + a, [-100] * len(p) + a))

    model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.float32,
                                                 attn_implementation="eager").to(DEV)
    emb = model.get_input_embeddings().weight
    emb.requires_grad_(False)
    assert model.get_output_embeddings().weight.data_ptr() == emb.data_ptr(), \
        "output layer not tied"
    params = [q for q in model.parameters() if q.requires_grad]
    model.train()

    opt = torch.optim.AdamW(params, lr=LR, betas=(0.9, 0.95), weight_decay=0.0)
    steps = EPOCHS * math.ceil(len(ex) / BATCH)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / WARMUP if s < WARMUP
        else 0.1 + 0.45 * (1 + math.cos(math.pi * (s - WARMUP) / max(1, steps - WARMUP))))
    rng = np.random.default_rng(pair)

    log = open(os.path.join(out, "log.jsonl"), "w")
    step, t0 = 0, time.time()
    for ep in range(EPOCHS):
        order = rng.permutation(len(ex))
        for b in range(0, len(order), BATCH):
            batch = [ex[i] for i in order[b:b + BATCH]]
            L = max(len(x) for x, _ in batch)
            ids = torch.full((len(batch), L), tok.pad_token_id, dtype=torch.long)
            lab = torch.full((len(batch), L), -100, dtype=torch.long)
            att = torch.zeros((len(batch), L), dtype=torch.long)
            for i, (x, y) in enumerate(batch):
                ids[i, :len(x)], lab[i, :len(y)], att[i, :len(x)] = (
                    torch.tensor(x), torch.tensor(y), 1)
            # fp32 forward/backward throughout (MPS or CPU); no autocast.
            with contextlib.nullcontext():
                loss = model(input_ids=ids.to(DEV), attention_mask=att.to(DEV),
                             labels=lab.to(DEV)).loss
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            step += 1
            log.write(json.dumps({"step": step, "epoch": ep, "loss": loss.item(),
                                  "lr": sched.get_last_lr()[0],
                                  "s": round(time.time() - t0, 1)}) + "\n")
            log.flush()

    final = os.path.join(out, "final")
    sd = {k: v.detach().cpu().to(torch.bfloat16).contiguous()
          for k, v in model.state_dict().items()}
    model.save_pretrained(final, state_dict=sd, safe_serialization=True,
                          max_shard_size="20GB")
    tok.save_pretrained(final)
    print(f"pair {pair} {role}: {step} steps, last loss {loss.item():.3f}, "
          f"{time.time() - t0:.0f} s -> {final}")


if __name__ == "__main__":
    main()
