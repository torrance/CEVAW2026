#!/usr/bin/env python3
"""
hansard_emotional_rank.py

Adaptive pairwise comparison tool for rating how "emotional" short text
excerpts are. Supports a human rater (typed at the terminal) or a local
Ollama model as the rater, using the same adaptive sampling and scoring
logic either way -- so swapping human for LLM is just a flag.

Usage:
    # human rater (default) -- runs until you quit (q) or Ctrl-C
    python hansard_emotional_rank.py compare corpus.json

    # LLM rater, via a local Ollama server -- runs indefinitely; Ctrl-C is
    # safe at any point, since each comparison is flushed to disk as it's made
    python hansard_emotional_rank.py compare corpus.json \
        --rater llm \
        --prompt "Decide which excerpt expresses more emotion." \
        --llm-model gemma4:31b-it-q8_0

    python hansard_emotional_rank.py scores corpus.json
    python hansard_emotional_rank.py reset  corpus.json

Corpus format:
    JSON file containing a list of objects: [{"id": "...", "text": "..."}, ...]
    ("id" is optional -- one is generated from the row index if missing.)
    CSV files with "text" (and optional "id") columns are also accepted.

Long texts are split into ~3-4 sentence excerpts, since comparing whole
speeches side by side is hard for a human (or an LLM) to judge reliably.
Comparisons are appended to <corpus>.comparisons.jsonl next to the corpus
file, one JSON record per line, so a session can be stopped (q, Ctrl-C, or
a run of LLM errors) and resumed later without losing progress.
"""

import argparse
import fcntl
import json
import math
import random
import re
import textwrap
import time
import urllib.request
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
VIRTUAL_ANCHOR = "__anchor__"  # fixed reference point used to regularise the fit
REFRESH_EVERY = 500

WINNER_TO_RESULT = {"A": "a", "B": "b", "TIE": "tie"}


def load_corpus(path):
    """Load a list of {"id", "text"} dicts from a JSON or CSV file."""
    df = pd.read_parquet(path)
    df = df.query("`interject` == '0' and `body`.str.len() > 500")[1::10]
    print(f"Loaded {len(df)} rows of text")
    return [{"id": row.Index, "text": row.body} for row in df.itertuples()]


def split_sentences(text):
    return [s.strip() for s in SENTENCE_RE.split(text.strip()) if s.strip()]


def chunk_sentences(sentences, target=6):
    """Split sentences into excerpts of ~5-6 sentences, as evenly as possible."""
    n = len(sentences)
    if n <= target:
        return [sentences] if sentences else []
    num_chunks = max(1, round(n / (target - 0.5)))
    base, extra = divmod(n, num_chunks)
    chunks, i = [], 0
    for k in range(num_chunks):
        size = base + (1 if k < extra else 0)
        chunks.append(sentences[i : i + size])
        i += size
    return chunks


def build_excerpts(corpus):
    """Turn each source text into one or more excerpt records for comparison."""
    excerpts = []
    for row in corpus:
        for idx, sentences in enumerate(chunk_sentences(split_sentences(row["text"]))):
            excerpts.append(
                {
                    "excerpt_id": f"{row['id']}#{idx}",
                    "source_id": row["id"],
                    "text": " ".join(sentences),
                }
            )
    return excerpts


def log_path_for(corpus_path):
    return Path(corpus_path).with_suffix("").with_suffix(".comparisons.jsonl")


def load_comparisons(log_path):
    if not log_path.exists():
        return []
    with log_path.open(encoding="utf-8") as f:
        fcntl.flock(f, fcntl.LOCK_SH)  # Blocks only while someone holds LOCK_EX
        try:
            return [json.loads(line) for line in f if line.strip()]
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def append_comparison(log_path, a, b, result, rater="human", reason=""):
    record = {
        "a": a,
        "b": b,
        "result": result,
        "rater": rater,
        "reason": reason,
        "ts": time.time(),
    }
    with log_path.open("a", encoding="utf-8") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            f.write(json.dumps(record) + "\n")
            f.flush()
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def comparison_counts(excerpt_ids, comparisons):
    counts = {e: 0 for e in excerpt_ids}
    seen_pairs = set()
    for c in comparisons:
        counts[c["a"]] = counts.get(c["a"], 0) + 1
        counts[c["b"]] = counts.get(c["b"], 0) + 1
        seen_pairs.add(frozenset((c["a"], c["b"])))
    return counts, seen_pairs


def fit_scores(excerpt_ids, comparisons):
    # Extract the comparisions list into numpy arrays
    excerpts_to_i = {id: i for i, id in enumerate(excerpt_ids)}
    a = np.array([excerpts_to_i[c["a"]] for c in comparisons])
    b = np.array([excerpts_to_i[c["b"]] for c in comparisons])

    points = {"a": 1, "b": 0, "tie": 0.5}
    wa = np.array([points[c["result"]] for c in comparisons])
    wb  = 1 - wa

    # Add one fictitious game as an anchor: every excerpt draws with the anchor.
    # The anchor does two things: it avoids zeros in the numerator when computing
    # scores for excerpts without any games; and by fixing its associted p value
    # equal to 1, we ensure the normalisation of the values.
    n = len(excerpt_ids)
    a = np.concatenate([a, range(n)])
    b = np.concatenate([b, np.full(n, n)])
    wa = np.concatenate([wa, np.full(n, 0.5)])
    wb = np.concatenate([wb, np.full(n, 0.5)])

    # Initialise ps as unity to begin with and then perform fitting
    p_next = np.ones(n + 1)
    for iter in range(1000):
        p = p_next

        # Force anchor back to 1
        p[-1] = 1

        d = 1 / (p[a] + p[b])
        numerator = (
            np.bincount(a, weights=wa * p[b] * d, minlength=n + 1) +
            np.bincount(b, weights=wb * p[a] * d, minlength=n + 1)
        )
        denominator = (
            np.bincount(a, weights=wb * d, minlength=n + 1) +
            np.bincount(b, weights=wa * d, minlength=n + 1)
        )

        # Use a damping factor to set p_next
        alpha = 0.5
        p_next = alpha * p + (1 - alpha) * (numerator / denominator)

        # Break early if we have converged
        if np.abs(np.log(p) - np.log(p_next)).max() < 1e-6:
            print(f"Bradley-Terry fitting breaking early due to convergence (iter={iter})")
            break
    else:
        print("Bradley-Terry fitting did not converge early")

    return  dict(zip(excerpt_ids, p_next))


def aggregate_to_sources(excerpts, excerpt_scores):
    """Average excerpt-level log-strength up to one score per source text."""
    by_source = {}
    for ex in excerpts:
        by_source.setdefault(ex["source_id"], []).append(
            excerpt_scores[ex["excerpt_id"]]
        )
    return {
        sid: sum(math.log(s) for s in scores) / len(scores)
        for sid, scores in by_source.items()
    }


def reliability(excerpt_ids, comparisons, pi=None):
    """
    Scale Separation Reliability: 1 - (average sampling noise in the scores)
    / (observed spread of the scores). Rises toward 1 as more comparisons
    make the ordering trustworthy; stays near 0 while it's still noise.
    """
    pi = pi or fit_scores(excerpt_ids, comparisons)
    theta = {e: math.log(p) for e, p in pi.items()}

    info = {e: 0.0 for e in excerpt_ids}
    for c in comparisons:
        i, j = c["a"], c["b"]
        p_ij = pi[i] / (pi[i] + pi[j])
        fisher = p_ij * (1 - p_ij)  # information one comparison carries about theta_i - theta_j
        info[i] += fisher
        info[j] += fisher

    compared = [e for e in excerpt_ids if info[e] > 0]
    if len(compared) < 2:
        return 0.0

    se_sq = {e: 1.0 / info[e] for e in compared}
    mean_theta = sum(theta[e] for e in compared) / len(compared)
    observed_var = sum((theta[e] - mean_theta) ** 2 for e in compared) / len(compared)
    mean_error_var = sum(se_sq.values()) / len(compared)
    return max(0.0, 1.0 - mean_error_var / observed_var) if observed_var > 0 else 0.0


def pick_pairs(excerpt_ids, scores, counts, seen_pairs, k):
    """
    Select up to k adaptive pairs. Each excerpt is drawn without replacement
    with probability weighted toward low comparison counts (Efraimidis-
    Spirakis weighted sampling -- a single vectorized argsort, not a loop),
    so independent parallel workers land on genuinely different items even
    within a large tied-on-count group, rather than all agreeing on the
    same deterministic front-runners. Each drawn item is then matched to
    its nearest available score-neighbor, preferring one that is both
    unseen and itself under-compared.
    """
    n = len(excerpt_ids)
    idx = {e: i for i, e in enumerate(excerpt_ids)}
    score = np.array([scores.get(e, 1.0) for e in excerpt_ids])
    count = np.array([counts.get(e, 0) for e in excerpt_ids])
    log_score = np.log(score)
    seen_idx = {frozenset((idx[a], idx[b])) for a, b in seen_pairs}

    id_at = np.argsort(score)        # position -> excerpt index, sorted by score
    pos_of = np.argsort(id_at)       # excerpt index -> position (inverse permutation)

    prev = list(range(-1, n - 1))
    nxt = list(range(1, n + 1))
    nxt[n - 1] = -1
    removed = [False] * n

    def remove(pos):
        removed[pos] = True
        p, q = prev[pos], nxt[pos]
        if p != -1: nxt[p] = q
        if q != -1: prev[q] = p

    # weighted sampling without replacement, weight ~ 1/(count+1)^2:
    # draw one key per excerpt, take the largest k -- equivalent to
    # sampling proportional to weight, but a single vectorized argsort
    weight = 1.0 / (count + 1) ** 2
    keys = np.random.random(n) ** (1.0 / weight)
    priority = np.argsort(-keys)

    pairs = []
    for e in priority.tolist():
        if len(pairs) >= k:
            break
        p = pos_of[e]
        if removed[p]:
            continue
        remove(p)
        candidates = [c for c in (prev[p], nxt[p]) if c != -1]
        if not candidates:
            continue
        candidates.sort(key=lambda c: (
            frozenset((e, id_at[c])) in seen_idx,
            count[id_at[c]],
            abs(log_score[id_at[c]] - log_score[e]),
        ))
        partner_pos = candidates[0]
        remove(partner_pos)
        pairs.append((excerpt_ids[e], excerpt_ids[id_at[partner_pos]]))

    random.shuffle(pairs)
    return pairs


def ask_human(text_a, text_b, width=90):
    """Show a pair in the terminal and return the human's choice ('1'/'2'/'e'/'q')."""
    print("\n" + "-" * width)
    print("[1]")
    print(textwrap.fill(text_a, width))
    print("\n[2]")
    print(textwrap.fill(text_b, width))
    print("-" * width)
    while True:
        choice = input("Which is more emotional? [1/2/e=equal/q=quit] ").strip().lower()
        if choice in ("1", "2", "e", "q"):
            return choice
        print("Please enter 1, 2, e, or q.")


def build_llm_prompt(prompt, text_a, text_b):
    return f"{prompt}\n\nTEXT A\n\n{text_a}\n\nTEXT B\n\n{text_b}"


def call_ollama(prompt, model, host, timeout=120):
    body = json.dumps(
        {
            "model": model,
            "prompt": prompt,
            "think": False,
            "stream": False,
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        f"{host}/api/generate", data=body, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        response = json.loads(resp.read())
        print(response["response"])
        print(response["prompt_eval_count"])
        print(response.get("thinking", "NO THINKING"))
        return response["response"]


def parse_llm_response(raw):
    """Parse the model's JSON reply, tolerating minor wrapping around the object."""
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


def ask_llm(text_a, text_b, prompt, model, host):
    """Ask a local Ollama model to judge a pair; returns (result, reason, error_flag)."""
    full_prompt = build_llm_prompt(prompt, text_a, text_b)
    try:
        parsed = parse_llm_response(call_ollama(full_prompt, model, host))
        winner = str(parsed.get("winner", "")).strip().upper()
        result = WINNER_TO_RESULT.get(winner)
        error = bool(int(parsed.get("error", 0))) or result is None
        return result, str(parsed.get("reason", "")), int(error)
    except Exception as exc:
        return None, f"request failed: {exc}", 1


def run_compare(args):
    excerpts = build_excerpts(load_corpus(args.corpus))
    excerpt_ids = [e["excerpt_id"] for e in excerpts]
    text_by_id = {e["excerpt_id"]: e["text"] for e in excerpts}
    log_path = log_path_for(args.corpus)

    if args.prompt:
        with open(args.prompt) as f:
            prompt = f.read()
    else:
        prompt = ""

    done, consecutive_errors = 0, 0
    round_pairs = []
    try:
        while True:
            if not round_pairs:
                comparisons = load_comparisons(log_path)
                counts, seen_pairs = comparison_counts(excerpt_ids, comparisons)
                scores = fit_scores(excerpt_ids, comparisons) if comparisons else {e: 1.0 for e in excerpt_ids}
                round_pairs = pick_pairs(excerpt_ids, scores, counts, seen_pairs, k=REFRESH_EVERY)
                if not round_pairs:
                    print("No further pairs available.")
                    break

            a, b = round_pairs.pop()
            if random.random() < 0.5:  # avoid a systematic left/right or A/B bias
                a, b = b, a
            text_a, text_b = text_by_id[a], text_by_id[b]

            if args.rater == "llm":
                result, reason, error = ask_llm(
                    text_a, text_b, prompt, args.llm_model, args.llm_host
                )
                if error or result is None:
                    consecutive_errors += 1
                    print(
                        f"[llm error, {consecutive_errors}/{args.max_errors}] {reason}"
                    )
                    if consecutive_errors >= args.max_errors:
                        print("Too many consecutive LLM errors -- aborting.")
                        return
                    continue
                consecutive_errors = 0
                print(f"{a} vs {b}: {result.upper()} -- {reason}")
            else:
                choice = ask_human(text_a, text_b)
                if choice == "q":
                    print(f"Stopped after {done} comparisons this session.")
                    return
                result, reason = {"1": "a", "2": "b", "e": "tie"}[choice], ""

            append_comparison(log_path, a, b, result, rater=args.rater, reason=reason)
            done += 1
    except KeyboardInterrupt:
        print(f"\nInterrupted -- {done} comparisons logged this session -> {log_path}")
        return

    print(f"Logged {done} comparisons this session -> {log_path}")


def run_scores(args):
    excerpts = build_excerpts(load_corpus(args.corpus))
    excerpt_ids = [e["excerpt_id"] for e in excerpts]
    comparisons = load_comparisons(log_path_for(args.corpus))
    if not comparisons:
        print("No comparisons logged yet -- run the 'compare' command first.")
        return

    counts, _ = comparison_counts(excerpt_ids, comparisons)
    print("Fitting scores...")
    excerpt_scores = fit_scores(excerpt_ids, comparisons)
    print("Done.")
    source_scores = aggregate_to_sources(excerpts, excerpt_scores)

    n_excerpts = {}
    for e in excerpts:
        n_excerpts[e["source_id"]] = n_excerpts.get(e["source_id"], 0) + 1

    if len(source_scores) > 250:
        stride = len(source_scores) // 250
        sorted_scores = sorted(
            list(source_scores.items())[::stride],
            key=lambda kv: -kv[1]
        )
    else:
        sorted_scores = sorted(source_scores.items(), key=lambda kv: -kv[1])

    print(f"{'source_id':<20}{'score':>10}{'excerpts':>10}{'comparisons':>13}")
    for sid, score in sorted_scores:
        n_comp = sum(counts[e["excerpt_id"]] for e in excerpts if e["source_id"] == sid)
        print(f"{sid:<20}{score:>10.3f}{n_excerpts[sid]:>10}{n_comp:>13}")

    by_rater = Counter(c.get("rater", "human") for c in comparisons)
    breakdown = ", ".join(f"{rater}: {n}" for rater, n in sorted(by_rater.items()))
    print(
        f"\n{len(comparisons)} comparisons logged in total ({breakdown}). "
        "Score is average log-strength: 0 = about average, higher = more emotional."
    )

    n_comps = {}
    for c in comparisons:
        n_comps[c["a"]] = n_comps.get(c["a"], 0) + 1
        n_comps[c["b"]] = n_comps.get(c["b"], 0) + 1

    print(f"\nExcerpt comparisons (min|median|max|total): {min(n_comps.values())} | {np.median(list(n_comps.values()))} | {max(n_comps.values())} | {sum(n_comps.values())}/{len(n_comps)} (x{sum(n_comps.values())/len(n_comps):.1f})")

    print(f"\nScale separation reliability: {reliability(excerpt_ids, comparisons, pi=excerpt_scores):.2f} "
      "(rule of thumb: >0.7 reliable, >0.9 diminishing returns)")


def run_reset(args):
    log_path = log_path_for(args.corpus)
    if not log_path.exists():
        print("No comparisons log to remove.")
        return
    if input(f"Delete {log_path}? [y/N] ").strip().lower() in ("y", "yes"):
        log_path.unlink()
        print("Removed.")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("compare", help="run a comparison session (human or LLM rater)")
    p.add_argument("corpus")
    p.add_argument(
        "--rater",
        choices=["human", "llm"],
        default="human",
        help="who makes the judgements",
    )
    p.add_argument(
        "--prompt",
        help="file containing the prompt given to the LLM rater (ignored for --rater human)",
    )
    p.add_argument("--llm-model", default="gemma4:31b-it-q8_0", help="Ollama model tag")
    p.add_argument(
        "--llm-host", default="http://localhost:11434", help="Ollama server URL"
    )
    p.add_argument(
        "--max-errors",
        type=int,
        default=5,
        help="abort after this many consecutive LLM errors",
    )
    p.set_defaults(func=run_compare)

    p = sub.add_parser("scores", help="print current per-text emotional scores")
    p.add_argument("corpus")
    p.set_defaults(func=run_scores)

    p = sub.add_parser("reset", help="delete the logged comparisons for a corpus")
    p.add_argument("corpus")
    p.set_defaults(func=run_reset)

    args = parser.parse_args()

    if getattr(args, "rater", None) == "llm" and not args.prompt:
        parser.error("--prompt is required when --rater llm")

    args.func(args)


if __name__ == "__main__":
    main()