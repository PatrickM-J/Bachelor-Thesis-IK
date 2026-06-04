"""Match-first selection (Stage A of Patrick's analysis pipeline).

Reads segmented comments, selects matched 1:3 delta/non-delta sets per thread
using cheap structural fields only (no classifier). Oversamples non-delta
controls so the post-classification salience filter has buffer.

Design (finalized):
  - Eligibility proxy: comment must have >= MIN_SENTS raw sentences.
  - One matched set PER DELTA (multi-delta threads contribute multiple sets).
  - Each delta gets OVERSAMPLE non-delta candidates, drawn WITHOUT replacement
    across sets within a thread (disjoint), randomly.
  - Threads/deltas that cannot field a full disjoint candidate pool are dropped.
  - Random within-thread matching (no covariate matching).

Output: selected_for_classification.jsonl — the comments to send to the cascade,
each tagged with set_id and role (delta / nondelta_candidate).
"""
import argparse, json, random
from collections import defaultdict

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="outputs/comments_segmented.jsonl")
    ap.add_argument("--output", default="outputs/selected_for_classification.jsonl")
    ap.add_argument("--min-sents", type=int, default=3, help="raw sentence proxy threshold")
    ap.add_argument("--oversample", type=int, default=6, help="non-delta candidates per delta")
    ap.add_argument("--keep", type=int, default=3, help="final controls per delta (post-salience)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--counts-json", default="consort/selection_counts.json")
    args = ap.parse_args()
    random.seed(args.seed)

    # Pass 1: bucket eligible comments by thread
    thread_deltas = defaultdict(list)
    thread_nondeltas = defaultdict(list)
    total = 0
    with open(args.input) as f:
        for line in f:
            r = json.loads(line)
            total += 1
            if len(r.get("sentences", [])) < args.min_sents:
                continue
            tid = r.get("thread_id")
            if r.get("y") == 1:
                thread_deltas[tid].append(r)
            elif r.get("y") == 0:
                thread_nondeltas[tid].append(r)

    # Pass 2: form disjoint sets
    out = []
    set_id = 0
    sets_formed = 0
    deltas_dropped_no_pool = 0
    for tid, deltas in thread_deltas.items():
        pool = list(thread_nondeltas.get(tid, []))
        random.shuffle(pool)
        random.shuffle(deltas)
        for d in deltas:
            if len(pool) < args.keep:
                deltas_dropped_no_pool += 1
                continue  # cannot field even the minimum controls
            take = min(args.oversample, len(pool))
            cands = [pool.pop() for _ in range(take)]
            set_id += 1
            sets_formed += 1
            drec = dict(d); drec["set_id"] = set_id; drec["role"] = "delta"
            out.append(drec)
            for c in cands:
                crec = dict(c); crec["set_id"] = set_id; crec["role"] = "nondelta_candidate"
                out.append(crec)

    with open(args.output, "w") as w:
        for r in out:
            w.write(json.dumps(r) + "\n")

    import os
    os.makedirs("consort", exist_ok=True)
    counts = {
        "total_comments_scanned": total,
        "sets_formed": sets_formed,
        "deltas_dropped_insufficient_pool": deltas_dropped_no_pool,
        "comments_selected_for_classification": len(out),
        "min_sents_proxy": args.min_sents,
        "oversample": args.oversample,
        "keep_final": args.keep,
        "seed": args.seed,
    }
    with open(args.counts_json, "w") as cf:
        json.dump(counts, cf, indent=2)
    print(json.dumps(counts, indent=2))

if __name__ == "__main__":
    main()
