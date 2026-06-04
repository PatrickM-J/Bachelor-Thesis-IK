"""Stage C: aggregate cascade predictions to comment level, apply salience
filter, finalize 1:3 matching, attach controls. Produces the analysis dataset.

Decisions baked in:
  - far-proportion denominator = SALIENT sentences (non-salient are N/A, excluded)
  - salience filter: keep comments with >= MIN_SALIENT salient sentences
  - per set_id: delta must survive; keep KEEP surviving non-delta controls;
    drop sets that can't field KEEP controls
  - controls: comment_length, is_top_level, comment_order, author_activity
"""
import argparse, json, glob, os
from collections import defaultdict
import pandas as pd

def _to_num(v):
    """Coerce created_utc to float for sorting; unconvertible -> +inf (sorts last)."""
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("inf")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-glob", default="outputs/cascade_output/pred_sentences_shard_*.json")
    ap.add_argument("--segmented", default="outputs/comments_segmented.jsonl",
                    help="full segmented file, for corpus-wide author activity + comment_order")
    ap.add_argument("--min-salient", type=int, default=3)
    ap.add_argument("--keep", type=int, default=3)
    ap.add_argument("--out-parquet", default="outputs/analysis_dataset.parquet")
    ap.add_argument("--out-csv", default="outputs/analysis_dataset.csv")
    ap.add_argument("--counts-json", default="consort/aggregate_counts.json")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    # ---- corpus-wide author activity + per-thread comment order (full file) ----
    author_count = defaultdict(int)
    thread_times = defaultdict(list)
    with open(args.segmented) as f:
        for line in f:
            r = json.loads(line)
            a = r.get("author")
            if a and a != "[deleted]":
                author_count[a] += 1
            thread_times[r.get("thread_id")].append((r.get("created_utc"), r.get("comment_id")))
    comment_order = {}
    for tid, lst in thread_times.items():
        for rank, (_, cid) in enumerate(sorted(lst, key=lambda x: _to_num(x[0])), start=1):
            comment_order[cid] = rank

    # ---- group predictions by comment ----
    comm = {}
    for path in sorted(glob.glob(args.pred_glob)):
        with open(path) as f:
            recs = json.load(f)
        for r in recs:
            cid = r["comment_id"]
            c = comm.setdefault(cid, {
                "comment_id": cid, "thread_id": r.get("thread_id"),
                "parent_id": r.get("parent_id"), "author": r.get("author"),
                "created_utc": r.get("created_utc"), "y": r.get("y"),
                "set_id": r.get("set_id"), "role": r.get("role"),
                "n_sent": 0, "n_salient": 0, "far_social": 0, "far_hypo": 0,
            })
            c["n_sent"] += 1
            if r.get("salient"):
                c["n_salient"] += 1
                if r.get("social") == 1: c["far_social"] += 1
                if r.get("hypothetical") == 1: c["far_hypo"] += 1

    # ---- per-comment variables + salience filter ----
    rows = []
    for cid, c in comm.items():
        if c["n_salient"] < args.min_salient:
            continue
        rows.append({
            "comment_id": cid, "thread_id": c["thread_id"], "set_id": c["set_id"],
            "role": c["role"], "y": c["y"],
            "prop_far_social": c["far_social"] / c["n_salient"],
            "prop_far_hypothetical": c["far_hypo"] / c["n_salient"],
            "comment_length": c["n_sent"], "n_salient": c["n_salient"],
            "is_top_level": 1 if str(c["parent_id"]).startswith("t3_") else 0,
            "comment_order": comment_order.get(cid),
            "author_activity": author_count.get(c["author"], 0),
        })
    df = pd.DataFrame(rows)

    # ---- finalize matching per set_id ----
    keep_rows = []
    sets_kept = sets_dropped_no_delta = sets_dropped_few_controls = 0
    for sid, g in df.groupby("set_id"):
        deltas = g[g["role"] == "delta"]
        controls = g[g["role"] == "nondelta_candidate"]
        if len(deltas) == 0:
            sets_dropped_no_delta += 1; continue
        if len(controls) < args.keep:
            sets_dropped_few_controls += 1; continue
        chosen = controls.sample(n=args.keep, random_state=args.seed)
        keep_rows.append(deltas); keep_rows.append(chosen); sets_kept += 1

    final = pd.concat(keep_rows, ignore_index=True) if keep_rows else pd.DataFrame()
    os.makedirs("consort", exist_ok=True)
    final.to_parquet(args.out_parquet, index=False)
    final.to_csv(args.out_csv, index=False)

    counts = {
        "comments_after_salience_filter": len(df),
        "sets_kept": sets_kept,
        "sets_dropped_no_surviving_delta": sets_dropped_no_delta,
        "sets_dropped_insufficient_controls": sets_dropped_few_controls,
        "final_rows": len(final),
        "final_deltas": int((final["y"] == 1).sum()) if len(final) else 0,
        "final_nondeltas": int((final["y"] == 0).sum()) if len(final) else 0,
        "min_salient": args.min_salient, "keep": args.keep, "seed": args.seed,
    }
    with open(args.counts_json, "w") as cf:
        json.dump(counts, cf, indent=2)
    print(json.dumps(counts, indent=2))

if __name__ == "__main__":
    main()
