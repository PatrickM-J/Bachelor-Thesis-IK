"""Flatten selected comments into cascade input (sentence-level), sharded.

Each output record: {sentence_text, comment_id, thread_id, parent_id, author,
created_utc, score, level, y, set_id, role, sentence_idx}. The cascade preserves
all non-'sentence_text' fields in its output, so these ride through to aggregation.

Output: shards of a JSON array (cascade does json.load per file), named
sentences_shard_000.json, sentences_shard_001.json, ...
"""
import argparse, json, os

PASSTHROUGH = ["comment_id","thread_id","parent_id","author","created_utc",
               "score","level","y","set_id","role"]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="outputs/selected_for_classification.jsonl")
    ap.add_argument("--outdir", default="outputs/cascade_input")
    ap.add_argument("--shard-size", type=int, default=200000, help="sentences per shard")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    shard = []
    shard_idx = 0
    total_sents = 0

    def flush(buf, idx):
        p = os.path.join(args.outdir, f"sentences_shard_{idx:03d}.json")
        with open(p, "w") as f:
            json.dump(buf, f, ensure_ascii=False)
        print(f"wrote {p}  ({len(buf)} sentences)")

    with open(args.input) as f:
        for line in f:
            r = json.loads(line)
            base = {k: r.get(k) for k in PASSTHROUGH}
            for i, sent in enumerate(r.get("sentences", [])):
                if not sent or not sent.strip():
                    continue
                rec = dict(base)
                rec["sentence_text"] = sent
                rec["sentence_idx"] = i
                shard.append(rec)
                total_sents += 1
                if len(shard) >= args.shard_size:
                    flush(shard, shard_idx); shard_idx += 1; shard = []
    if shard:
        flush(shard, shard_idx); shard_idx += 1

    print(f"\nTotal sentences: {total_sents:,} across {shard_idx} shards")

if __name__ == "__main__":
    main()
