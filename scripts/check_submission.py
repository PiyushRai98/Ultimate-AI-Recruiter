"""Quick validation script to check submission quality."""

import csv
from pathlib import Path

import orjson


def main():
    rows = list(csv.DictReader(open("submission.csv")))
    print(f"Rows: {len(rows)}")
    print(f"Score range: {rows[0]['score']} - {rows[-1]['score']}")

    # Check monotonicity
    scores = [float(r["score"]) for r in rows]
    is_monotone = all(scores[i] >= scores[i + 1] for i in range(len(scores) - 1))
    print(f"Non-increasing: {is_monotone}")

    # Reasoning uniqueness
    reasonings = [r["reasoning"] for r in rows]
    print(f"Unique reasonings: {len(set(reasonings))}/100")
    print(f"Empty reasonings: {sum(1 for r in reasonings if not r.strip())}")

    # Load candidate titles
    cand_map = {}
    with open("data/raw/candidates.jsonl") as f:
        for line in f:
            d = orjson.loads(line)
            cand_map[d["candidate_id"]] = d["profile"]["current_title"]

    print("\nTop 10:")
    for r in rows[:10]:
        title = cand_map.get(r["candidate_id"], "?")
        print(f"  #{r['rank']}: {title} (score={r['score']})")

    # Check for honeypots (non-tech titles in top 100)
    bad_titles = ["marketing", "accountant", "hr manager", "sales", "operations",
                  "content writer", "graphic designer", "customer support",
                  "civil engineer", "mechanical engineer"]
    honeypot_count = 0
    for r in rows:
        title = cand_map.get(r["candidate_id"], "").lower()
        if any(bt in title for bt in bad_titles):
            honeypot_count += 1
            print(f"  WARNING: Rank {r['rank']}: {title}")

    print(f"\nPotential honeypots in top-100: {honeypot_count}")


if __name__ == "__main__":
    main()
