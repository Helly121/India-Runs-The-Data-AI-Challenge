import json

with open("candidates.jsonl", "r", encoding="utf-8") as f:
    candidates = [json.loads(line) for line in f if line.strip()]

print(len(candidates))