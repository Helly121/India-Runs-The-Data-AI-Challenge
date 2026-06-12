#!/usr/bin/env python3
"""
Test Suite and Pre-Submission Verification Script
Validates formatting constraints, quality rules, and score properties of ranking output.
"""

import os
import sys
import csv
import re
import argparse
import pandas as pd

def test_data_integrity(df):
    """Verifies dimensions, columns, and IDs."""
    print("Running Data Integrity Checks...")
    
    # 1. Check Row Count
    assert len(df) == 100, f"Error: Output CSV must have exactly 100 rows, found {len(df)}."
    print("  [OK] Exactly 100 data rows.")
    
    # 2. Check Column Names
    expected_cols = ["candidate_id", "rank", "score", "reasoning"]
    assert list(df.columns) == expected_cols, f"Error: Columns must be exactly {expected_cols}, got {list(df.columns)}."
    print("  [OK] Correct columns and order.")
    
    # 3. Check Candidate ID formatting
    pattern = re.compile(r"^CAND_[0-9]{7}$")
    for cid in df["candidate_id"]:
        assert pattern.match(cid), f"Error: Candidate ID '{cid}' does not match format CAND_XXXXXXX."
    print("  [OK] Candidate IDs formatted correctly.")
    
    # 4. Check Uniqueness of Candidate IDs
    assert df["candidate_id"].nunique() == 100, "Error: Duplicate candidate IDs found in output."
    print("  [OK] All candidate IDs are unique.")

def test_score_and_rank_properties(df):
    """Verifies score ordering, range, decimals, and rank uniqueness."""
    print("Running Score and Rank Properties Checks...")
    
    # 1. Check Rank Uniqueness and Values
    ranks = df["rank"].tolist()
    assert len(set(ranks)) == 100, "Error: Ranks must be unique."
    assert set(ranks) == set(range(1, 101)), "Error: Ranks must cover 1 to 100 exactly."
    assert ranks == list(range(1, 101)), "Error: Ranks must be sorted sequentially 1 to 100."
    print("  [OK] Ranks are unique, sequential, and range 1-100.")
    
    # 2. Check Score Range [0, 1]
    scores = df["score"].tolist()
    assert all(0.0 <= s <= 1.0 for s in scores), "Error: Scores must be in range [0, 1]."
    print("  [OK] Scores are within [0, 1] range.")
    
    # 3. Check Monotonicity (Non-increasing scores)
    for i in range(len(scores) - 1):
        assert scores[i] >= scores[i+1], f"Error: Scores must be non-increasing. Index {i} ({scores[i]}) < Index {i+1} ({scores[i+1]})."
    print("  [OK] Scores are strictly non-increasing down the ranking.")
    
    # 4. Check Decimal Places
    for score in df["score"]:
        score_str = f"{score:.4f}"
        s_val = float(score_str)
        # Ensure it rounds properly
        assert abs(s_val - score) < 1e-6, f"Error: Score {score} has more than 4 decimal places."
    print("  [OK] Scores have exactly 4 decimal places format.")

def test_reasoning_quality(df):
    """Verifies that the reasoning column is present, formatted, and non-empty."""
    print("Running Reasoning Quality Checks...")
    
    empty_count = df[df["reasoning"].isna() | (df["reasoning"] == "")].shape[0]
    assert empty_count == 0, f"Error: Found {empty_count} blank reasoning entries."
    print("  [OK] All reasoning fields are non-empty.")
    
    # Check that they end with periods
    for idx, row in df.iterrows():
        reasoning = row["reasoning"].strip()
        assert reasoning.endswith("."), f"Error at rank {row['rank']}: Reasoning must end with a period."
        assert len(reasoning) > 30, f"Error at rank {row['rank']}: Reasoning is too short ({len(reasoning)} chars)."
        
    print("  [OK] All reasoning entries end with a period and meet minimum length.")

def main():
    parser = argparse.ArgumentParser(description="Ranking Output Verification Suite")
    parser.add_argument("--output", default="submission.csv", help="Path to output CSV to validate")
    args = parser.parse_args()
    
    if not os.path.exists(args.output):
        print(f"Error: Output file '{args.output}' does not exist.")
        sys.exit(1)
        
    try:
        df = pd.read_csv(args.output)
    except Exception as e:
        print(f"Error reading output CSV: {e}")
        sys.exit(1)
        
    print(f"Validating ranking output file: {args.output}")
    print("--------------------------------------------------")
    
    try:
        test_data_integrity(df)
        test_score_and_rank_properties(df)
        test_reasoning_quality(df)
        print("--------------------------------------------------")
        print("[SUCCESS] ALL LOCAL SANITY TESTS PASSED SUCCESSFULLY!")
        print("--------------------------------------------------")
    except AssertionError as e:
        print("--------------------------------------------------")
        print(f"[FAIL] TEST SUITE FAILED: {e}")
        print("--------------------------------------------------")
        sys.exit(1)

if __name__ == "__main__":
    main()
