# Intelligent Candidate Discovery & Ranking System
## India Runs Hackathon Submission

This system is an intelligent talent acquisition ranking engine built for the Redrob platform. It moves beyond keyword-based filtering (such as BM25) to identify the highest-fit, most engaged, and currently available candidates from a pool of 100,000 profiles against a job description (JD).

---

## 1. System Architecture

To process 100,000 candidates on CPU hardware within a strict 5-minute wall-clock time budget, the system implements a **Two-Stage Retrieval & Reranking** architecture:

```
                  ┌─────────────────────────────────────┐
                  │          Candidates JSONL           │
                  │        (100,000 Profiles)           │
                  └──────────────────┬──────────────────┘
                                     │
                                     ▼
                  ┌─────────────────────────────────────┐
                  │    Stage 1: Heuristic L1 Filter     │
                  │   - Fast exact skill overlap count  │
                  │   - Quick title keyword alignment   │
                  │   - Fast experience level checking  │
                  │   - Notice period & activity mult   │
                  │   - Honeypot timeline logic         │
                  └──────────────────┬──────────────────┘
                                     │
                                     ▼ (Pruned to Top 2,000)
                  ┌─────────────────────────────────────┐
                  │    Stage 2: Semantic Reranking      │
                  │   - SentenceTransformer Embeddings  │
                  │   - Cosine similarity calculation   │
                  │   - Seniority & trajectory scoring  │
                  │   - Recency & conversion modifier   │
                  └──────────────────┬──────────────────┘
                                     │
                                     ▼ (Sorted & Normalized)
                  ┌─────────────────────────────────────┐
                  │         Explainable Ranking         │
                  │   - Final deterministic ranking     │
                  │   - Verifiable reasoning builder    │
                  │   - Output: 100-row submission CSV  │
                  └─────────────────────────────────────┘
```

### Key Phases:
1. **JD Parser:** Parses unstructured job description text to extract experience requirements, preferred job titles, required technical skills, and nice-to-haves.
2. **L1 Heuristic Filter:** Streams candidate profiles and calculates a fast, non-embedding retrieval score to select the top 2,000 candidate profiles. Removes obvious mismatches and flags honeypots.
3. **L2 Semantic Scoring:** Batches text segments (titles/headline and skills list) of the top 2,000 candidates and embeds them using the `all-MiniLM-L6-v2` SentenceTransformer (22MB). Calculates exact semantic similarity weights.
4. **Behavioral Signal Modifier:** Computes a multiplicative factor based on Redrob engagement metrics (responsiveness, platform activity recency, notice period, interview completion rate).
5. **Deterministic Ranking & Reasoning:** Ranks profiles using strict tie-breaker criteria, normalizes scores into sequential percentiles, and builds factual, non-templated descriptions of candidate fits.

---

## 2. Scoring Methodology

### 2.1 Core Scoring Formula

$$\text{Final Score} = \left( \begin{aligned}
  &\text{Skills Match Score} \times 0.30 \\
  + &\text{Experience Relevance Score} \times 0.25 \\
  + &\text{Title Alignment Score} \times 0.15 \\
  + &\text{Education Fit Score} \times 0.10
\end{aligned} \right) \times \text{Behavioral Multiplier} \times \text{Honeypot Filter}$$

---

### 2.2 Component Scoring Breakdown

#### A. Skills Match Score (Weight: 0.30)
Evaluates candidate capabilities through exact keyword matching and semantic context.
- **Exact Match (70%):** Standardizes candidate skill names using an alias dictionary (e.g., standardizing `NLP` and `Natural Language Processing`) and counts occurrences of must-have JD skills.
- **Semantic Match:** Computes the cosine similarity between candidate skill lists and required skills using SentenceTransformers.
- **Proficiency Boost (30%):** Multiplies proficiency levels (`expert = 1.0`, `advanced = 0.75`, `intermediate = 0.5`, `beginner = 0.25`) by normalized endorsements count ($\frac{\text{endorsements}}{50}$) for required skills.

#### B. Experience Relevance Score (Weight: 0.25)
Combines quantitative experience matching with qualitative trajectory scoring.
- **Experience Level (65%):** Penalizes profiles below minimum required experience and maps years into a curve based on target and ideal experience bands.
- **Career Trajectory (35%):** Reverses career history chronologically and evaluates seniority progression (Junior $\rightarrow$ Senior $\rightarrow$ Lead) using title-specific seniority levels.

#### C. Title Alignment Score (Weight: 0.15)
Assesses job title relevance using exact matches and semantic similarity.
- **Exact Match:** Yields a perfect `1.0` if the current title matches the preferred list.
- **Semantic Similarity (50%):** Cosine similarity between candidate title/headline embedding and the primary JD title embedding.
- **Related Role Check (30%):** Maps candidate titles to related role keywords (e.g. `ML Scientist` for `Data Scientist`).
- **Seniority Match (20%):** Checks title keywords for senior vs junior signals to align with target role level.

#### D. Education Fit Score (Weight: 0.10)
Examines academic degree credentials, study field, and institution tier.
- **Degree Relevance (40%):** Ranks degrees (Ph.D/M.Tech/M.S. $\rightarrow$ B.Tech/B.E. $\rightarrow$ MBA $\rightarrow$ BA).
- **Field Match (35%):** Validates engineering, math, and statistics fields against CS/AI expectations.
- **Institution Tier (25%):** Incorporates institution tiers (`tier_1` down to `tier_4`).
- **Recency Bonus:** Multiplies the score based on graduation recency to boost fresh graduates for appropriate entry curves.

---

### 2.3 Redrob Behavioral Multiplier (Range: 0.4 - 1.2)
Hiring is about availability as much as skills. Redrob signals are aggregated into five weighted categories to scale the base score:
1. **Engagement Quality (35%):** Recruiter response rate thresholds (e.g. $>0.70$ receives a $1.1$ multiplier, while $<0.20$ receives $0.5$).
2. **Platform Recency (30%):** Multiplier based on the candidate's last active date (e.g. $<7$ days receives a $1.1$ boost, while $>90$ days gets a heavy $0.5$ penalty).
3. **Availability (15%):** Boosts candidates with an active `open_to_work_flag` ($+0.05$) or a short notice period ($<30$ days, $+0.05$).
4. **Conversion Probability (15%):** Boosts candidates with historical high interview completion rates ($>85\%$) and offer acceptance rates ($>70\%$).
5. **Market Validation (5%):** Small boost if views/saves/searches in the last 30 days are high ($>15$).

---

## 3. Honeypot Detection Filter
Candidates with logically inconsistent profiles are flagged. The engine verifies 5 logical rules:
1. **Timeline Check:** Stated experience years must match the sum of durations in career history (within 24 months).
2. **Improbable Durations:** Stated experience must not exceed 50 years.
3. **Skill Duration:** Candidates cannot claim "expert" proficiency in a skill with 0 months of use, or expert level with $<12$ months of use.
4. **Education Timeline:** Graduating end year must be consistent with the start year of their earliest employment (within 1 year buffer).
5. **Skill Explosion:** Profiles with $>40$ skills where $>60\%$ are marked "expert" are flagged.
6. **Entry-level Paradox:** Stated experience $<1$ year but listing $>15$ skills is flagged.

**Action:** $3+$ flags identify a profile as a honeypot, forcing its final score to $0.0$ and pushing it to the bottom.

---

## 4. Explainable Reasoning Strategy
To make recommendations transparent and verifiable, the engine dynamically constructs non-templated reasoning blocks (60-120 words) using facts checked directly against the candidate's profile.

- **Ranks 1-10 (Top Tier):** Emphasizes matched skills count, years of experience, positive engagement metrics (response rates), and recent activity on the platform.
- **Ranks 11-30 (Strong Fit):** Explains candidate strengths (e.g. high interview completion) while noting any minor trade-offs (e.g. notice period length).
- **Ranks 31-60 (Moderate Fit):** Provides a balanced view, acknowledging skill or availability gaps while highlighting compensating strengths (like strong GitHub scores).
- **Ranks 61-100 (Lower Tier):** Honestly highlights primary limitations (e.g., mismatched experience levels or low direct skill overlap) while mentioning minor positive details (such as relocation willingness).

---

## 5. Performance Characteristics
Benchmarks measured on CPU-only hardware:
- **Total Runtime:** ~75 seconds (well within the 300-second budget).
- **Peak Memory Usage:** ~1.4 GB (well below the 16 GB limit).
- **Execution Bottleneck Mitigation:** The L1 filter eliminates $98\%$ of expensive embedding computations, reducing SentenceTransformer processing time from $50$ minutes to under $15$ seconds.

---

## 6. How to Run & Reproduce

### 1. Install dependencies:
```bash
pip install -r requirements.txt
```

### 2. Run ranking pipeline:
```bash
python ranking_system.py \
  --candidates ../candidates.jsonl \
  --jd_path ../job_description.docx \
  --out submission.csv
```

### 3. Run validation scripts:
```bash
# Run local test suite
python test_script.py --output submission.csv

# Run official challenge validator
python ../validate_submission.py submission.csv
```
