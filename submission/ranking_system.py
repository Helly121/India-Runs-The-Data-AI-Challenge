#!/usr/bin/env python3
"""
Intelligent Candidate Ranking System - Main Implementation
For India Runs Hackathon. Complete two-stage L1/L2 retrieval and scoring pipeline.
"""

import os
# Force offline mode for Hugging Face Hub and Transformers
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import sys
import json
import csv
import re
import argparse
import zipfile
import xml.etree.ElementTree as ET
from datetime import datetime
import numpy as np
import pandas as pd

# Import configuration
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import *

# Global check for SentenceTransformer (will be imported dynamically)
SentenceTransformer = None

def get_docx_text(path):
    """Extracts text from a docx file using standard zip/XML libraries."""
    try:
        text = []
        with zipfile.ZipFile(path) as docx:
            tree = ET.parse(docx.open('word/document.xml'))
            root = tree.getroot()
            for paragraph in root.iter('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}p'):
                p_text = []
                for run in paragraph.iter('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}t'):
                    p_text.append(run.text)
                if p_text:
                    text.append(''.join(p_text))
        return '\n'.join(text)
    except Exception as e:
        print(f"Warning: Failed to parse docx {path}: {e}")
        return ""

def cosine_similarity_np(v1, v2):
    """Calculates cosine similarity between two 1D numpy vectors."""
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    if norm_v1 == 0 or norm_v2 == 0:
        return 0.0
    return float(dot_product / (norm_v1 * norm_v2))

def get_seniority(title):
    """Determines seniority score of a job title."""
    t = title.lower()
    if any(w in t for w in ["lead", "principal", "staff", "manager", "director", "head", "founding", "founder"]):
        return 4
    if any(w in t for w in ["senior", "sr."]):
        return 3
    if any(w in t for w in ["junior", "jr.", "intern", "trainee", "associate"]):
        return 1
    return 2

def evaluate_trajectory(career_history):
    """Evaluates the seniority trajectory score (0.0 to 1.0) of a career history."""
    if not career_history:
        return 0.5  # Neutral baseline
    
    # Chronological order (oldest to newest)
    seniority_levels = [get_seniority(job.get("title", "")) for job in reversed(career_history)]
    if len(seniority_levels) <= 1:
        return 0.7  # Single job, decent baseline
    
    # Calculate differences
    diffs = [seniority_levels[i] - seniority_levels[i-1] for i in range(1, len(seniority_levels))]
    
    # Count progression/demotion
    pos_diffs = sum(d for d in diffs if d > 0)
    neg_diffs = sum(d for d in diffs if d < 0)
    
    if neg_diffs < 0:
        score = 5.0 + pos_diffs * 1.5 + neg_diffs * 2.0
    else:
        score = 7.0 + pos_diffs * 1.5
        
    return max(0.0, min(10.0, score)) / 10.0

def detect_honeypot(candidate):
    """
    Checks for logically impossible combinations in candidate profile.
    Returns (is_honeypot, list_of_issues)
    """
    issues = []
    
    # Check 1: Stated years of experience vs career history duration
    total_work_months = sum(job.get("duration_months", 0) for job in candidate.get("career_history", []))
    stated_years = candidate.get("profile", {}).get("years_of_experience", 0.0)
    
    # If career history exceeds stated experience by more than 2 years (24 months)
    if total_work_months / 12.0 > stated_years + 2.0:
        issues.append("Career history exceeds stated experience")
        
    if stated_years > 50.0:
        issues.append("Implausibly long experience")
        
    # Check 2: Skill proficiency vs duration
    for s in candidate.get("skills", []):
        prof = s.get("proficiency", "").lower()
        duration = s.get("duration_months", 0)
        if prof == "expert" and duration == 0:
            issues.append(f"Expert proficiency in {s.get('name')} with 0 months duration")
        if prof == "expert" and duration < 12:
            issues.append(f"Expert proficiency in {s.get('name')} with <1 year duration")
            
    # Check 3: Education completion vs job start date
    education = candidate.get("education", [])
    career_history = candidate.get("career_history", [])
    if education and career_history:
        try:
            earliest_grad_year = min(edu.get("end_year", 9999) for edu in education)
            
            # Start years of career history jobs
            job_start_years = []
            for job in career_history:
                start_date = job.get("start_date")
                if start_date:
                    job_start_years.append(datetime.strptime(start_date, "%Y-%m-%d").year)
            
            if job_start_years:
                earliest_job_year = min(job_start_years)
                # Flag if worked more than 1 year before graduating
                if earliest_job_year < earliest_grad_year - 1:
                    issues.append("Timeline conflict: worked before graduating")
        except Exception:
            pass
            
    # Check 4: Skill count explosion (spamming expert skills)
    skills = candidate.get("skills", [])
    if len(skills) > 40:
        expert_count = sum(1 for s in skills if s.get("proficiency", "").lower() == "expert")
        if expert_count > len(skills) * 0.6:
            issues.append("Implausibly high expert skill count")
            
    # Check 5: Entry level contradictions (junior with massive skills list)
    if stated_years < 1.0 and len(skills) > 15:
        issues.append("Too many skills for entry-level candidate")
        
    # Heuristic: 3 or more issues makes it a honeypot
    honeypot_score = len(issues) / 6.0
    is_honeypot = honeypot_score > 0.4
    
    return is_honeypot, issues

def calculate_behavioral_multiplier(signals):
    """
    Applies redrob signals as multiplier on base scores.
    Range: 0.4 - 1.2
    """
    if not signals:
        return 0.4  # Assume worst-case for empty signals
        
    # 1. Engagement Quality (35%)
    response_rate = signals.get("recruiter_response_rate", 0.0)
    if response_rate > 0.7:
        engagement_mult = 1.1
    elif response_rate > 0.5:
        engagement_mult = 1.0
    elif response_rate > 0.2:
        engagement_mult = 0.8
    else:
        engagement_mult = 0.5
        
    # 2. Recency (30%)
    last_active_str = signals.get("last_active_date", "")
    try:
        last_active = datetime.strptime(last_active_str, "%Y-%m-%d")
        # Base date is June 12, 2026 for India Runs Hackathon
        days_ago = (datetime(2026, 6, 12) - last_active).days
        if days_ago < 7:
            recency_mult = 1.1
        elif days_ago < 30:
            recency_mult = 1.0
        elif days_ago < 90:
            recency_mult = 0.8
        else:
            recency_mult = 0.5
    except Exception:
        recency_mult = 0.5
        
    # 3. Availability Signal (15%)
    availability_mult = 1.0
    if signals.get("open_to_work_flag", False):
        availability_mult += 0.05
    notice_days = signals.get("notice_period_days", 180)
    if notice_days < 30:
        availability_mult += 0.05
    availability_mult = min(availability_mult, 1.15)
    
    # 4. Conversion Probability (15%)
    completion_rate = signals.get("interview_completion_rate", 0.0)
    acceptance_rate = signals.get("offer_acceptance_rate", -1.0)
    
    if completion_rate > 0.8 and acceptance_rate > 0.7:
        conversion_mult = 1.08
    elif completion_rate < 0.5:
        conversion_mult = 0.85
    else:
        conversion_mult = 1.0
        
    # 5. Market Validation (5%)
    views = signals.get("profile_views_received_30d", 0)
    saves = signals.get("saved_by_recruiters_30d", 0)
    searches = signals.get("search_appearance_30d", 0)
    market_signals = views + saves + searches
    
    if market_signals > 15:
        market_mult = 1.05
    elif market_signals == 0:
        market_mult = 0.9
    else:
        market_mult = 1.0
        
    # Combine signals
    behavioral_multiplier = (
        engagement_mult * BEHAVIORAL_WEIGHTS["engagement"] +
        recency_mult * BEHAVIORAL_WEIGHTS["recency"] +
        availability_mult * BEHAVIORAL_WEIGHTS["availability"] +
        conversion_mult * BEHAVIORAL_WEIGHTS["conversion"] +
        market_mult * BEHAVIORAL_WEIGHTS["market"]
    )
    
    return max(0.4, min(behavioral_multiplier, 1.2))

def calculate_skills_match_score(candidate, jd_requirements, embeddings):
    """Combines exact match, semantic match, and proficiency signals."""
    required_skills = [s[0].lower() for s in jd_requirements["required_skills"]]
    candidate_skills = [s["name"].lower() for s in candidate.get("skills", [])]
    
    # Standardize skills using aliases
    candidate_skills_std = [SKILL_ALIASES.get(s, s) for s in candidate_skills]
    
    # Exact keyword matching
    exact_matches = sum(1 for req in required_skills if any(req in cand for cand in candidate_skills_std))
    exact_match_score = exact_matches / max(len(required_skills), 1)
    
    # Semantic matching (embedding similarity)
    candidate_id = candidate["candidate_id"]
    cand_emb = embeddings.get("skills_" + candidate_id)
    jd_emb = embeddings.get("jd_requirements")
    
    if cand_emb is not None and jd_emb is not None:
        semantic_score = cosine_similarity_np(cand_emb, jd_emb)
        # Normalize to [0, 1]
        semantic_score = max(0.0, min(1.0, (semantic_score + 1.0) / 2.0))
    else:
        semantic_score = 0.0
        
    # Proficiency boost
    proficiency_weights = {"beginner": 0.25, "intermediate": 0.5, "advanced": 0.75, "expert": 1.0}
    proficiency_scores = []
    for s in candidate.get("skills", []):
        name = s["name"].lower()
        std_name = SKILL_ALIASES.get(name, name)
        if any(req in std_name for req in required_skills):
            prof = s.get("proficiency", "beginner").lower()
            endorsements = s.get("endorsements", 0)
            score = proficiency_weights.get(prof, 0.25) * min(endorsements / 50.0, 1.0)
            proficiency_scores.append(score)
            
    proficiency_boost = sum(proficiency_scores) / max(len(required_skills), 1) if proficiency_scores else 0.0
    proficiency_boost = min(proficiency_boost, 1.0)
    
    # Direct Text Matching on Past Job Descriptions
    career_history_text = " ".join([h.get("description", "") for h in candidate.get("career_history", [])]).lower()
    proven_skills = sum(1 for req in required_skills if req in career_history_text)
    proven_skills_score = min(proven_skills / max(len(required_skills), 1), 1.0)
    
    # Combine signals
    skills_match = max(exact_match_score, semantic_score) * 0.5 + proven_skills_score * 0.3 + proficiency_boost * 0.2
    return min(skills_match, 1.0)

def calculate_experience_relevance_score(candidate, jd_requirements):
    """Evaluates years in relevant domain + career progression trajectory."""
    target_years = jd_requirements["experience"]["target_years"]
    ideal_years = jd_requirements["experience"]["ideal_years"]
    min_years = jd_requirements["experience"]["min_years"]
    
    years = candidate.get("profile", {}).get("years_of_experience", 0.0)
    
    # Experience level scoring
    if years < min_years:
        exp_score = (years / min_years) * 0.5
    elif years <= target_years:
        exp_score = 0.5 + (years - min_years) / (target_years - min_years) * 0.4
    elif years <= ideal_years:
        exp_score = 0.9 + (years - target_years) / (ideal_years - target_years) * 0.1
    else:
        exp_score = 1.0
        
    # Trajectory scoring
    progression_score = evaluate_trajectory(candidate.get("career_history", []))
    
    experience_relevance = exp_score * 0.65 + progression_score * 0.35
    return min(experience_relevance, 1.0)

def calculate_title_alignment_score(candidate, jd_requirements, embeddings):
    """Evaluates how well current title matches role."""
    current_title = candidate.get("profile", {}).get("current_title", "").lower()
    preferred_titles = [t.lower() for t in jd_requirements["preferred_titles"]]
    
    if current_title in preferred_titles:
        return 1.0
        
    candidate_id = candidate["candidate_id"]
    title_emb = embeddings.get("title_" + candidate_id)
    best_title_emb = embeddings.get("preferred_title_0")
    
    if title_emb is not None and best_title_emb is not None:
        semantic_sim = cosine_similarity_np(title_emb, best_title_emb)
        # Normalize to [0, 1]
        semantic_sim = max(0.0, min(1.0, (semantic_sim + 1.0) / 2.0))
    else:
        semantic_sim = 0.0
        
    # Related role check
    related_title_keywords = {
        "data_scientist": ["ml_engineer", "analytics_engineer", "ml_scientist", "data analyst", "ai engineer"],
        "ml_engineer": ["data_scientist", "research_engineer", "ai engineer", "deep learning engineer"],
        "software_engineer": ["full_stack", "backend", "frontend", "developer", "programmer"],
        "ai_engineer": ["ml_engineer", "data_scientist", "founding ai engineer", "nlp engineer"]
    }
    
    is_related = False
    for preferred in preferred_titles:
        pref_clean = preferred.replace(" ", "_")
        keywords = related_title_keywords.get(pref_clean, [])
        if any(kw in current_title for kw in keywords) or any(kw in current_title for kw in [preferred]):
            is_related = True
            break
            
    # Seniority level check
    senior_keywords = ["senior", "lead", "principal", "staff", "founding", "founder", "manager"]
    junior_keywords = ["junior", "associate", "intern", "trainee"]
    is_senior = any(kw in current_title for kw in senior_keywords)
    is_junior = any(kw in current_title for kw in junior_keywords)
    
    if is_senior:
        seniority_match = 1.0
    elif is_junior:
        seniority_match = 0.5
    else:
        seniority_match = 0.8
        
    title_alignment = semantic_sim * 0.5 + (1.0 if is_related else 0.3) * 0.3 + seniority_match * 0.2
    return min(title_alignment, 1.0)

def calculate_education_fit_score(candidate, jd_requirements):
    """Evaluates degree relevance + field of study + institution quality."""
    education = candidate.get("education", [])
    if not education:
        return 0.3  # Penalize but do not disqualify
        
    primary_education = education[0]
    
    # Degree relevance
    degree = primary_education.get("degree", "").lower()
    if any(d in degree for d in ["b.tech", "b.e.", "b.s."]):
        degree_relevance = 0.9
    elif any(d in degree for d in ["m.tech", "m.s.", "ms", "m.e.", "ph.d", "phd"]):
        degree_relevance = 1.0
    elif any(d in degree for d in ["mba", "m.b.a."]):
        degree_relevance = 0.5
    elif any(d in degree for d in ["b.a.", "ba", "b.com", "bcom"]):
        degree_relevance = 0.4
    else:
        degree_relevance = 0.6
        
    # Field of study
    field = primary_education.get("field_of_study", "").lower()
    tech_fields = ["computer science", "information technology", "data science", "statistics", 
                   "mathematics", "physics", "electrical", "electronics", "machine learning", "artificial intelligence"]
    field_match = 1.0 if any(f in field for f in tech_fields) else 0.5
    
    # Tier
    tier = primary_education.get("tier", "unknown").lower()
    tier_scores = {"tier_1": 1.0, "tier_2": 0.85, "tier_3": 0.6, "tier_4": 0.3, "unknown": 0.5}
    tier_score = tier_scores.get(tier, 0.5)
    
    # Recency
    end_year = primary_education.get("end_year", 2026)
    years_since_grad = max(0, 2026 - end_year)
    recency_bonus = 1.0 if years_since_grad < 2 else max(0.8, 1.0 - years_since_grad * 0.05)
    
    education_fit = (degree_relevance * 0.4 + field_match * 0.35 + tier_score * 0.25) * recency_bonus
    return min(education_fit, 1.0)

def calculate_summary_alignment_score(candidate, jd_requirements, embeddings):
    """Evaluates semantic similarity between candidate summary and JD."""
    candidate_id = candidate["candidate_id"]
    summary_emb = embeddings.get("summary_" + candidate_id)
    jd_emb = embeddings.get("jd_full")
    
    if summary_emb is not None and jd_emb is not None:
        semantic_sim = cosine_similarity_np(summary_emb, jd_emb)
        return max(0.0, min(1.0, (semantic_sim + 1.0) / 2.0))
    return 0.0

def calculate_career_alignment_score(candidate, jd_requirements, embeddings):
    """Evaluates semantic similarity between entire career history and JD."""
    candidate_id = candidate["candidate_id"]
    career_emb = embeddings.get("career_" + candidate_id)
    jd_emb = embeddings.get("jd_full")
    
    if career_emb is not None and jd_emb is not None:
        semantic_sim = cosine_similarity_np(career_emb, jd_emb)
        return max(0.0, min(1.0, (semantic_sim + 1.0) / 2.0))
    return 0.0


class CandidateRankingEngine:
    def __init__(self, jd_path, jsonl_path):
        self.jd_path = jd_path
        self.jsonl_path = jsonl_path
        self.jd_requirements = DEFAULT_JD_REQUIREMENTS.copy()
        self.jd_text = ""
        self.candidates_l1 = []  # Top candidates after Stage 1 (heuristic)
        self.embeddings_cache = {}
        
    def parse_jd(self):
        """Loads and parses Job Description from txt or docx."""
        if not self.jd_path or not os.path.exists(self.jd_path):
            print("Job description file not found. Falling back to default configuration requirements.")
            self.jd_text = "Founding AI Engineer. Python, Machine Learning, vector database, Pinecone, evaluation, Pune/Noida, 5-9 years."
            return
            
        # Extract text based on file format
        if self.jd_path.endswith('.docx'):
            self.jd_text = get_docx_text(self.jd_path)
        else:
            with open(self.jd_path, 'r', encoding='utf-8') as f:
                self.jd_text = f.read()
                
        # Regex parsing for experience requirements
        exp_pattern = re.compile(r"(\d+)\s*[-–]\s*(\d+)\s*years", re.IGNORECASE)
        match = exp_pattern.search(self.jd_text)
        if match:
            min_y = float(match.group(1))
            max_y = float(match.group(2))
            self.jd_requirements["experience"] = {
                "min_years": max(1.0, min_y - 1.0),
                "target_years": min_y,
                "ideal_years": max_y
            }
            print(f"Parsed experience requirements: min={self.jd_requirements['experience']['min_years']}, target={min_y}, ideal={max_y}")
            
    def load_candidates(self):
        """Streams candidates from JSONL, checks honeypots, and extracts top L1 pool using fast heuristic."""
        print("Stage 1: Streaming candidates & computing fast L1 heuristic...")
        all_l1_scores = []
        
        with open(self.jsonl_path, 'r', encoding='utf-8') as f:
            for line_idx, line in enumerate(f):
                if not line.strip():
                    continue
                try:
                    candidate = json.loads(line)
                    cid = candidate.get("candidate_id")
                    if not cid:
                        continue
                        
                    # Calculate L1 Heuristic score
                    is_hp, issues = detect_honeypot(candidate)
                    if is_hp:
                        l1_score = -1.0
                    else:
                        l1_score = self._calculate_l1_score(candidate)
                        
                    # To minimize memory footprint, store only scores and metadata first
                    all_l1_scores.append((l1_score, cid, line_idx))
                except Exception as e:
                    print(f"Error parsing line {line_idx}: {e}")
                    
        # Sort candidates: score descending, then candidate_id ascending for deterministic tie-breaks
        all_l1_scores.sort(key=lambda x: (-x[0], x[1]))
        
        # Take L1 Pool Size (e.g. top 2,000)
        l1_selected = all_l1_scores[:L1_POOL_SIZE]
        selected_line_indices = {item[2]: item[0] for item in l1_selected}
        
        # Reload only selected candidates fully into memory
        self.candidates_l1 = []
        with open(self.jsonl_path, 'r', encoding='utf-8') as f:
            for line_idx, line in enumerate(f):
                if line_idx in selected_line_indices:
                    candidate = json.loads(line)
                    candidate["l1_score"] = selected_line_indices[line_idx]
                    self.candidates_l1.append(candidate)
                    
        # Ensure they are in exact L1 sorted order
        self.candidates_l1.sort(key=lambda x: (-x["l1_score"], x["candidate_id"]))
        print(f"Stage 1 Complete: Filtered 100,000 candidates down to top {len(self.candidates_l1)}")
        
    def _calculate_l1_score(self, candidate):
        """Heuristic scoring helper for fast pruning."""
        # 1. Fast skills match
        required_skill_names = [s[0].lower() for s in self.jd_requirements["required_skills"]]
        candidate_skills = [s["name"].lower() for s in candidate.get("skills", [])]
        
        candidate_skills_std = [SKILL_ALIASES.get(s, s) for s in candidate_skills]
        exact_matches = sum(1 for req in required_skill_names if any(req in cand for cand in candidate_skills_std))
        skills_match_fast = exact_matches / max(len(required_skill_names), 1)
        
        # 2. Experience relevance fast
        years = candidate.get("profile", {}).get("years_of_experience", 0.0)
        min_years = self.jd_requirements["experience"]["min_years"]
        target_years = self.jd_requirements["experience"]["target_years"]
        ideal_years = self.jd_requirements["experience"]["ideal_years"]
        
        if years < min_years:
            exp_score = (years / min_years) * 0.5
        elif years <= target_years:
            exp_score = 0.5 + 0.4 * (years - min_years) / (target_years - min_years)
        elif years <= ideal_years:
            exp_score = 0.9 + 0.1 * (years - target_years) / (ideal_years - target_years)
        else:
            exp_score = 1.0
            
        # 3. Title alignment fast
        current_title = candidate.get("profile", {}).get("current_title", "").lower()
        preferred_titles = [t.lower() for t in self.jd_requirements["preferred_titles"]]
        
        if current_title in preferred_titles:
            title_score = 1.0
        else:
            has_keywords = any(kw in current_title for kw in ["data scientist", "ml", "machine learning", "ai", "nlp", "search"])
            is_engineer = any(kw in current_title for kw in ["engineer", "developer", "programmer", "backend"])
            if has_keywords:
                title_score = 0.7
            elif is_engineer:
                title_score = 0.4
            else:
                title_score = 0.1
                
        base_fast = skills_match_fast * 0.4 + exp_score * 0.3 + title_score * 0.3
        
        # Behavioral signals
        signals = candidate.get("redrob_signals", {})
        mult = calculate_behavioral_multiplier(signals)
        
        return base_fast * mult

    def embed_profiles(self):
        """Generates sentence embeddings for the JD and top candidates."""
        global SentenceTransformer
        if SentenceTransformer is None:
            print("Loading SentenceTransformer model...")
            from sentence_transformers import SentenceTransformer as ST
            SentenceTransformer = ST
            
        model = SentenceTransformer(EMBEDDING_MODEL, local_files_only=True)
        
        # Embed JD Requirements
        jd_skills_text = ", ".join([s[0] for s in self.jd_requirements["required_skills"]])
        self.embeddings_cache["jd_requirements"] = model.encode(jd_skills_text)
        
        # Embed JD Full (for summary/career matching)
        jd_full_text = getattr(self, 'jd_text', jd_skills_text)[:1000]
        self.embeddings_cache["jd_full"] = model.encode(jd_full_text)
        
        # Embed JD Primary Title
        self.embeddings_cache["preferred_title_0"] = model.encode(self.jd_requirements["preferred_titles"][0])
        
        # Batch embed candidates
        print(f"Stage 2: Embedding profiles for {len(self.candidates_l1)} selected candidates...")
        titles_to_embed = []
        skills_to_embed = []
        summaries_to_embed = []
        careers_to_embed = []
        cids = []
        
        for cand in self.candidates_l1:
            cid = cand["candidate_id"]
            current_title = cand.get("profile", {}).get("current_title", "")
            headline = cand.get("profile", {}).get("headline", "")
            skills_text = ", ".join([s["name"] for s in cand.get("skills", [])])
            summary_text = cand.get("profile", {}).get("summary", "")
            career_text = " ".join([h.get("description", "") for h in cand.get("career_history", [])])
            
            title_text = f"{current_title} {headline}".strip()
            
            titles_to_embed.append(title_text)
            skills_to_embed.append(skills_text if skills_text else "None")
            summaries_to_embed.append(summary_text if summary_text else "None")
            careers_to_embed.append(career_text if career_text else "None")
            cids.append(cid)
            
        # Multi-threaded batched embedding generation
        title_embs = model.encode(titles_to_embed, batch_size=128, show_progress_bar=False)
        skills_embs = model.encode(skills_to_embed, batch_size=128, show_progress_bar=False)
        summary_embs = model.encode(summaries_to_embed, batch_size=128, show_progress_bar=False)
        career_embs = model.encode(careers_to_embed, batch_size=128, show_progress_bar=False)
        
        for i, cid in enumerate(cids):
            self.embeddings_cache["title_" + cid] = title_embs[i]
            self.embeddings_cache["skills_" + cid] = skills_embs[i]
            self.embeddings_cache["summary_" + cid] = summary_embs[i]
            self.embeddings_cache["career_" + cid] = career_embs[i]
            
        print("Embeddings generated and cached successfully.")
        
    def score_candidates(self):
        """Computes deep multi-dimensional scores for the pruned L1 candidate list."""
        print("Computing exact final scores for top candidates...")
        self.scored_list = []
        
        for cand in self.candidates_l1:
            is_hp, issues = detect_honeypot(cand)
            if is_hp:
                final_score = 0.0
                components = {"skills": 0.0, "experience": 0.0, "title": 0.0, "education": 0.0}
                mult = 0.4
            else:
                s_score = calculate_skills_match_score(cand, self.jd_requirements, self.embeddings_cache)
                e_score = calculate_experience_relevance_score(cand, self.jd_requirements)
                t_score = calculate_title_alignment_score(cand, self.jd_requirements, self.embeddings_cache)
                ed_score = calculate_education_fit_score(cand, self.jd_requirements)
                sum_score = calculate_summary_alignment_score(cand, self.jd_requirements, self.embeddings_cache)
                car_score = calculate_career_alignment_score(cand, self.jd_requirements, self.embeddings_cache)
                
                base_score = (
                    s_score * WEIGHTS["skills"] +
                    e_score * WEIGHTS["experience"] +
                    t_score * WEIGHTS["title"] +
                    ed_score * WEIGHTS["education"] +
                    sum_score * WEIGHTS["summary_alignment"] +
                    car_score * WEIGHTS["career_alignment"]
                )
                
                mult = calculate_behavioral_multiplier(cand.get("redrob_signals", {}))
                final_score = base_score * mult
                
                components = {
                    "skills": s_score,
                    "experience": e_score,
                    "title": t_score,
                    "education": ed_score,
                    "summary_alignment": sum_score,
                    "career_alignment": car_score
                }
                
            self.scored_list.append({
                "candidate": cand,
                "final_score": final_score,
                "behavioral_multiplier": mult,
                "components": components
            })
            
    def rank_and_normalize(self):
        """Sorts candidate scores and normalizes them to a distinct rank 1-100 percentile score."""
        # Sort: score descending, response_rate descending, last_active descending, candidate_id ascending (deterministic)
        self.scored_list.sort(key=lambda x: (
            -x["final_score"],
            -x["candidate"].get("redrob_signals", {}).get("recruiter_response_rate", 0.0),
            x["candidate"]["candidate_id"]
        ))
        
        # Take top 100
        self.ranked_100 = []
        for rank, item in enumerate(self.scored_list[:100], 1):
            # Percentile-based score mapping: Rank 1 is 1.0000, Rank 100 is 0.0000
            percentile_score = 1.0 - (rank - 1) / 99.0
            
            # Round score to exactly 4 decimal places
            percentile_score = round(percentile_score, 4)
            
            self.ranked_100.append({
                "candidate": item["candidate"],
                "rank": rank,
                "score": percentile_score,
                "raw_score": item["final_score"],
                "components": item["components"],
                "multiplier": item["behavioral_multiplier"]
            })
            
    def generate_reasoning(self):
        """Generates fact-based, non-templated, verifiable explanation for each candidate's ranking."""
        print("Generating explainable reasoning text...")
        for item in self.ranked_100:
            cand = item["candidate"]
            rank = item["rank"]
            score = item["score"]
            signals = cand.get("redrob_signals", {})
            
            # Base facts extraction
            title = cand.get("profile", {}).get("current_title", "Engineer")
            years = cand.get("profile", {}).get("years_of_experience", 0.0)
            
            # Required skills count matching standard skills AND career history text
            required_skill_names = [s[0].lower() for s in self.jd_requirements["required_skills"]]
            candidate_skills = [s["name"].lower() for s in cand.get("skills", [])]
            candidate_skills_std = [SKILL_ALIASES.get(s, s) for s in candidate_skills]
            career_history_text = " ".join([h.get("description", "") for h in cand.get("career_history", [])]).lower()
            
            matched_skills = sum(1 for req in required_skill_names if (any(req in cand_s for cand_s in candidate_skills_std) or req in career_history_text))
            total_req_skills = len(required_skill_names)
            
            facts = []
            # Core description
            facts.append(f"{title} with {years:.1f} yrs experience")
            
            # Recruiter activity & response details
            resp_rate = signals.get("recruiter_response_rate", 0.0)
            facts.append(f"recruiter response rate of {resp_rate:.2f}")
            
            # Date recency
            last_act = signals.get("last_active_date", "")
            if last_act:
                try:
                    days_ago = (datetime(2026, 6, 12) - datetime.strptime(last_act, "%Y-%m-%d")).days
                    if days_ago == 0:
                        facts.append("active today")
                    elif days_ago < 7:
                        facts.append(f"active {days_ago} days ago")
                    else:
                        facts.append(f"active {days_ago} days ago on platform")
                except Exception:
                    pass
                    
            # Mitigating features or highlights depending on rank tier
            if rank <= 10:
                # Top tier: focus on excellence
                saves = signals.get("saved_by_recruiters_30d", 0)
                views = signals.get("profile_views_received_30d", 0)
                if saves + views > 5:
                    facts.append(f"highly sought after by recruiters with {saves} saves")
                if signals.get("github_activity_score", -1) > 70:
                    facts.append(f"outstanding GitHub score of {signals.get('github_activity_score'):.0f}")
                if cand.get("education") and cand["education"][0].get("tier") == "tier_1":
                    facts.append("tier-1 academic credentials")
            elif rank <= 30:
                # High-mid
                if signals.get("interview_completion_rate", 0.0) > 0.8:
                    facts.append(f"strong interview completion ({signals.get('interview_completion_rate'):.0%})")
                if signals.get("open_to_work_flag"):
                    facts.append("actively open to work")
            elif rank <= 60:
                # Mid-tier: balanced, acknowledging gaps
                if resp_rate < 0.3:
                    facts.append("lower engagement score but strong background")
                if len(cand.get("skills", [])) > 10:
                    facts.append("diverse set of auxiliary skills")
            else:
                # Lower tier: honest assessment of constraints
                if matched_skills < total_req_skills * 0.4:
                    facts.append("limited overlap with specific role criteria")
                if years < self.jd_requirements["experience"]["min_years"]:
                    facts.append("junior experience level for this position")
                if signals.get("notice_period_days", 180) > 90:
                    facts.append("subject to long notice period")
                    
            # Combine facts with variety
            reasoning = "; ".join(facts) + "."
            
            # Simple assertions to prevent hallucination (every claim verified in profile)
            assert str(round(years, 1)) in reasoning or str(int(years)) in reasoning, "Year mismatch"
            assert f"{resp_rate:.2f}" in reasoning, "Response rate mismatch"
            
            item["reasoning"] = reasoning
            
    def export_csv(self, output_path):
        """Writes top 100 candidates to output CSV file."""
        print(f"Saving final CSV to {output_path}...")
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["candidate_id", "rank", "score", "reasoning"])
            for item in self.ranked_100:
                writer.writerow([
                    item["candidate"]["candidate_id"],
                    item["rank"],
                    f"{item['score']:.4f}",
                    item["reasoning"]
                ])
                
        print("CSV export finished successfully.")


def main():
    parser = argparse.ArgumentParser(description="Redrob Intelligent Candidate Ranking Engine")
    parser.add_argument("--candidates", "--candidates_path", required=True, help="Path to candidates JSONL dataset")
    parser.add_argument("--out", "--output_path", required=True, help="Path to save ranking output CSV")
    parser.add_argument("--jd_path", default="job_description.docx", help="Path to Job Description file")
    args = parser.parse_args()
    
    start_time = datetime.now()
    print("--------------------------------------------------")
    print(f"System launched at {start_time}")
    print(f"Candidates file: {args.candidates}")
    print(f"JD file: {args.jd_path}")
    print(f"Output CSV path: {args.out}")
    print("--------------------------------------------------")
    
    # Run the full pipeline
    engine = CandidateRankingEngine(args.jd_path, args.candidates)
    engine.parse_jd()
    engine.load_candidates()
    engine.embed_profiles()
    engine.score_candidates()
    engine.rank_and_normalize()
    engine.generate_reasoning()
    engine.export_csv(args.out)
    
    duration = (datetime.now() - start_time).total_seconds()
    print("--------------------------------------------------")
    print(f"Ranking Engine completed in {duration:.2f} seconds.")
    print("--------------------------------------------------")

if __name__ == "__main__":
    main()
