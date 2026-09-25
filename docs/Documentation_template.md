# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** [Your Team Name]  
**Team Members:** [List all team members]  
**Submission Date:** [Date]

---

## 1. Executive Summary
*Provide a brief 2-3 sentence overview of your approach and key innovations.*

---

## 2. Methodology

### 2.1 Problem Analysis
*Key insights discovered during EDA — noise patterns, address variations, missing fields, etc.*

### 2.2 Solution Strategy
*Outline your high-level approach.*

**Approach Type:** [Blocking + Classifier / End-to-End / Graph-Based / Hybrid, etc]  
**Core Innovation:** [Brief description of your main technical contribution]

---

## 3. Candidate Generation (Blocking)
*Describe how you reduced the comparison space to a manageable candidate set.*

- **Blocking keys used:** TF-IDF character-level kNN (chunked safe sparse dot product with country-grouping fallback) and Dense Embeddings via multi-lingual sentence-transformers with batched GPU acceleration.
- **Candidate pairs generated:** [INSERT FINAL PAIR COUNT HERE]
- **How you ensured true matches were not lost:** We implemented a dual blocking approach combining sparse lexical matching (names and addresses via TF-IDF character kNN) with dense semantic matching (GPU-accelerated exact `top-k` cdist cross-joins). The `blend_rank` logic in `union.py` then aggregates the hits. To protect the singleton F_0.5 score, we strictly penalize and drop zero-signal pairs, thereby reducing false merges while retaining high-confidence true matches.

---

## 4. Matching Model

**Features used:**
- Name features: [e.g., Jaccard, Levenshtein, phonetic encoding]
- Address features: [e.g., token overlap, edit distance, PIN code matching]
- Other: Vector cosine similarities (TF-IDF and Dense Embeddings), and context rank features.

**Model type:** LightGBM  
**Threshold selection method:** 5-fold GroupKFold LightGBM strategy with F_0.5 score optimization via threshold sweeping.

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro):** [your best validation score]
- **Common false positives (wrong merges):** [brief description]
- **Common false negatives (missed matches):** [brief description]

---

## 6. Conclusion
*Summarize your approach, key achievements, and lessons learned in 2-3 sentences.*

---

## Appendix

### A. Code Artefacts
*Your complete, runnable code ships in the submission zip under
`code/business_entity_resolution/` (all source in `src/`, with a `README.md` and
`requirements.txt`). Summarise its structure and the entry point(s) to reproduce
`output/matching_results.tsv` and `output/candidate_pairs.tsv` here.*

### B. Additional Results
*Include any additional charts, graphs, or detailed results.*

---

**Note:** Teams can modify sections according to their approach while maintaining clarity and technical depth.
