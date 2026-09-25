# Blocker Recall & Candidate Statistics

## Summary
- Target Recall@K: 97%+
- K per S1 entity: 50

| Blocker | Method | Target Coverage | Recall | Candidate Count |
|---|---|---|---|---|
| B1 | Name char n-gram kNN | Typos, abbreviations | - | - |
| B2 | Address char n-gram kNN | Shared address / DBA | - | - |
| B3 | Rare token index | Inverted index on rare words | - | - |
| B4 | Numeric keys | Shared postal/numeric tokens | - | - |
| B5 | Multilingual MiniLM + FAISS | Semantic similarity | - | - |
| Union | Blended union capped at K | Consolidated candidate pairs | - | - |
