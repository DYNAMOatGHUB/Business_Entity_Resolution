"""Blocker B5: Multilingual MiniLM Exact nearest neighbors via PyTorch GPU."""

import math
import gc
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from tqdm.auto import tqdm

_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
_BATCH_SIZE = 512
_SIM_THRESHOLD = 0.65  # Prune low similarity to save memory

def _get_text(df: pd.DataFrame) -> list[str]:
    # Use fallback normalisations if available
    name = df["name_core"] if "name_core" in df.columns else df["business_name"]
    addr = df["addr_norm"] if "addr_norm" in df.columns else df.get("business_address", pd.Series(""))
    return (name.fillna("") + " " + addr.fillna("")).str.strip().tolist()

@torch.no_grad()
def _encode_in_batches(texts: list[str], tokenizer, model, device, desc="Encoding") -> torch.Tensor:
    all_embs = []
    for i in tqdm(range(0, len(texts), _BATCH_SIZE), desc=desc, leave=False):
        batch_texts = texts[i : i + _BATCH_SIZE]
        enc = tokenizer(batch_texts, padding=True, truncation=True, max_length=128, return_tensors='pt')
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.amp.autocast('cuda', dtype=torch.float16):
            out = model(**enc)
            # Mean pooling
            attention_mask = enc['attention_mask']
            token_embeddings = out.last_hidden_state
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            emb = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            emb = F.normalize(emb, p=2, dim=1)
        all_embs.append(emb.half())
    if not all_embs:
        return torch.empty(0, 384, dtype=torch.float16, device=device)
    return torch.cat(all_embs, dim=0)

def block_embed_ann(s1_df: pd.DataFrame, cand_df: pd.DataFrame, k: int = 25) -> pd.DataFrame:
    """B5: Multilingual MiniLM Exact GPU candidate generation."""
    if s1_df.empty or cand_df.empty:
        return pd.DataFrame(columns=["s1_id", "cand_id", "sim_b5", "rank_b5"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"B5: Loading {_MODEL_NAME} on {device}")
    
    # Load model in FP16 to maximize RTX 5070 12GB VRAM
    tokenizer = AutoTokenizer.from_pretrained(_MODEL_NAME)
    model = AutoModel.from_pretrained(_MODEL_NAME, torch_dtype=torch.float16).to(device).eval()

    print(f"B5: Encoding {len(cand_df)} candidates...")
    cand_texts = _get_text(cand_df)
    cand_ids = cand_df["entity_id"].values
    cand_embs = _encode_in_batches(cand_texts, tokenizer, model, device, desc="Cand Embed")

    print(f"B5: Encoding {len(s1_df)} S1 queries...")
    s1_texts = _get_text(s1_df)
    s1_ids = s1_df["entity_id"].values
    s1_embs = _encode_in_batches(s1_texts, tokenizer, model, device, desc="S1 Embed")

    # Free model VRAM before large matrix multiplications
    del model
    del tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("B5: Computing exact top-K on GPU...")
    results = []
    
    # Dynamically scale chunk size based on candidate size.
    # We have 12GB VRAM. 1000 queries * 5,000,000 candidates * 2 bytes (FP16) = ~10GB.
    # To be safe and leave room for PyTorch overhead, aim for ~4GB chunks.
    max_chunk_elements = (4 * 1024**3) // 2 
    chunk_s1 = max(1, int(max_chunk_elements / max(len(cand_embs), 1)))
    chunk_s1 = min(chunk_s1, 10000)
    
    cand_embs_t = cand_embs.t()
    
    for start in tqdm(range(0, len(s1_embs), chunk_s1), desc="GPU Exact kNN"):
        s1_chunk = s1_embs[start : start + chunk_s1]
        
        # (chunk, 384) @ (384, N) -> (chunk, N)
        sims = torch.matmul(s1_chunk, cand_embs_t) 
        
        k_eff = min(k, sims.shape[1])
        top_sims, top_indices = torch.topk(sims, k_eff, dim=1)
        
        top_sims_cpu = top_sims.cpu().numpy()
        top_indices_cpu = top_indices.cpu().numpy()
        
        for local_i in range(len(s1_chunk)):
            s1_id = s1_ids[start + local_i]
            for rank in range(k_eff):
                sim = float(top_sims_cpu[local_i, rank])
                if sim < _SIM_THRESHOLD:
                    break
                cand_id = cand_ids[top_indices_cpu[local_i, rank]]
                results.append((s1_id, cand_id, sim, rank + 1))
                
    del cand_embs, s1_embs, cand_embs_t
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    res_df = pd.DataFrame(results, columns=["s1_id", "cand_id", "sim_b5", "rank_b5"])
    return res_df
