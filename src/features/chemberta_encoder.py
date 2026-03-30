"""
ChemBERTa SMILES 인코더
SMILES → CLS embedding (768-dim), 결과를 .npy/.pkl로 캐싱.
"""

import os
import pickle
import numpy as np

SCRIPT_DIR       = os.path.dirname(os.path.abspath(__file__))
CACHE_EMB_PATH   = os.path.join(SCRIPT_DIR, "chemberta_embeddings.npy")
CACHE_SMILES_PATH = os.path.join(SCRIPT_DIR, "chemberta_smiles.pkl")

MODEL_NAME = "seyonec/ChemBERTa-zinc-base-v1"


def encode_smiles(
    smiles_list: list,
    batch_size: int = 32,
    use_cache: bool = True,
) -> np.ndarray:
    """
    SMILES list → CLS embedding (n, 768) numpy array.

    캐시가 존재하고 SMILES 목록이 동일하면 파일에서 로드.
    그렇지 않으면 ChemBERTa 모델로 추출 후 캐시 저장.
    """
    smiles_list = list(smiles_list)

    if use_cache and os.path.exists(CACHE_EMB_PATH) and os.path.exists(CACHE_SMILES_PATH):
        with open(CACHE_SMILES_PATH, "rb") as f:
            cached_smiles = pickle.load(f)
        if cached_smiles == smiles_list:
            print("[ChemBERTa] 캐시에서 임베딩 로드")
            return np.load(CACHE_EMB_PATH)

    import torch
    from transformers import AutoTokenizer, AutoModel

    print(f"[ChemBERTa] '{MODEL_NAME}' 로드 중...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model     = AutoModel.from_pretrained(MODEL_NAME)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = model.to(device)
    print(f"[ChemBERTa] 디바이스: {device}, SMILES 수: {len(smiles_list)}")

    embeddings = []
    for start in range(0, len(smiles_list), batch_size):
        batch = smiles_list[start : start + batch_size]
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        # CLS token: last_hidden_state[:, 0, :]
        cls_emb = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        embeddings.append(cls_emb)
        if (start // batch_size) % 10 == 0:
            print(f"  [{start + len(batch)}/{len(smiles_list)}]")

    result = np.vstack(embeddings)  # (n, 768)

    if use_cache:
        np.save(CACHE_EMB_PATH, result)
        with open(CACHE_SMILES_PATH, "wb") as f:
            pickle.dump(smiles_list, f)
        print(f"[ChemBERTa] 캐시 저장 완료: {CACHE_EMB_PATH}")

    return result


def load_embeddings_for_df(smiles_series, batch_size: int = 32) -> np.ndarray:
    """
    Feature.csv의 SMILES Series에 맞춰 임베딩을 정렬하여 반환.
    전체 Dataset의 임베딩 캐시에서 SMILES 기준으로 매핑.
    """
    target_smiles = smiles_series.tolist()

    # 캐시 확인
    if os.path.exists(CACHE_EMB_PATH) and os.path.exists(CACHE_SMILES_PATH):
        with open(CACHE_SMILES_PATH, "rb") as f:
            cached_smiles = pickle.load(f)
        cached_emb = np.load(CACHE_EMB_PATH)

        # SMILES → index 매핑
        smiles_to_idx = {s: i for i, s in enumerate(cached_smiles)}
        if all(s in smiles_to_idx for s in target_smiles):
            idxs = [smiles_to_idx[s] for s in target_smiles]
            return cached_emb[idxs]

    # 캐시 미스: target_smiles 전체를 새로 인코딩
    print("[ChemBERTa] 캐시 미스 - 직접 인코딩")
    return encode_smiles(target_smiles, batch_size=batch_size, use_cache=False)
