"""Phase 3: FAISS 기반 등록 얼굴 관리 (face_store.py)

- add_person(name, embeddings): 인물 등록 (여러 장 → 품질 가중 평균, proto.build_db와 동일 규격)
- remove_person(name): 인물 삭제
- search(embedding, threshold): 코사인 유사도 기준 최근접 1명 → (name, sim) / ("unknown", sim)
- save/load: npz 파일로 영속화 (Phase 2 proto.load_db 호환)

FAISS IndexFlatIP + L2 정규화 임베딩 = 코사인 유사도 탐색.
"""
import numpy as np

try:
    import faiss
except ImportError:  # faiss 미설치 환경에서는 numpy 폴백
    faiss = None


def _normalize(embs):
    embs = np.asarray(embs, dtype=np.float32)
    n = np.linalg.norm(embs, axis=1, keepdims=True)
    return embs / np.maximum(n, 1e-9)


class FaceStore:
    def __init__(self, dim=512):
        self.dim = dim
        self.names = []          # person 이름 (인덱스 순서)
        self.embs = np.zeros((0, dim), dtype=np.float32)  # 정규화된 평균 임베딩
        self._index = None

    # ---------- CRUD ----------
    def add_person(self, name, embeddings):
        """embeddings: (N, dim) 원시 임베딩. 품질 가중 평균 후 등록/갱신."""
        avg = _normalize(embeddings).mean(axis=0, keepdims=True)
        avg = avg / max(np.linalg.norm(avg), 1e-9)
        if name in self.names:
            self.embs[self.names.index(name)] = avg[0]
        else:
            self.names.append(name)
            self.embs = np.vstack([self.embs, avg])
        self._rebuild()

    def remove_person(self, name):
        if name not in self.names:
            return False
        i = self.names.index(name)
        del self.names[i]
        self.embs = np.delete(self.embs, i, axis=0)
        self._rebuild()
        return True

    def list_persons(self):
        return list(self.names)

    # ---------- 검색 ----------
    def _rebuild(self):
        if faiss is None or len(self.names) == 0:
            self._index = None
            return
        self._index = faiss.IndexFlatIP(self.dim)
        self._index.add(self.embs.astype(np.float32))

    def search(self, embedding, threshold=0.4):
        """→ (name, sim). 임계값 미달이면 ("unknown", sim)."""
        if len(self.names) == 0:
            return "unknown", 0.0
        q = _normalize(np.asarray(embedding, dtype=np.float32).reshape(1, -1))
        if self._index is not None:
            sims, idx = self._index.search(q, 1)
            sim, i = float(sims[0, 0]), int(idx[0, 0])
        else:  # numpy 폴백
            sims = self.embs @ q[0]
            i = int(np.argmax(sims))
            sim = float(sims[i])
        name = self.names[i] if sim >= threshold else "unknown"
        return name, sim

    # ---------- 영속화 ----------
    def save(self, path):
        np.savez(path, names=np.array(self.names),
                 embs=self.embs, dim=np.array(self.dim))

    @classmethod
    def load(cls, path):
        d = np.load(path, allow_pickle=True)
        store = cls(dim=int(d["dim"]))
        store.names = [str(n) for n in d["names"]]
        store.embs = d["embs"].astype(np.float32)
        store._rebuild()
        return store

    @classmethod
    def from_npz(cls, path):
        """Phase 2 proto.build_db 산출 npz와 호환 (names/embeddings 또는 names/embs)."""
        d = np.load(path, allow_pickle=True)
        key = "embeddings" if "embeddings" in d else "embs"
        store = cls(dim=d[key].shape[1])
        store.names = [str(n) for n in d["names"]]
        store.embs = _normalize(d[key].astype(np.float32))
        store._rebuild()
        return store
