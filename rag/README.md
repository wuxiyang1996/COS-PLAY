# RAG Embedding Models

This package provides two configurable embedders for the Game-AI-Agent framework:

1. **Text (RAG) embedding** – for experience summaries, queries, and retrieval (default: `Qwen/Qwen3-Embedding-0.6B`).
2. **Multimodal embedding** – for text + image / video (default: `Qwen/Qwen3-VL-Embedding-2B`).

Both support alternative models via constructor arguments or environment variables.

## Quick start

```python
from rag import get_text_embedder, get_multimodal_embedder

# Text embedder (RAG): default Qwen/Qwen3-Embedding-0.6B
text_embedder = get_text_embedder()
embeddings = text_embedder.encode(["state summary A", "state summary B"])
query_emb = text_embedder.encode("current game state", prompt_name="query")

# Multimodal embedder: default Qwen/Qwen3-VL-Embedding-2B
mm_embedder = get_multimodal_embedder()
inputs = [
    {"text": "A woman playing with her dog on a beach."},
    {"image": "https://example.com/image.jpeg"},
    {"text": "Caption here", "image": "/path/to/local.jpg"},
]
embeddings = mm_embedder.process(inputs)
```

## Ranking and retrieval

Compare a query’s embedding to all stored memory embeddings and get **top-k** similar memories. **k** is a hyperparameter (default 5).

### Text memory store (RAG)

```python
from rag import get_text_embedder, get_memory_store

embedder = get_text_embedder()
store = get_memory_store(embedder, top_k=5)

# Add text memories (e.g. experience summaries)
store.add_texts(
    ["Player 1 moved north.", "Player 2 proposed trade.", "All agreed on peace."],
    payloads=[{"id": 1, "action": "move"}, {"id": 2, "action": "trade"}, {"id": 3, "action": "peace"}],
)

# Top-k similar memories for a query
results = store.retrieve("current state: player 2 wants to trade", k=2)
# [(payload, score), ...] for the 2 most similar

# Or (index, score, payload)
for idx, score, payload in store.rank("who proposed trade?", k=2):
    print(score, payload)
```

### Multimodal memory store

```python
from rag import get_multimodal_embedder, get_memory_store

embedder = get_multimodal_embedder()
store = get_memory_store(embedder, top_k=5)

# Add multimodal memories
store.add_multimodal(
    [{"text": "Board state: castles at A1, B2."}, {"image": "path/to/screenshot.png"}],
    payloads=[{"turn": 1}, {"turn": 2}],
)

# Query with text or multimodal
results = store.retrieve({"text": "Where are the castles?"}, k=2)
```

### Low-level: precomputed embeddings

```python
from rag import rank_memories
import numpy as np

query_emb = ...   # shape (dim,)
memory_embs = ...  # shape (n, dim)
payloads = [...]   # length n

top_k = 5
for idx, score, payload in rank_memories(query_emb, memory_embs, payloads=payloads, k=top_k):
    print(idx, score, payload)
```

- **`MemoryStore(embedder, top_k=5)`** – holds memories and uses the embedder for queries (and for `add_texts` / `add_multimodal`).
- **`store.retrieve(query, k=None)`** – returns top-k `(payload, score)` or payloads; **k** defaults to the store’s **top_k**.
- **`store.rank(query, k=None)`** – returns top-k `(index, score, payload)`.
- **`rank_memories(query_embedding, memory_embeddings, payloads, k)`** – no embedder; use when you already have vectors.

## Configuration

### Defaults

| Use case              | Default model                     | Env override               |
|-----------------------|------------------------------------|-----------------------------|
| RAG / text embedding  | `Qwen/Qwen3-Embedding-0.6B`       | `RAG_EMBEDDING_MODEL`       |
| Multimodal embedding  | `Qwen/Qwen3-VL-Embedding-2B`       | `MULTIMODAL_EMBEDDING_MODEL`|

### Using other models

**Text (RAG):** Any [Sentence Transformers](https://www.sbert.net/) / Hugging Face text embedding model:

```python
from rag import TextEmbedder

# Other Qwen3 sizes
embedder = TextEmbedder(model_name_or_path="Qwen/Qwen3-Embedding-4B")

# Or another provider
embedder = TextEmbedder(model_name_or_path="BAAI/bge-m3")
```

**Multimodal:** Any Qwen3-VL-Embedding–compatible model (e.g. 8B):

```python
from rag import MultimodalEmbedder

embedder = MultimodalEmbedder(model_name_or_path="Qwen/Qwen3-VL-Embedding-8B")
```

### Environment variables

```bash
export RAG_EMBEDDING_MODEL=Qwen/Qwen3-Embedding-4B
export MULTIMODAL_EMBEDDING_MODEL=Qwen/Qwen3-VL-Embedding-8B
```

## Dependencies

- **Text embedding:** `sentence-transformers>=2.7.0`, `transformers>=4.51.0`
- **Multimodal embedding:** `transformers>=4.57.0`, `qwen-vl-utils>=0.0.14`, `torch`

Install from the project root:

```bash
pip install -r rag/requirements.txt
```

For multimodal, ensure a recent `transformers` build includes Qwen3-VL support and install `qwen-vl-utils`.

## References

- [Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B)
- [Qwen3-VL-Embedding-2B](https://huggingface.co/Qwen/Qwen3-VL-Embedding-2B)
