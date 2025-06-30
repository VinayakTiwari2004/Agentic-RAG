from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load reranker once globally
reranker_tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-reranker-large")
reranker_model = AutoModelForSequenceClassification.from_pretrained("BAAI/bge-reranker-large")
reranker_model.eval()

def rerank_chunks(query: str, chunks: list, top_n=10):
    # Create list of (query, passage) pairs
    pairs = [(query, chunk["chunk"]) for chunk in chunks]

    with torch.no_grad():
        inputs = reranker_tokenizer(pairs, padding=True, truncation=True, return_tensors="pt")
        scores = reranker_model(**inputs).logits.squeeze(-1)  # (batch,)

    # Combine original chunks with scores
    for i, score in enumerate(scores.tolist()):
        chunks[i]["rerank_score"] = score

    # Sort and return top N chunks
    top_chunks = sorted(chunks, key=lambda x: x["rerank_score"], reverse=True)[:top_n]
    return top_chunks
