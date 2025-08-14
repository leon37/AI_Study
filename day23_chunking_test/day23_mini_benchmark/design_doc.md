# Search Service Design

The search service provides hybrid retrieval combining lexical BM25 and dense embeddings.
Queries are normalized, tokenized, and routed through a retriever that merges results by normalized scores.
A cross-encoder reranker is applied to the top 100 candidates to improve precision at small k.

## Chunking
Documents are chunked using a sentence-aware splitter with a target of 512 tokens and 20% overlap.
This improves recall for cross-sentence entities while keeping latency bounded.
For code and configuration files, a structural splitter based on fences and headings performs better than fixed-size windows.

## Caching
Hot queries are cached for 5 minutes. The cache key includes user locale and filters to avoid leakage.