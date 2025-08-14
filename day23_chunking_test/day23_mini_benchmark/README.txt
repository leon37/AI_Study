# Day23 Mini Benchmark (Chunking Evaluation)

This dataset contains 6 documents across API docs, FAQs, design docs, release notes, troubleshooting, and ToS.
Use it to evaluate different chunking strategies (fixed, sentence-aware, structural) and parameters (size, overlap).

Files:
- api_guide.md
- faq.md
- design_doc.md
- release_notes.md
- troubleshooting.md
- terms_of_service.md
- sentences.json  (sentence list per document; indices are 0-based)
- queries_gold.json (18 queries with gold relevant sentence IDs)

Evaluation (Protocol A - sentence-level):
1. Chunk the documents using your strategy.
2. When a chunk is retrieved for a query, mark any sentence IDs that the chunk covers.
3. Recall = |covered gold sentences| / |gold sentences|
4. Precision = #chunks that cover at least one gold sentence / #retrieved chunks

Optionally report additional metrics (MRR, nDCG@k) using your retriever scores.