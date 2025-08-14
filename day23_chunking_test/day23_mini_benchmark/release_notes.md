# Release Notes 1.8.0

- Added webhook signature verification toggle in the dashboard.
- Increased default rate limit from 100 to 120 requests per minute.
- Pagination cursors now expire after 15 minutes instead of 10.
- Fixed a bug where 429 responses occasionally missed the `Retry-After` header.
- Improved reranker model to v3, boosting nDCG@10 by 6% on our internal benchmark.