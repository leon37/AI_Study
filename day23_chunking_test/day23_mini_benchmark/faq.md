# Frequently Asked Questions

**Q: Why are my requests failing with 401?**
A: Most 401 errors occur because the API key is missing or invalid.
Double-check the `Authorization` header, ensure there are no leading or trailing spaces, and verify the key is active.

**Q: How do I handle 429 rate limit errors?**
A: Respect the `Retry-After` header, implement exponential backoff, and reduce concurrency for write operations.
You can request a higher quota from support if your use case requires sustained bursts.

**Q: How do webhooks verify authenticity?**
A: Webhook requests include a signature header derived from a shared secret. Recompute the signature on your side and compare it.
Reject requests with missing or mismatched signatures.