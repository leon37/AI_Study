# Troubleshooting Guide

If you receive HTTP 401, verify your API key and ensure the `Authorization` header is present.
For HTTP 429, implement exponential backoff and honor `Retry-After` to prevent retries overwhelming the system.
Webhook deliveries that do not respond within 5 seconds are retried automatically for up to 24 hours.
If webhooks appear forged, verify the signature using the configured secret.