# Widget API Guide

The Widget API allows clients to create, read, update, and delete widgets over HTTP.
All endpoints are versioned under `/v1` and require an API key sent via the `Authorization` header.
Requests must be encoded as JSON and responses include a `request_id` for tracing.

## Authentication
Include `Authorization: Bearer <API_KEY>` with every request. Keys can be rotated from the dashboard.
If the key is missing or invalid, the API returns HTTP 401 with an error code of `auth_failed`.

## Rate Limits
Each project has a default rate limit of 120 requests per minute and 10 concurrent writes.
Burst traffic is smoothed by a token bucket and exceeding the limit returns HTTP 429 with `retry_after` seconds.
Use exponential backoff and respect the `Retry-After` header to avoid thundering herd.

## Pagination
List endpoints return results in pages of up to 100 items. Use the `next_cursor` field to request the next page.
Cursors expire after 15 minutes. If a cursor is expired, start again without the cursor to receive a fresh one.

## Webhooks
You can configure webhooks to receive events when widgets are created or deleted.
Webhooks must respond within 5 seconds; otherwise the delivery is retried with exponential backoff up to 24 hours.
Signatures are verified with a secret stored in the dashboard to prevent spoofing.