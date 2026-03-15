# friendlyface-sdk

Python SDK for the [FriendlyFace](https://github.com/Dicoangelo/FriendlyFace) forensic-friendly AI platform. Add verifiable AI compliance to your pipeline in 5 lines of code.

## Install

```bash
pip install friendlyface-sdk
```

## Quick Start

```python
from friendlyface_sdk import FriendlyFaceClient

client = FriendlyFaceClient("https://your-instance.com", api_key="your-key")

# Issue a ForensicSeal compliance certificate
seal = client.issue_seal("system-1", "My AI System")

# Verify it (works without auth)
result = client.verify_seal(seal_id=seal.seal_id)
print(f"Valid: {result.valid}, Score: {result.compliance_score}")
```

## Features

- **ForensicSeal** — Issue and verify W3C Verifiable Credential compliance certificates
- **Forensic Logging** — Record hash-chained forensic events for any AI operation
- **Consent Management** — Check and grant consent before processing
- **Bias Auditing** — Run fairness audits with demographic parity and equalized odds
- **Compliance Proxy** — Route recognition API calls through forensic logging
- **Forensic Bundles** — Package events into self-verifiable evidence bundles

## API

### Client

```python
client = FriendlyFaceClient(base_url, api_key=None)
```

| Method | Description |
|--------|-------------|
| `log_event(event_type, content, metadata)` | Record a forensic event |
| `check_consent(subject_id)` | Check consent status |
| `grant_consent(subject_id, purpose, granted_by)` | Grant consent |
| `create_bundle(event_ids, provenance_node_ids)` | Create forensic bundle |
| `issue_seal(system_id, system_name, ...)` | Issue ForensicSeal |
| `verify_seal(credential, seal_id)` | Verify a seal |
| `run_audit(predictions, demographics)` | Run bias audit |
| `proxy_recognize(image_bytes, upstream_url, ...)` | Proxy recognition request |

### Decorator

```python
from friendlyface_sdk import FriendlyFaceClient, forensic_trace

client = FriendlyFaceClient("https://your-instance.com", api_key="key")

@forensic_trace(client)
def my_inference(image_path):
    # Your inference code here
    return predictions
```

Every call to `my_inference` automatically logs inputs and outputs as forensic events.

### Context Manager

```python
with client.forensic_session() as session:
    client.log_event("inference_request", {"image": "photo.jpg"})
    client.log_event("inference_result", {"match": "subject-1"})
# All events auto-bundled on exit
```

## Requirements

- Python 3.9+
- `requests` (only dependency)

## License

MIT
