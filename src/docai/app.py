# app.py -- DocAI OCR worker for AI document pipeline
import os
import json
import base64
import datetime as dt

from flask import Flask, request, abort
from google.cloud import firestore
from google.cloud import storage
from google.cloud import documentai

app = Flask(__name__)

# ---------------------------------------------------------------------------
# ENV CONFIG
# ---------------------------------------------------------------------------
PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT") or os.environ.get("PROJECT_ID")
if not PROJECT_ID:
    raise RuntimeError("PROJECT_ID or GOOGLE_CLOUD_PROJECT must be set")

# Firestore
FS_DATABASE = os.environ.get("FS_DATABASE", "(default)")
FS_COLLECTION = os.environ.get("FS_COLLECTION", "jobs")

# Buckets / prefixes
OUTPUT_BUCKET = os.environ.get("OUTPUT_BUCKET", "fsmb-legal-docs")
TXT_PREFIX = os.environ.get("TXT_PREFIX", "txt/")

# DocAI
DOCAI_LOCATION = os.environ.get("DOCAI_LOCATION", "us")  # e.g. "us" or "us-central1"
DOCAI_PROCESSOR_ID = os.environ.get("DOCAI_PROCESSOR_ID")  # REQUIRED

# If you want to keep this OCR-only worker pure, leave schema/prompt out for now.
# You can still add them later.
# ---------------------------------------------------------------------------
# CLIENTS
# ---------------------------------------------------------------------------
db = firestore.Client(project=PROJECT_ID, database=FS_DATABASE)
storage_client = storage.Client()
docai_client = documentai.DocumentProcessorServiceClient()


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------
def _now_iso() -> str:
    return dt.datetime.utcnow().isoformat() + "Z"


def _update_job(doc_id: str, payload: dict):
    """Merge fields into the job doc."""
    payload["updatedAt"] = _now_iso()
    db.collection(FS_COLLECTION).document(doc_id).set(payload, merge=True)


def _run_docai_ocr(gcs_uri: str) -> str:
    """
    Runs Document AI on the input GCS PDF and returns the extracted plain text.
    We use "raw document" here since you're pulling from GCS yourself.
    """
    if not DOCAI_PROCESSOR_ID:
        raise RuntimeError("DOCAI_PROCESSOR_ID is not set")

    # DocAI resource name
    name = docai_client.processor_path(PROJECT_ID, DOCAI_LOCATION, DOCAI_PROCESSOR_ID)

    # download PDF from GCS
    bucket_name, object_name = gcs_uri.replace("gs://", "").split("/", 1)
    blob = storage_client.bucket(bucket_name).blob(object_name)
    pdf_bytes = blob.download_as_bytes()

    raw_document = documentai.RawDocument(
        content=pdf_bytes,
        mime_type="application/pdf"
    )
    request = documentai.ProcessRequest(
        name=name,
        raw_document=raw_document,
    )
    result = docai_client.process_document(request=request).document

    # Return plain text (DocAI already does layout-aware text)
    return result.text or ""


def _write_text_to_gcs(text: str, doc_id: str) -> str:
    """
    Write the OCR text to GCS in the same bucket, under txt/<docId>.txt
    Returns the GCS URI.
    """
    out_name = f"{TXT_PREFIX}{doc_id}.txt"
    bucket = storage_client.bucket(OUTPUT_BUCKET)
    blob = bucket.blob(out_name)
    blob.upload_from_string(text, content_type="text/plain")
    return f"gs://{OUTPUT_BUCKET}/{out_name}"


# ---------------------------------------------------------------------------
# PUB/SUB ENTRYPOINT
# ---------------------------------------------------------------------------
@app.post("/work")
def work():
    """
    Pub/Sub push endpoint.
    Expects message like:
    {
      "message": {
        "data": "eyJkb2NJZCI6ICJyYXdfYm9hcmRfb3JkZXIucGRmIiwgImdzc1VyaSI6ICJnczovL2ZzbWItbGVnYWwtZG9jcy9yYXcvYm9hcmRfb3JkZXIucGRmIn0="
      }
    }
    where decoded data is:
    {
      "docId": "<safe-id>",
      "gcsUri": "gs://fsmb-legal-docs/raw/board-order.pdf"
    }
    """
    envelope = request.get_json(silent=True)
    if not envelope or "message" not in envelope:
        abort(400, "Bad Pub/Sub push: no message")

    msg = envelope["message"]
    data_b64 = msg.get("data", "")

    try:
        decoded = base64.b64decode(data_b64).decode("utf-8")
        payload = json.loads(decoded or "{}")
    except Exception as e:
        app.logger.error(f"Failed to decode Pub/Sub message: {e}")
        # Tell Pub/Sub we handled it so it doesn't retry forever
        return ("BAD MESSAGE", 200)

    doc_id = payload.get("docId")
    gcs_uri = payload.get("gcsUri")

    if not doc_id or not gcs_uri:
        app.logger.error("Missing docId or gcsUri in message")
        return ("MISSING FIELDS", 200)

    # Mark job as processing OCR
    _update_job(doc_id, {
        "status": "RUNNING",
        "phase": "OCR",
        "ocr": {
            "processorId": DOCAI_PROCESSOR_ID,
            "location": DOCAI_LOCATION,
        }
    })

    try:
        # 1) Run DocAI
        ocr_text = _run_docai_ocr(gcs_uri)

        # 2) Write OCR text to GCS
        text_uri = _write_text_to_gcs(ocr_text, doc_id)

        # 3) Update job to show OCR is done
        _update_job(doc_id, {
            "status": "SUCCEEDED",
            "phase": "OCR_DONE",
            "artifacts": {
                "textUri": text_uri
            }
        })

        app.logger.info(f"OCR succeeded for {doc_id}: {text_uri}")
        return ("OK", 200)

    except Exception as e:
        app.logger.error(f"OCR failed for {doc_id}: {e}")
        _update_job(doc_id, {
            "status": "FAILED",
            "phase": "OCR",
            "errors": [str(e)]
        })
        # IMPORTANT: return 200 so Pub/Sub stops retrying bad input
        return ("FAILED", 200)


# ---------------------------------------------------------------------------
# HEALTH
# ---------------------------------------------------------------------------
@app.get("/")
def health():
    return "ok", 200

