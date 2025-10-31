# src/docai/app.py
import os
import json
import base64
import datetime as dt

from flask import Flask, request, abort
from google.cloud import firestore, storage, documentai, pubsub_v1

app = Flask(__name__)

# ---------------------------------------------------------------------
# ENV
# ---------------------------------------------------------------------
PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT") or os.environ.get("PROJECT_ID")
if not PROJECT_ID:
    raise RuntimeError("PROJECT_ID / GOOGLE_CLOUD_PROJECT must be set")

FS_DATABASE = os.environ.get("FS_DATABASE", "(default)")
FS_COLLECTION = os.environ.get("FS_COLLECTION", "jobs")

OUTPUT_BUCKET = os.environ.get("OUTPUT_BUCKET", "fsmb-legal-docs")
TXT_PREFIX = os.environ.get("TXT_PREFIX", "txt/")

DOCAI_LOCATION = os.environ.get("DOCAI_LOCATION", "us")
DOCAI_PROCESSOR_ID = os.environ.get("DOCAI_PROCESSOR_ID")  # may be None for now

# topic to notify the next stage (Gemini worker)
NEXT_TOPIC = os.environ.get("NEXT_TOPIC", "doc-ocr-complete")

# ---------------------------------------------------------------------
# CLIENTS
# ---------------------------------------------------------------------
db = firestore.Client(project=PROJECT_ID, database=FS_DATABASE)
storage_client = storage.Client()
publisher = pubsub_v1.PublisherClient()
docai_client = documentai.DocumentProcessorServiceClient()

next_topic_path = publisher.topic_path(PROJECT_ID, NEXT_TOPIC)


# ---------------------------------------------------------------------
def _now_iso() -> str:
    return dt.datetime.utcnow().isoformat() + "Z"


def _update_job(doc_id: str, payload: dict):
    payload["updatedAt"] = _now_iso()
    db.collection(FS_COLLECTION).document(doc_id).set(payload, merge=True)


def _run_docai_ocr(gcs_uri: str) -> str:
    if not DOCAI_PROCESSOR_ID:
        # fallback / dry run
        return f"[DocAI not configured] Source: {gcs_uri}"

    name = docai_client.processor_path(PROJECT_ID, DOCAI_LOCATION, DOCAI_PROCESSOR_ID)

    bucket_name, object_name = gcs_uri.replace("gs://", "").split("/", 1)
    pdf_bytes = storage_client.bucket(bucket_name).blob(object_name).download_as_bytes()

    raw_document = documentai.RawDocument(content=pdf_bytes, mime_type="application/pdf")
    request = documentai.ProcessRequest(name=name, raw_document=raw_document)
    result = docai_client.process_document(request=request).document
    return result.text or ""


def _write_text_to_gcs(text: str, doc_id: str) -> str:
    out_name = f"{TXT_PREFIX}{doc_id}.txt"
    blob = storage_client.bucket(OUTPUT_BUCKET).blob(out_name)
    blob.upload_from_string(text, content_type="text/plain")
    return f"gs://{OUTPUT_BUCKET}/{out_name}"


def _publish_next(payload: dict):
    publisher.publish(next_topic_path, json.dumps(payload).encode("utf-8"))


# ---------------------------------------------------------------------
@app.post("/work")
def work():
    # Pub/Sub push envelope
    envelope = request.get_json(silent=True)
    if not envelope or "message" not in envelope:
        abort(400, "Bad Pub/Sub push")

    msg = envelope["message"]
    data_b64 = msg.get("data", "")

    try:
        decoded = base64.b64decode(data_b64).decode("utf-8")
        payload = json.loads(decoded or "{}")
    except Exception as e:
        app.logger.error(f"decode error: {e}")
        return ("BAD MESSAGE", 200)

    doc_id = payload.get("docId")
    gcs_uri = payload.get("gcsUri")
    routing = payload.get("routing") or {}

    if not doc_id or not gcs_uri:
        app.logger.error("missing docId or gcsUri")
        return ("MISSING FIELDS", 200)

    # mark OCR start
    _update_job(doc_id, {
        "status": "RUNNING",
        "phase": "OCR",
        "ocr": {
            "processorId": DOCAI_PROCESSOR_ID,
            "location": DOCAI_LOCATION
        }
    })

    try:
        # 1) run OCR (real or fake)
        ocr_text = _run_docai_ocr(gcs_uri)

        # 2) write to txt/
        text_uri = _write_text_to_gcs(ocr_text, doc_id)

        # 3) update job
        _update_job(doc_id, {
            "status": "SUCCEEDED",
            "phase": "OCR_DONE",
            "artifacts": {
                "textUri": text_uri
            }
        })

        # 4) tell next worker (Gemini) to run
        _publish_next({
            "type": "ocr.done",
            "docId": doc_id,
            "textUri": text_uri,
            "gcsUri": gcs_uri,
            "routing": routing,   # pass through schemaId / promptId
            "ts": _now_iso()
        })

        app.logger.info(f"OCR OK for {doc_id} -> {text_uri}")
        return ("OK", 200)

    except Exception as e:
        app.logger.error(f"OCR failed for {doc_id}: {e}")
        _update_job(doc_id, {
            "status": "FAILED",
            "phase": "OCR",
            "errors": [str(e)]
        })
        # return 200 so Pub/Sub stops retrying forever
        return ("FAILED", 200)


@app.get("/")
def health():
    return "ok", 200
