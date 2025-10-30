# app.py (doc-worker)
import os
import json
import base64
import datetime as dt

from flask import Flask, request, abort
from google.cloud import firestore
from google.cloud import storage
# from google.cloud import documentai  # uncomment when you plug Doc AI

app = Flask(__name__)

PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT") or os.environ.get("PROJECT_ID")
FS_DATABASE = os.environ.get("FS_DATABASE", "(default)")
FS_COLLECTION = os.environ.get("FS_COLLECTION", "jobs")
OUTPUT_BUCKET = os.environ.get("OUTPUT_BUCKET", "fsmb-legal-docs")
PROCESSED_PREFIX = os.environ.get("PROCESSED_PREFIX", "processed/")
DOCAI_LOCATION = os.environ.get("DOCAI_LOCATION", "us")
DOCAI_PROCESSOR_ID = os.environ.get("DOCAI_PROCESSOR_ID")  # optional for now

if not PROJECT_ID:
    raise RuntimeError("PROJECT_ID/GOOGLE_CLOUD_PROJECT not set")

db = firestore.Client(project=PROJECT_ID, database=FS_DATABASE)
storage_client = storage.Client()
# docai_client = documentai.DocumentProcessorServiceClient()  # when ready


def _update_job(doc_id: str, payload: dict):
    payload["updatedAt"] = dt.datetime.utcnow().isoformat() + "Z"
    db.collection(FS_COLLECTION).document(doc_id).set(payload, merge=True)


def _fake_process(gcs_uri: str) -> dict:
    """
    Temporary processor so we can prove end-to-end works
    before wiring real Doc AI / Vertex.
    """
    return {
        "summary": f"Processed document at {gcs_uri}",
        "severity": "info",
        "engine": "fake",
    }


# Example real Doc AI stub (leave commented for now)
"""
def _process_with_docai(gcs_uri: str) -> dict:
    name = docai_client.processor_path(PROJECT_ID, DOCAI_LOCATION, DOCAI_PROCESSOR_ID)
    # download file from GCS
    bucket_name, object_name = gcs_uri.replace("gs://", "").split("/", 1)
    blob = storage_client.bucket(bucket_name).blob(object_name)
    content = blob.download_as_bytes()

    raw_document = documentai.RawDocument(content=content, mime_type="application/pdf")
    request = documentai.ProcessRequest(name=name, raw_document=raw_document)
    result = docai_client.process_document(request=request).document

    return {
        "text": result.text,
        "entities": [
            {"type": e.type_, "mentionText": e.mention_text}
            for e in result.entities
        ],
    }
"""


@app.post("/work")
def work():
    # Pub/Sub push format: {"message": {"data": "<base64>"}}
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
        abort(400, "Bad Pub/Sub data")

    doc_id = payload.get("docId")
    gcs_uri = payload.get("gcsUri")
    routing_type = payload.get("type", "job.queued")

    if not doc_id or not gcs_uri:
        app.logger.error("Missing docId or gcsUri in message")
        abort(400, "Missing docId or gcsUri")

    # mark job RUNNING
    _update_job(doc_id, {"status": "RUNNING", "phase": "PROCESS"})

    # pick engine (you can expand later)
    # if routing_type == "job.queued" and DOCAI_PROCESSOR_ID:
    #     result = _process_with_docai(gcs_uri)
    # else:
    result = _fake_process(gcs_uri)

    # write output JSON to GCS
    out_key = f"{PROCESSED_PREFIX}{doc_id}.json"
    out_blob = storage_client.bucket(OUTPUT_BUCKET).blob(out_key)
    out_blob.upload_from_string(
        json.dumps(result, ensure_ascii=False, indent=2),
        content_type="application/json",
    )

    # mark job SUCCEEDED
    _update_job(
        doc_id,
        {
            "status": "SUCCEEDED",
            "phase": "INDEX",
            "outputs": {
                "jsonUri": f"gs://{OUTPUT_BUCKET}/{out_key}"
            },
        },
    )

    app.logger.info(
        f"Processed job docId={doc_id} from {gcs_uri} -> gs://{OUTPUT_BUCKET}/{out_key}"
    )

    return ("OK", 200)


@app.get("/")
def health():
    return "ok", 200
