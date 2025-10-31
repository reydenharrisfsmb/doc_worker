# src/gemini/app.py
import os
import json
import base64
import datetime as dt

from flask import Flask, request, abort
from google.cloud import firestore, storage

app = Flask(__name__)

# ---------------------------------------------------------------------
# ENV
# ---------------------------------------------------------------------
PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT") or os.environ.get("PROJECT_ID")
if not PROJECT_ID:
    raise RuntimeError("PROJECT_ID / GOOGLE_CLOUD_PROJECT must be set")

FS_DATABASE = os.environ.get("FS_DATABASE", "(default)")
FS_COLLECTION = os.environ.get("FS_COLLECTION", "jobs")

SCHEMAS_COLLECTION = os.environ.get("SCHEMAS_COLLECTION", "schemas")
PROMPTS_COLLECTION = os.environ.get("PROMPTS_COLLECTION", "prompts")

OUTPUT_BUCKET = os.environ.get("OUTPUT_BUCKET", "fsmb-legal-docs")
PROCESSED_PREFIX = os.environ.get("PROCESSED_PREFIX", "processed/")

# ---------------------------------------------------------------------
# CLIENTS
# ---------------------------------------------------------------------
db = firestore.Client(project=PROJECT_ID, database=FS_DATABASE)
storage_client = storage.Client()


# ---------------------------------------------------------------------
def _now_iso() -> str:
    return dt.datetime.utcnow().isoformat() + "Z"


def _update_job(doc_id: str, payload: dict):
    payload["updatedAt"] = _now_iso()
    db.collection(FS_COLLECTION).document(doc_id).set(payload, merge=True)


def _download_text(text_uri: str) -> str:
    bucket_name, object_name = text_uri.replace("gs://", "").split("/", 1)
    blob = storage_client.bucket(bucket_name).blob(object_name)
    return blob.download_as_text()


def _fetch_schema(schema_id: str):
    if not schema_id:
        return None
    snap = db.collection(SCHEMAS_COLLECTION).document(schema_id).get()
    return snap.to_dict() if snap.exists else None


def _fetch_prompt(prompt_id: str):
    if not prompt_id:
        return None
    snap = db.collection(PROMPTS_COLLECTION).document(prompt_id).get()
    return snap.to_dict() if snap.exists else None


def _build_gemini_prompt(prompt_doc: dict, ocr_text: str, schema_doc: dict):
    # very simple template
    user_template = (prompt_doc or {}).get("userTemplate") or "Extract structured data from this document text."
    if "{document_text}" in user_template:
        user_part = user_template.replace("{document_text}", ocr_text)
    else:
        user_part = user_template + "\n\nDOCUMENT TEXT:\n" + ocr_text

    if schema_doc:
        user_part += "\n\nReturn JSON that matches this schema:\n"
        user_part += json.dumps(schema_doc, indent=2)

    return user_part


def _call_gemini(prompt_text: str) -> dict:
    """
    STUB: replace with real Vertex/Gemini call.
    Keep this shape so the rest of the pipeline works.
    """
    return {
        "model": "gemini-stub",
        "promptPreview": prompt_text[:500],
        "extracted": {
            "exampleField": "value"
        }
    }


def _write_json_to_gcs(doc_id: str, result: dict) -> str:
    out_name = f"{PROCESSED_PREFIX}{doc_id}.json"
    blob = storage_client.bucket(OUTPUT_BUCKET).blob(out_name)
    blob.upload_from_string(
        json.dumps(result, indent=2),
        content_type="application/json"
    )
    return f"gs://{OUTPUT_BUCKET}/{out_name}"


# ---------------------------------------------------------------------
@app.post("/work")
def work():
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
    text_uri = payload.get("textUri")
    gcs_uri = payload.get("gcsUri")  # original PDF
    routing = payload.get("routing") or {}
    schema_id = routing.get("schemaId")
    prompt_id = routing.get("promptId")

    if not doc_id or not text_uri:
        app.logger.error("Missing docId or textUri")
        return ("MISSING FIELDS", 200)

    # mark processing
    _update_job(doc_id, {
        "status": "RUNNING",
        "phase": "EXTRACT"
    })

    try:
        # 1) Download OCR text
        ocr_text = _download_text(text_uri)

        # 2) Fetch schema / prompt from Firestore
        schema_doc = _fetch_schema(schema_id)
        prompt_doc = _fetch_prompt(prompt_id)

        # 3) Build prompt and call Gemini
        prompt_text = _build_gemini_prompt(prompt_doc, ocr_text, schema_doc)
        gemini_result = _call_gemini(prompt_text)

        # 4) Write JSON to GCS
        json_uri = _write_json_to_gcs(doc_id, {
            "sourcePdf": gcs_uri,
            "textUri": text_uri,
            "schemaId": schema_id,
            "promptId": prompt_id,
            "geminiResult": gemini_result,
            "generatedAt": _now_iso()
        })

        # 5) Update job
        _update_job(doc_id, {
            "status": "SUCCEEDED",
            "phase": "DONE",
            "artifacts": {
                "textUri": text_uri,
                "jsonUri": json_uri
            },
            "used": {
                "schemaId": schema_id,
                "promptId": prompt_id,
                "engine": "gemini"
            }
        })

        app.logger.info(f"Gemini step OK for {doc_id} -> {json_uri}")
        return ("OK", 200)

    except Exception as e:
        app.logger.error(f"Gemini step failed for {doc_id}: {e}")
        _update_job(doc_id, {
            "status": "FAILED",
            "phase": "EXTRACT",
            "errors": [str(e)]
        })
        # return 200 so Pub/Sub doesn't retry forever
        return ("FAILED", 200)


@app.get("/")
def health():
    return "ok", 200
