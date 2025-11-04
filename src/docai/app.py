# src/docai/app.py
import os
import json
import base64
import datetime as dt
import io

from flask import Flask, request, abort
from google.cloud import firestore, storage, documentai, pubsub_v1
from pypdf import PdfReader

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
# ---------------------------------------------------------------------
# STARTUP LOG
# ---------------------------------------------------------------------
app.logger.info(
    f"OCR worker boot: "
    f"PROJECT_ID={PROJECT_ID}, "
    f"NEXT_TOPIC={NEXT_TOPIC}, "
    f"OUTPUT_BUCKET={OUTPUT_BUCKET}, "
    f"DOCAI_LOCATION={DOCAI_LOCATION}, "
    f"DOCAI_PROCESSOR_ID={'set' if DOCAI_PROCESSOR_ID else 'unset'}"
)
def _now_iso() -> str:
    return dt.datetime.utcnow().isoformat() + "Z"


def _update_job(doc_id: str, payload: dict):
    payload["updatedAt"] = _now_iso()
    db.collection(FS_COLLECTION).document(doc_id).set(payload, merge=True)


# In your imports at the top of the file, add PdfWriter
# from pypdf import PdfReader, PdfWriter
from pypdf import PdfReader, PdfWriter


def _run_docai_ocr(gcs_uri: str) -> str:
    """
    Picks the best OCR strategy based on page count.
    """
    if not DOCAI_PROCESSOR_ID:
        return f"[DocAI not configured] Source: {gcs_uri}"

    # 1) Download the PDF
    bucket_name, object_name = gcs_uri.replace("gs://", "").split("/", 1)
    pdf_bytes = storage_client.bucket(bucket_name).blob(object_name).download_as_bytes()
    pdf_file = io.BytesIO(pdf_bytes)

    # 2) Count pages locally
    reader = PdfReader(pdf_file)
    num_pages = len(reader.pages)
    app.logger.info(f"DocAI: {gcs_uri} has {num_pages} pages")

    name = docai_client.processor_path(PROJECT_ID, DOCAI_LOCATION, DOCAI_PROCESSOR_ID)

    def _process_online(doc_bytes, process_options=None):
        """Helper to process a raw document from bytes."""
        raw_doc = documentai.RawDocument(
            content=doc_bytes, mime_type="application/pdf"
        )
        req = documentai.ProcessRequest(
            name=name,
            raw_document=raw_doc,
            process_options=process_options
        )
        return docai_client.process_document(request=req).document

    # Helper: build ProcessOptions for "imageless"
    def _opts_native():
        return documentai.ProcessOptions(
            ocr_config=documentai.OcrConfig(enable_native_pdf_parsing=True)
        )

    # 3) Strategy selection
    if num_pages <= 15:
        # Strategy 1: Image-based, all pages at once
        app.logger.info(f"Strategy: Online (1 chunk, {num_pages} pages)")
        doc = _process_online(pdf_bytes)
        return doc.text or ""

    if num_pages <= 30:
        # Strategy 2: Try native ("imageless") parsing first
        app.logger.info(f"Strategy: Online Imageless (1 chunk, {num_pages} pages)")
        try:
            doc = _process_online(pdf_bytes, _opts_native())
            text = (doc.text or "").strip()
            if text:
                return text
            app.logger.info("Native parsing returned empty text; falling back to chunks.")
        except Exception as ex:
            # Some processors ignore imageless and still enforce 15-page limit
            app.logger.warning(f"Native parsing failed, falling back to chunks: {ex}")

    # Strategy 3: Chunking (for >30 pages OR imageless fallback)
    # We must create new, smaller PDFs for each chunk.
    app.logger.info(f"Strategy: Online Chunking ({num_pages} total pages)")
    combined = []
    start = 1
    chunk_size = 15  # Max pages for online processing

    while start <= num_pages:
        end = min(start + chunk_size - 1, num_pages)
        app.logger.info(f"Processing chunk: pages {start}-{end}")

        # Create a new PDF in memory with just this chunk's pages
        writer = PdfWriter()
        for page_num in range(start - 1, end):  # pypdf is 0-indexed
            writer.add_page(reader.pages[page_num])

        # Get the bytes from the in-memory PDF
        chunk_io = io.BytesIO()
        writer.write(chunk_io)
        chunk_bytes = chunk_io.getvalue()

        # Process *only* the chunk bytes
        d = _process_online(chunk_bytes)
        combined.append(d.text or "")
        start = end + 1

    return "\n".join(combined)


def _write_text_to_gcs(text: str, doc_id: str) -> str:
    out_name = f"{TXT_PREFIX}{doc_id}.txt"
    blob = storage_client.bucket(OUTPUT_BUCKET).blob(out_name)
    blob.upload_from_string(text, content_type="text/plain; charset=utf-8")
    return f"gs://{OUTPUT_BUCKET}/{out_name}"


def _publish_next(payload: dict):
    data = json.dumps(payload).encode("utf-8")
    msg_id = publisher.publish(next_topic_path, data).result(timeout=10)
    app.logger.info(f"Published to {NEXT_TOPIC} msg_id={msg_id} for docId={payload.get('docId')}")


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
