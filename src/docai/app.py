import os
import json
import base64
import datetime as dt
import time
from io import BytesIO
from typing import List, Tuple, Optional

from flask import Flask, request, abort

from pypdf import PdfReader, PdfWriter

from google.cloud import firestore, storage, pubsub_v1
from google.cloud import documentai_v1 as documentai
from google.api_core.client_options import ClientOptions
from google.api_core import exceptions as gcp_exceptions


# ==========================
# ENV / CONFIG
# ==========================

def _get_env_or_raise(name: str) -> str:
    v = os.environ.get(name)
    if not v:
        raise RuntimeError(f"Required environment variable '{name}' is not set.")
    return v

PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT") or os.environ.get("PROJECT_ID")
if not PROJECT_ID:
    raise RuntimeError("PROJECT_ID / GOOGLE_CLOUD_PROJECT must be set")

# Firestore
FS_DATABASE = os.environ.get("FS_DATABASE")  # optional (let client default if None)
FS_COLLECTION = os.environ.get("FS_COLLECTION") or "jobs"

# Storage (mandatory)
OUTPUT_BUCKET = _get_env_or_raise("OUTPUT_BUCKET")
TXT_PREFIX = os.environ.get("TXT_PREFIX", "")  # optional prefix for text outputs

# Document AI (mandatory)
DOCAI_LOCATION = _get_env_or_raise("DOCAI_LOCATION")
DOCAI_PROCESSOR_ID = _get_env_or_raise("DOCAI_PROCESSOR_ID")
DOCAI_OUTPUT_PREFIX = _get_env_or_raise("DOCAI_OUTPUT_PREFIX")  # e.g. "docai-results/"

# Pub/Sub (mandatory)
NEXT_TOPIC = _get_env_or_raise("NEXT_TOPIC")

# Chunk size (defaults to your processor’s 15-page limit; you can raise if you switch processors)
MAX_PAGES_PER_DOC = int(os.environ.get("MAX_PAGES_PER_DOC", "15"))

# ==========================
# CLIENTS
# ==========================

app = Flask(__name__)
db = firestore.Client(project=PROJECT_ID, database=FS_DATABASE or None)
storage_client = storage.Client(project=PROJECT_ID)
publisher = pubsub_v1.PublisherClient()

docai_client_options = ClientOptions(api_endpoint=f"{DOCAI_LOCATION}-documentai.googleapis.com")
docai_client = documentai.DocumentProcessorServiceClient(client_options=docai_client_options)

next_topic_path = publisher.topic_path(PROJECT_ID, NEXT_TOPIC)

app.logger.info(
    f"OCR worker boot: "
    f"PROJECT_ID={PROJECT_ID}, "
    f"NEXT_TOPIC={NEXT_TOPIC}, "
    f"OUTPUT_BUCKET={OUTPUT_BUCKET}, "
    f"DOCAI_LOCATION={DOCAI_LOCATION}, "
    f"DOCAI_PROCESSOR_ID={DOCAI_PROCESSOR_ID}, "
    f"FS_COLLECTION={FS_COLLECTION}, "
    f"MAX_PAGES_PER_DOC={MAX_PAGES_PER_DOC}"
)

# ==========================
# HELPERS
# ==========================

def _now_iso() -> str:
    return dt.datetime.utcnow().isoformat() + "Z"

def _update_job(doc_id: str, payload: dict):
    payload["updatedAt"] = _now_iso()
    db.collection(FS_COLLECTION).document(doc_id).set(payload, merge=True)

def _publish_next(payload: dict):
    data = json.dumps(payload).encode("utf-8")
    msg_id = publisher.publish(next_topic_path, data).result(timeout=10)
    app.logger.info(f"Published -> {NEXT_TOPIC} msg_id={msg_id} docId={payload.get('docId')}")

def _parse_gcs_uri(gcs_uri: str) -> Tuple[str, str]:
    # returns (bucket, key)
    assert gcs_uri.startswith("gs://"), f"Bad GCS URI: {gcs_uri}"
    parts = gcs_uri[5:].split("/", 1)
    bucket = parts[0]
    key = parts[1] if len(parts) > 1 else ""
    return bucket, key

def _download_bytes(gcs_uri: str) -> bytes:
    bkt, key = _parse_gcs_uri(gcs_uri)
    blob = storage_client.bucket(bkt).blob(key)
    return blob.download_as_bytes()

def _upload_bytes_to_gcs(bkt: str, key: str, data: bytes, content_type: str):
    blob = storage_client.bucket(bkt).blob(key)
    blob.upload_from_string(data, content_type=content_type)
    return f"gs://{bkt}/{key}"

def _write_text_to_gcs(text_content: str, name_without_ext: str) -> str:
    out_name = f"{TXT_PREFIX}{name_without_ext}.txt"
    blob = storage_client.bucket(OUTPUT_BUCKET).blob(out_name)
    blob.upload_from_string(text_content, content_type="text/plain; charset=utf-8")
    gcs_uri = f"gs://{OUTPUT_BUCKET}/{out_name}"
    app.logger.info(f"Wrote text -> {gcs_uri}")
    return gcs_uri

def _is_part_doc(doc_id: str) -> bool:
    return "__part" in doc_id

def _parent_id(doc_id: str) -> str:
    # "abc__part001" -> "abc"
    return doc_id.split("__part", 1)[0]

def _child_index(doc_id: str) -> Optional[int]:
    # "abc__part007" -> 7
    if "__part" not in doc_id:
        return None
    suffix = doc_id.split("__part", 1)[1]
    try:
        return int(suffix)
    except ValueError:
        return None

# ==========================
# SPLITTING
# ==========================

def _split_pdf_to_chunks(gcs_uri: str, temp_prefix: str, max_pages: int) -> List[str]:
    """
    Download the source PDF from GCS, split into <= max_pages per file,
    and upload chunk PDFs alongside the original. Returns list of chunk URIs.
    """
    bkt, key = _parse_gcs_uri(gcs_uri)
    src_blob = storage_client.bucket(bkt).blob(key)
    pdf_bytes = src_blob.download_as_bytes()

    reader = PdfReader(BytesIO(pdf_bytes))
    total = len(reader.pages)
    app.logger.info(f"PDF has {total} pages. Splitting by {max_pages} pages.")

    chunks = []
    start = 0
    idx = 1
    while start < total:
        end = min(start + max_pages, total)
        writer = PdfWriter()
        for p in range(start, end):
            writer.add_page(reader.pages[p])

        out = BytesIO()
        writer.write(out)
        out.seek(0)

        chunk_name = f"{temp_prefix}chunk_{idx:03d}_{start+1:05d}-{end:05d}.pdf"
        _upload_bytes_to_gcs(bkt, chunk_name, out.getvalue(), "application/pdf")
        chunks.append(f"gs://{bkt}/{chunk_name}")

        start = end
        idx += 1

    return chunks

# ==========================
# DOCAI CALLS
# ==========================

def _start_docai_batch_job(gcs_uri: str, doc_id: str) -> Tuple[str, str]:
    """
    Start Document AI batch job for a single (possibly chunked) PDF.
    Returns (operation_name, output_subprefix).
    """
    processor_name = docai_client.processor_path(PROJECT_ID, DOCAI_LOCATION, DOCAI_PROCESSOR_ID)

    input_config = documentai.BatchProcessRequest.BatchInputConfig(
        gcs_source=documentai.GcsSource(uri=gcs_uri),
        mime_type="application/pdf"
    )

    # Write results to a stable folder under DOCAI_OUTPUT_PREFIX + doc_id + '/'
    output_gcs_uri = f"gs://{OUTPUT_BUCKET}/{DOCAI_OUTPUT_PREFIX}{doc_id}/"
    document_output_config = documentai.DocumentOutputConfig(
        gcs_output_config=documentai.DocumentOutputConfig.GcsOutputConfig(
            gcs_uri=output_gcs_uri
        )
    )

    request = documentai.BatchProcessRequest(
        name=processor_name,
        input_configs=[input_config],
        document_output_config=document_output_config,
        # Optional process options; do NOT assume this lifts batch limits
        # Add layout chunking only if your processor supports it
        # process_options=documentai.ProcessOptions(
        #     layout_config=documentai.ProcessOptions.LayoutConfig(
        #         chunking_config=documentai.ProcessOptions.LayoutConfig.ChunkingConfig(
        #             pages_per_chunk=150, overlap_pages=2
        #         )
        #     )
        # )
    )

    max_retries = 3
    delay = 2
    for attempt in range(max_retries):
        try:
            op = docai_client.batch_process_documents(request=request)
            app.logger.info(
                f"DocAI batch started for {doc_id} (attempt {attempt+1}/{max_retries}) "
                f"op={op.operation.name} out={output_gcs_uri}"
            )
            return op.operation.name, f"{DOCAI_OUTPUT_PREFIX}{doc_id}/"
        except (gcp_exceptions.ResourceExhausted, gcp_exceptions.Aborted, gcp_exceptions.Unavailable) as e:
            if attempt < max_retries - 1:
                app.logger.warning(f"Transient DocAI start error: {e}. Retry in {delay}s")
                time.sleep(delay)
                delay *= 2
            else:
                app.logger.error("DocAI start failed after retries.")
                raise
        except gcp_exceptions.InvalidArgument as e:
            em = str(e)
            if "page limit" in em.lower() or "quota" in em.lower():
                app.logger.critical(f"Critical API error: {em}")
            raise

def _read_docai_output_to_text(doc_id: str) -> str:
    """
    Reads DocAI JSON outputs for doc_id and returns the concatenated text of the first document.
    We look under: gs://OUTPUT_BUCKET/{DOCAI_OUTPUT_PREFIX}{doc_id}/<LRO_FOLDER>/document.json or output-*-shard*.json
    """
    docai_folder_prefix = f"{DOCAI_OUTPUT_PREFIX}{doc_id}/"
    app.logger.info(f"Reading DocAI output for {doc_id} from gs://{OUTPUT_BUCKET}/{docai_folder_prefix}")

    # Find the LRO folder created by DocAI
    iterator = storage_client.list_blobs(OUTPUT_BUCKET, prefix=docai_folder_prefix, delimiter="/")
    lro_prefixes = [p for p in iterator.prefixes if p.startswith(docai_folder_prefix)]

    if not lro_prefixes:
        raise FileNotFoundError(f"LRO folder not found for {doc_id} under {docai_folder_prefix}")

    lro_path = lro_prefixes[0]

    # Find JSON(s)
    result_blobs = list(storage_client.list_blobs(OUTPUT_BUCKET, prefix=lro_path))
    json_blobs = [b for b in result_blobs if b.name.endswith(".json") and "operation.json" not in b.name]
    if not json_blobs:
        raise FileNotFoundError(f"No DocAI result JSON for {doc_id} under {lro_path}")

    # Use the first JSON (single doc per batch input)
    raw = json_blobs[0].download_as_bytes()
    document = documentai.Document.from_json(raw, ignore_unknown_fields=True)
    return document.text or ""

# ==========================
# MERGE HELPERS
# ==========================

def _try_finalize_parent_if_ready(parent_id: str):
    """
    If all child parts have SUCCEEDED, merge their texts (in order) into {parent_id}.txt,
    update parent job, and publish downstream.
    """
    parent_ref = db.collection(FS_COLLECTION).document(parent_id)
    parent = parent_ref.get()
    if not parent.exists:
        app.logger.warning(f"Parent job {parent_id} missing; cannot finalize.")
        return

    pdata = parent.to_dict()
    total_parts = pdata.get("totalParts", 0)
    if total_parts <= 0:
        # Single-part workflow—nothing to aggregate
        return

    children = pdata.get("children", [])
    if len(children) != total_parts:
        app.logger.info(f"Parent {parent_id}: children count {len(children)}/{total_parts}. Not ready.")
        return

    # Check all child statuses
    all_ok = all(ch.get("status") == "SUCCEEDED" for ch in children)
    if not all_ok:
        app.logger.info(f"Parent {parent_id}: at least one child not SUCCEEDED. Not ready.")
        return

    # Concatenate text in part order
    # Expect artifacts.textUri on each child
    children_sorted = sorted(children, key=lambda x: x.get("partIndex", 0))
    full_text_parts = []
    for ch in children_sorted:
        text_uri = (ch.get("artifacts") or {}).get("textUri")
        if not text_uri:
            app.logger.error(f"Child {ch.get('docId')} missing artifacts.textUri; aborting finalize.")
            return
        full_text_parts.append(storage_client.bucket(_parse_gcs_uri(text_uri)[0]).blob(_parse_gcs_uri(text_uri)[1]).download_as_text())

    merged_text = "\n\f\n".join(full_text_parts)  # insert form-feed separators between chunks
    final_text_uri = _write_text_to_gcs(merged_text, parent_id)

    # Update parent artifacts + status
    parent_ref.set({
        "status": "SUCCEEDED",
        "phase": "OCR_DONE",
        "artifacts": {"textUri": final_text_uri},
        "updatedAt": _now_iso()
    }, merge=True)

    # Publish downstream once
    routing = pdata.get("routing", {})
    gcs_uri = pdata.get("gcsUri")  # original source
    _publish_next({
        "type": "ocr.done",
        "docId": parent_id,
        "textUri": final_text_uri,
        "gcsUri": gcs_uri,
        "routing": routing,
        "ts": _now_iso()
    })

    app.logger.info(f"FINALIZED parent {parent_id} -> {final_text_uri}")

# ==========================
# ROUTES
# ==========================

@app.get("/")
def health():
    return "ok", 200

@app.post("/work")
def work():
    """
    Pub/Sub push trigger for a new OCR job.
    Splits large PDFs into parts <= MAX_PAGES_PER_DOC, starts one batch per part,
    and creates a parent job to track/merge.
    """
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
    routing_info = payload.get("routing") or {}

    if not doc_id or not gcs_uri:
        app.logger.error("missing docId or gcsUri")
        return ("MISSING FIELDS", 200)

    try:
        # Split into chunks (always—cheap for small PDFs; returns single chunk when short)
        # We'll write chunks under the same bucket/path prefix as the original’s directory.
        bkt, key = _parse_gcs_uri(gcs_uri)
        base_dir = key.rsplit("/", 1)[0] + "/" if "/" in key else ""
        temp_prefix = f"{base_dir}{doc_id}__chunks/"
        chunk_uris = _split_pdf_to_chunks(gcs_uri, temp_prefix, MAX_PAGES_PER_DOC)

        if len(chunk_uris) == 1:
            # Single-part path (no parent/child complexity)
            op_name, _ = _start_docai_batch_job(chunk_uris[0], doc_id)
            _update_job(doc_id, {
                "status": "PENDING",
                "phase": "DOCAI_BATCH_SENT",
                "lroName": op_name,
                "ocr": {"processorId": DOCAI_PROCESSOR_ID, "location": DOCAI_LOCATION},
                "routing": routing_info,
                "gcsUri": gcs_uri,
                "totalParts": 0,     # signifies "not a parent"
            })
        else:
            # Parent + children
            total = len(chunk_uris)
            children_meta = []
            for idx, uri in enumerate(chunk_uris, start=1):
                child_id = f"{doc_id}__part{idx:03d}"
                op_name, _ = _start_docai_batch_job(uri, child_id)
                _update_job(child_id, {
                    "status": "PENDING",
                    "phase": "DOCAI_BATCH_SENT",
                    "lroName": op_name,
                    "ocr": {"processorId": DOCAI_PROCESSOR_ID, "location": DOCAI_LOCATION},
                    "routing": routing_info,
                    "gcsUri": gcs_uri,
                    "parentId": doc_id,
                    "partIndex": idx,
                    "totalParts": total
                })
                children_meta.append({"docId": child_id, "status": "PENDING", "partIndex": idx})

            _update_job(doc_id, {
                "status": "PENDING",
                "phase": "DOCAI_CHILDREN_RUNNING",
                "routing": routing_info,
                "gcsUri": gcs_uri,
                "totalParts": total,
                "children": children_meta
            })

        return ("OK", 200)

    except Exception as e:
        app.logger.error(f"Failed to start job for {doc_id}: {e}")
        if doc_id:
            _update_job(doc_id, {
                "status": "FAILED",
                "error": str(e),
                "phase": "DOCAI_BATCH_START_ERROR",
                "routing": routing_info,
                "gcsUri": gcs_uri
            })
        return ("FAILED", 200)  # 200 to stop Pub/Sub retries for malformed payloads

@app.post("/docai_result")
def docai_result():
    """
    Triggered by GCS notification (Pub/Sub push) when DocAI writes outputs.
    We only react to files named 'operation.json'.
    For child parts, we write per-part text and then attempt parent finalization.
    For single-part jobs, we write final text and publish immediately.
    """
    doc_id = None
    try:
        envelope = request.get_json(silent=True)
        if not envelope or "message" not in envelope:
            abort(400, "Bad Pub/Sub push")

        message = envelope.get("message", {})
        data_encoded = message.get("data")
        if not data_encoded:
            raise ValueError("No data field in GCS Pub/Sub message.")

        gcs_event = json.loads(base64.b64decode(data_encoded).decode("utf-8"))
        object_name = gcs_event.get("name")  # bucket object path

        if not object_name or "operation.json" not in object_name:
            app.logger.info(f"Ignoring object: {object_name}")
            return ("OK - ignore", 200)

        # Path format: DOCAI_OUTPUT_PREFIX + {doc_id} + /LRO_FOLDER/operation.json
        # Strip the configured prefix and extract doc_id
        if not object_name.startswith(DOCAI_OUTPUT_PREFIX):
            app.logger.info(f"Ignoring operation outside prefix: {object_name}")
            return ("OK - ignore", 200)

        sub_path = object_name[len(DOCAI_OUTPUT_PREFIX):]
        doc_id = sub_path.split("/", 1)[0]
        if not doc_id:
            raise ValueError(f"Could not extract doc_id from: {object_name}")

        # Read text from DocAI result
        ocr_text = _read_docai_output_to_text(doc_id)
        text_uri = _write_text_to_gcs(ocr_text, doc_id)

        # Update this doc (child or single)
        _update_job(doc_id, {
            "status": "SUCCEEDED",
            "phase": "OCR_DONE",
            "artifacts": {"textUri": text_uri}
        })

        if _is_part_doc(doc_id):
            # Update parent children array with this child's success
            parent = _parent_id(doc_id)
            part_idx = _child_index(doc_id) or 0

            parent_ref = db.collection(FS_COLLECTION).document(parent)
            parent_snap = parent_ref.get()
            if parent_snap.exists:
                pdata = parent_snap.to_dict()
                children = pdata.get("children", [])
                for ch in children:
                    if ch.get("docId") == doc_id:
                        ch["status"] = "SUCCEEDED"
                        ch["artifacts"] = {"textUri": text_uri}
                        ch["partIndex"] = part_idx
                        break
                parent_ref.set({"children": children, "updatedAt": _now_iso()}, merge=True)

                # Try finalize if all children done
                _try_finalize_parent_if_ready(parent)
            else:
                app.logger.warning(f"Parent job {parent} missing for child {doc_id}.")
        else:
            # Single-part job -> publish immediately
            job_snap = db.collection(FS_COLLECTION).document(doc_id).get()
            routing = {}
            gcs_uri = None
            if job_snap.exists:
                j = job_snap.to_dict()
                routing = j.get("routing", {})
                gcs_uri = j.get("gcsUri")

            _publish_next({
                "type": "ocr.done",
                "docId": doc_id,
                "textUri": text_uri,
                "gcsUri": gcs_uri,
                "routing": routing,
                "ts": _now_iso()
            })

        return ("OK", 200)

    except Exception as e:
        app.logger.error(f"DocAI result processing failed for {doc_id}: {e}")
        if doc_id:
            try:
                _update_job(doc_id, {
                    "status": "FAILED",
                    "error": str(e),
                    "phase": "DOCAI_RESULT_ERROR"
                })
                # If this was a child and it failed, propagate failure to parent for visibility
                if _is_part_doc(doc_id):
                    parent = _parent_id(doc_id)
                    parent_ref = db.collection(FS_COLLECTION).document(parent)
                    snap = parent_ref.get()
                    if snap.exists:
                        pdata = snap.to_dict()
                        children = pdata.get("children", [])
                        for ch in children:
                            if ch.get("docId") == doc_id:
                                ch["status"] = "FAILED"
                                ch["error"] = str(e)
                                break
                        parent_ref.set({"children": children, "status": "FAILED", "phase": "CHILD_FAILED", "updatedAt": _now_iso()}, merge=True)
            except Exception as ue:
                app.logger.error(f"Failed to update job status after error: {ue}")

        # Return 500 so Pub/Sub retries transiently
        return (f"ERROR: {e}", 500)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
