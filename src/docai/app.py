import os
import json
import base64
import datetime as dt
import time
from io import BytesIO
from typing import List, Tuple, Optional

from flask import Flask, request, abort

# deps
from pypdf import PdfReader, PdfWriter
from google.cloud import firestore, storage, pubsub_v1
from google.cloud import documentai_v1 as documentai
from google.api_core.client_options import ClientOptions
from google.api_core import exceptions as gcp_exceptions


# --------------------------------------------------------------------------------------
# App & globals (clients/env are loaded lazily to avoid import-time crashes in Cloud Run)
# --------------------------------------------------------------------------------------

app = Flask(__name__)

# Lazy-initialized singletons
_clients = {
    "firestore": None,
    "storage": None,
    "publisher": None,
    "docai": None,
    "next_topic_path": None,
}

_env = None  # will hold validated env dict after first access


def _now_iso() -> str:
    return dt.datetime.utcnow().isoformat() + "Z"


def _load_env() -> dict:
    """Read & validate env, but only once."""
    global _env
    if _env is not None:
        return _env

    env = {}
    # Project
    env["PROJECT_ID"] = os.environ.get("GOOGLE_CLOUD_PROJECT") or os.environ.get("PROJECT_ID")

    # Firestore (optional FS_DATABASE)
    env["FS_DATABASE"] = os.environ.get("FS_DATABASE")
    env["FS_COLLECTION"] = os.environ.get("FS_COLLECTION") or "jobs"

    # Storage (required)
    env["OUTPUT_BUCKET"] = os.environ.get("OUTPUT_BUCKET")
    env["TXT_PREFIX"] = os.environ.get("TXT_PREFIX", "")

    # DocAI (required)
    env["DOCAI_LOCATION"] = os.environ.get("DOCAI_LOCATION")
    env["DOCAI_PROCESSOR_ID"] = os.environ.get("DOCAI_PROCESSOR_ID")
    env["DOCAI_OUTPUT_PREFIX"] = os.environ.get("DOCAI_OUTPUT_PREFIX")

    # Pub/Sub (required)
    env["NEXT_TOPIC"] = os.environ.get("NEXT_TOPIC")

    # Chunking (defaults to 15 for your processor)
    env["MAX_PAGES_PER_DOC"] = int(os.environ.get("MAX_PAGES_PER_DOC", "15"))

    # Validate requireds but don't raise at import time; raise on first request if missing
    missing = [
        k for k in
        ["PROJECT_ID", "OUTPUT_BUCKET", "DOCAI_LOCATION", "DOCAI_PROCESSOR_ID",
         "DOCAI_OUTPUT_PREFIX", "NEXT_TOPIC"]
        if not env.get(k)
    ]
    if missing:
        raise RuntimeError(f"Missing required environment variables: {', '.join(missing)}")

    _env = env
    return env


def _get_clients():
    """Create GCP clients lazily & cache them."""
    env = _load_env()
    if _clients["storage"] is None:
        _clients["storage"] = storage.Client(project=env["PROJECT_ID"])
    if _clients["firestore"] is None:
        if env["FS_DATABASE"]:
            _clients["firestore"] = firestore.Client(project=env["PROJECT_ID"], database=env["FS_DATABASE"])
        else:
            _clients["firestore"] = firestore.Client(project=env["PROJECT_ID"])
    if _clients["publisher"] is None:
        _clients["publisher"] = pubsub_v1.PublisherClient()
        _clients["next_topic_path"] = _clients["publisher"].topic_path(env["PROJECT_ID"], env["NEXT_TOPIC"])
    if _clients["docai"] is None:
        opts = ClientOptions(api_endpoint=f'{env["DOCAI_LOCATION"]}-documentai.googleapis.com')
        _clients["docai"] = documentai.DocumentProcessorServiceClient(client_options=opts)
    return _clients


# -----------------------
# Utility helper functions
# -----------------------

def _update_job(doc_id: str, payload: dict):
    env = _load_env()
    db = _get_clients()["firestore"]
    payload["updatedAt"] = _now_iso()
    db.collection(env["FS_COLLECTION"]).document(doc_id).set(payload, merge=True)


def _publish_next(payload: dict):
    pub = _get_clients()["publisher"]
    topic = _get_clients()["next_topic_path"]
    data = json.dumps(payload).encode("utf-8")
    msg_id = pub.publish(topic, data).result(timeout=10)
    app.logger.info(f"Published -> {topic} msg_id={msg_id} docId={payload.get('docId')}")


def _parse_gcs_uri(gcs_uri: str) -> Tuple[str, str]:
    assert gcs_uri.startswith("gs://"), f"Bad GCS URI: {gcs_uri}"
    parts = gcs_uri[5:].split("/", 1)
    bucket = parts[0]
    key = parts[1] if len(parts) > 1 else ""
    return bucket, key


def _write_text_to_gcs(text_content: str, name_without_ext: str) -> str:
    env = _load_env()
    storage_client = _get_clients()["storage"]
    out_name = f"{env['TXT_PREFIX']}{name_without_ext}.txt"
    blob = storage_client.bucket(env["OUTPUT_BUCKET"]).blob(out_name)
    blob.upload_from_string(text_content, content_type="text/plain; charset=utf-8")
    gcs_uri = f"gs://{env['OUTPUT_BUCKET']}/{out_name}"
    app.logger.info(f"Wrote text -> {gcs_uri}")
    return gcs_uri


def _is_part_doc(doc_id: str) -> bool:
    return "__part" in doc_id


def _parent_id(doc_id: str) -> str:
    return doc_id.split("__part", 1)[0]


def _child_index(doc_id: str) -> Optional[int]:
    if "__part" not in doc_id:
        return None
    suffix = doc_id.split("__part", 1)[1]
    try:
        return int(suffix)
    except ValueError:
        return None


# -----------
# PDF splitting
# -----------

def _split_pdf_to_chunks(gcs_uri: str, temp_prefix: str, max_pages: int) -> List[str]:
    env = _load_env()
    storage_client = _get_clients()["storage"]
    bkt, key = _parse_gcs_uri(gcs_uri)
    src_blob = storage_client.bucket(bkt).blob(key)
    pdf_bytes = src_blob.download_as_bytes()

    reader = PdfReader(BytesIO(pdf_bytes))
    if reader.is_encrypted:
        try:
            reader.decrypt("")  # try empty password if encrypted with no password
        except Exception:
            raise RuntimeError("Encrypted PDF not supported by this worker.")

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
        dst_blob = storage_client.bucket(bkt).blob(chunk_name)
        dst_blob.upload_from_file(out, content_type="application/pdf")
        chunks.append(f"gs://{bkt}/{chunk_name}")

        start = end
        idx += 1

    return chunks


# --------------------
# Document AI operations
# --------------------

def _start_docai_batch_job(gcs_uri: str, doc_id: str) -> Tuple[str, str]:
    env = _load_env()
    docai_client = _get_clients()["docai"]
    storage_client = _get_clients()["storage"]

    processor_name = docai_client.processor_path(
        env["PROJECT_ID"], env["DOCAI_LOCATION"], env["DOCAI_PROCESSOR_ID"]
    )

    input_config = documentai.BatchProcessRequest.BatchInputConfig(
        gcs_source=documentai.GcsSource(uri=gcs_uri),
        mime_type="application/pdf"
    )

    output_gcs_uri = f"gs://{env['OUTPUT_BUCKET']}/{env['DOCAI_OUTPUT_PREFIX']}{doc_id}/"
    document_output_config = documentai.DocumentOutputConfig(
        gcs_output_config=documentai.DocumentOutputConfig.GcsOutputConfig(
            gcs_uri=output_gcs_uri
        )
    )

    request = documentai.BatchProcessRequest(
        name=processor_name,
        input_configs=[input_config],
        document_output_config=document_output_config,
        # If your processor supports internal chunking, you could add ProcessOptions here.
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
            return op.operation.name, f"{env['DOCAI_OUTPUT_PREFIX']}{doc_id}/"
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
    """Read DocAI JSON (first doc) and return text."""
    env = _load_env()
    storage_client = _get_clients()["storage"]

    prefix = f"{env['DOCAI_OUTPUT_PREFIX']}{doc_id}/"
    app.logger.info(f"Reading DocAI output for {doc_id} from gs://{env['OUTPUT_BUCKET']}/{prefix}")

    # list with delimiter to discover the LRO folder
    iterator = storage_client.list_blobs(env["OUTPUT_BUCKET"], prefix=prefix, delimiter="/")
    # consume iterator to populate prefixes (google-cloud-storage quirk)
    for _ in iterator:
        pass
    lro_prefixes = list(getattr(iterator, "prefixes", []))
    if not lro_prefixes:
        raise FileNotFoundError(f"LRO folder not found for {doc_id} under {prefix}")
    lro_path = lro_prefixes[0]

    # list JSON files under LRO path (excluding operation.json)
    json_candidates = []
    for b in storage_client.list_blobs(env["OUTPUT_BUCKET"], prefix=lro_path):
        if b.name.endswith(".json") and not b.name.endswith("operation.json"):
            json_candidates.append(b)
    if not json_candidates:
        raise FileNotFoundError(f"No DocAI result JSON for {doc_id} under {lro_path}")

    raw_bytes = json_candidates[0].download_as_bytes()
    # IMPORTANT: from_json expects a string, not bytes
    document = documentai.Document.from_json(raw_bytes.decode("utf-8"), ignore_unknown_fields=True)
    return document.text or ""


# ------------
# Merge helpers
# ------------

def _try_finalize_parent_if_ready(parent_id: str):
    env = _load_env()
    db = _get_clients()["firestore"]
    storage_client = _get_clients()["storage"]

    parent_ref = db.collection(env["FS_COLLECTION"]).document(parent_id)
    snap = parent_ref.get()
    if not snap.exists:
        app.logger.warning(f"Parent job {parent_id} missing; cannot finalize.")
        return
    pdata = snap.to_dict()

    total_parts = pdata.get("totalParts", 0)
    if total_parts <= 0:
        return

    children = pdata.get("children", [])
    if len(children) != total_parts:
        app.logger.info(f"Parent {parent_id}: children {len(children)}/{total_parts}. Not ready.")
        return
    if not all(ch.get("status") == "SUCCEEDED" for ch in children):
        app.logger.info(f"Parent {parent_id}: not all SUCCEEDED.")
        return

    # merge in order by partIndex
    children_sorted = sorted(children, key=lambda x: x.get("partIndex", 0))
    text_parts = []
    for ch in children_sorted:
        text_uri = (ch.get("artifacts") or {}).get("textUri")
        if not text_uri:
            app.logger.error(f"Child {ch.get('docId')} missing textUri; aborting finalize.")
            return
        bkt, key = _parse_gcs_uri(text_uri)
        text = storage_client.bucket(bkt).blob(key).download_as_text()
        text_parts.append(text)

    merged = "\n\f\n".join(text_parts)
    final_text_uri = _write_text_to_gcs(merged, parent_id)

    parent_ref.set({
        "status": "SUCCEEDED",
        "phase": "OCR_DONE",
        "artifacts": {"textUri": final_text_uri},
        "updatedAt": _now_iso()
    }, merge=True)

    routing = pdata.get("routing", {})
    gcs_uri = pdata.get("gcsUri")
    _publish_next({
        "type": "ocr.done",
        "docId": parent_id,
        "textUri": final_text_uri,
        "gcsUri": gcs_uri,
        "routing": routing,
        "ts": _now_iso()
    })
    app.logger.info(f"FINALIZED {parent_id} -> {final_text_uri}")


# ------
# Routes
# ------

@app.get("/")
def health():
    return "ok", 200


@app.post("/work")
def work():
    # ensure env/clients exist (surface missing env clearly)
    _ = _get_clients()

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
        env = _load_env()
        # Split always; cheap for small PDFs
        bkt, key = _parse_gcs_uri(gcs_uri)
        base_dir = key.rsplit("/", 1)[0] + "/" if "/" in key else ""
        temp_prefix = f"{base_dir}{doc_id}__chunks/"
        chunk_uris = _split_pdf_to_chunks(gcs_uri, temp_prefix, env["MAX_PAGES_PER_DOC"])

        if len(chunk_uris) == 1:
            op_name, _ = _start_docai_batch_job(chunk_uris[0], doc_id)
            _update_job(doc_id, {
                "status": "PENDING",
                "phase": "DOCAI_BATCH_SENT",
                "lroName": op_name,
                "ocr": {"processorId": env["DOCAI_PROCESSOR_ID"], "location": env["DOCAI_LOCATION"]},
                "routing": routing_info,
                "gcsUri": gcs_uri,
                "totalParts": 0
            })
        else:
            total = len(chunk_uris)
            children_meta = []
            for idx, uri in enumerate(chunk_uris, start=1):
                child_id = f"{doc_id}__part{idx:03d}"
                op_name, _ = _start_docai_batch_job(uri, child_id)
                _update_job(child_id, {
                    "status": "PENDING",
                    "phase": "DOCAI_BATCH_SENT",
                    "lroName": op_name,
                    "ocr": {"processorId": env["DOCAI_PROCESSOR_ID"], "location": env["DOCAI_LOCATION"]},
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
        return ("FAILED", 200)


@app.post("/docai_result")
def docai_result():
    # ensure env/clients exist
    _ = _get_clients()

    doc_id = None
    try:
        env = _load_env()

        envelope = request.get_json(silent=True)
        if not envelope or "message" not in envelope:
            abort(400, "Bad Pub/Sub push")

        message = envelope.get("message", {})
        data_encoded = message.get("data")
        if not data_encoded:
            raise ValueError("No data field in GCS Pub/Sub message.")

        event = json.loads(base64.b64decode(data_encoded).decode("utf-8"))
        object_name = event.get("name")
        if not object_name or "operation.json" not in object_name:
            app.logger.info(f"Ignoring object: {object_name}")
            return ("OK - ignore", 200)

        if not object_name.startswith(env["DOCAI_OUTPUT_PREFIX"]):
            app.logger.info(f"Ignoring operation outside prefix: {object_name}")
            return ("OK - ignore", 200)

        sub_path = object_name[len(env["DOCAI_OUTPUT_PREFIX"]):]
        doc_id = sub_path.split("/", 1)[0]
        if not doc_id:
            raise ValueError(f"Could not extract doc_id from: {object_name}")

        # Pull text and write {doc_id}.txt (or per-part)
        ocr_text = _read_docai_output_to_text(doc_id)
        text_uri = _write_text_to_gcs(ocr_text, doc_id)

        _update_job(doc_id, {
            "status": "SUCCEEDED",
            "phase": "OCR_DONE",
            "artifacts": {"textUri": text_uri}
        })

        if _is_part_doc(doc_id):
            parent = _parent_id(doc_id)
            part_idx = _child_index(doc_id) or 0
            db = _get_clients()["firestore"]
            parent_ref = db.collection(env["FS_COLLECTION"]).document(parent)
            snap = parent_ref.get()
            if snap.exists:
                pdata = snap.to_dict()
                children = pdata.get("children", [])
                for ch in children:
                    if ch.get("docId") == doc_id:
                        ch["status"] = "SUCCEEDED"
                        ch["artifacts"] = {"textUri": text_uri}
                        ch["partIndex"] = part_idx
                        break
                parent_ref.set({"children": children, "updatedAt": _now_iso()}, merge=True)
                _try_finalize_parent_if_ready(parent)
            else:
                app.logger.warning(f"Parent job {parent} missing for child {doc_id}.")
        else:
            # single part -> publish now
            db = _get_clients()["firestore"]
            job_snap = db.collection(env["FS_COLLECTION"]).document(doc_id).get()
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
                if _is_part_doc(doc_id):
                    parent = _parent_id(doc_id)
                    env = _load_env()
                    db = _get_clients()["firestore"]
                    parent_ref = db.collection(env["FS_COLLECTION"]).document(parent)
                    snap = parent_ref.get()
                    if snap.exists:
                        pdata = snap.to_dict()
                        children = pdata.get("children", [])
                        for ch in children:
                            if ch.get("docId") == doc_id:
                                ch["status"] = "FAILED"
                                ch["error"] = str(e)
                                break
                        parent_ref.set({"children": children, "status": "FAILED",
                                        "phase": "CHILD_FAILED", "updatedAt": _now_iso()}, merge=True)
            except Exception as ue:
                app.logger.error(f"Failed to update job status after error: {ue}")

        return (f"ERROR: {e}", 500)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8080"))
    app.run(host="0.0.0.0", port=port, debug=True)
