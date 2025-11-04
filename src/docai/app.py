import os
import json
import base64
import datetime as dt
import time # Import time for retries
from flask import Flask, request, abort
from google.cloud import firestore, storage, pubsub_v1
from google.cloud import documentai_v1 as documentai
from google.api_core.client_options import ClientOptions
from google.api_core import exceptions as gcp_exceptions 

# --- HELPER TO ENFORCE ENV VARS ---
def _get_env_or_raise(name: str) -> str:
    """Gets an environment variable or raises a RuntimeError if it's not set."""
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"Required environment variable '{name}' is not set.")
    return value

# --- ENVIRONMENT VARIABLES & CONFIGURATION ---

# Project and location variables
PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT") or os.environ.get("PROJECT_ID")
if not PROJECT_ID:
    # Retaining existing check for PROJECT_ID
    raise RuntimeError("PROJECT_ID / GOOGLE_CLOUD_PROJECT must be set")

# Firestore configuration (setting non-critical ones to None lets the library use its default)
FS_DATABASE = os.environ.get("FS_DATABASE")
FS_COLLECTION = os.environ.get("FS_COLLECTION")

# Storage configuration (Mandatory)
OUTPUT_BUCKET = _get_env_or_raise("OUTPUT_BUCKET")
TXT_PREFIX = os.environ.get("TXT_PREFIX") # Optional prefix, defaults to None if unset

# Document AI configuration (Mandatory)
DOCAI_LOCATION = _get_env_or_raise("DOCAI_LOCATION")
DOCAI_PROCESSOR_ID = os.environ.get("DOCAI_PROCESSOR_ID") # Checked later, as it might be None for testing
# GCS prefix where Document AI will write its results (Mandatory for async lookup)
DOCAI_OUTPUT_PREFIX = _get_env_or_raise("DOCAI_OUTPUT_PREFIX")

# Pub/Sub configuration (Mandatory)
NEXT_TOPIC = _get_env_or_raise("NEXT_TOPIC")

# --- CLIENT INITIALIZATION ---
app = Flask(__name__)
# Pass None for FS_DATABASE if unset, allowing Firestore client to use default
db = firestore.Client(project=PROJECT_ID, database=FS_DATABASE) 
storage_client = storage.Client(project=PROJECT_ID)
publisher = pubsub_v1.PublisherClient()

# Initialize Document AI client with region-specific endpoint
docai_client_options = ClientOptions(api_endpoint=f"{DOCAI_LOCATION}-documentai.googleapis.com")
docai_client = documentai.DocumentProcessorServiceClient(client_options=docai_client_options)

next_topic_path = publisher.topic_path(PROJECT_ID, NEXT_TOPIC)

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

# --- HELPER FUNCTIONS ---

def _now_iso() -> str:
    """Returns current time in ISO 8601 format (your preferred format)."""
    return dt.datetime.utcnow().isoformat() + "Z"

def _update_job(doc_id: str, payload: dict):
    """Updates the Firestore job status document using FS_COLLECTION."""
    # Use FS_COLLECTION or a fallback if it was not explicitly set
    collection_name = FS_COLLECTION if FS_COLLECTION else "jobs"
    payload["updatedAt"] = _now_iso()
    db.collection(collection_name).document(doc_id).set(payload, merge=True)

def _publish_next(payload: dict):
    """Publishes a message to the next Pub/Sub topic."""
    data = json.dumps(payload).encode("utf-8")
    msg_id = publisher.publish(next_topic_path, data).result(timeout=10)
    app.logger.info(f"Published to {NEXT_TOPIC} msg_id={msg_id} for docId={payload.get('docId')}")

def _write_text_to_gcs(text_content: str, doc_id: str) -> str:
    """Writes the extracted text content to GCS using TXT_PREFIX."""
    # Prepend prefix if set, otherwise default to just the doc_id name
    prefix = TXT_PREFIX if TXT_PREFIX else ""
    out_name = f"{prefix}{doc_id}.txt"
    blob = storage_client.bucket(OUTPUT_BUCKET).blob(out_name)
    blob.upload_from_string(text_content, content_type="text/plain; charset=utf-8")
    gcs_uri = f"gs://{OUTPUT_BUCKET}/{out_name}"
    app.logger.info(f"Text content written to {gcs_uri}")
    return gcs_uri

# --- DOCUMENT AI ASYNCHRONOUS BATCH LOGIC ---

def _start_docai_batch_job(gcs_uri: str, doc_id: str) -> tuple[str, str]:
    """
    Starts an asynchronous Document AI batch process with retry logic.
    """
    if not DOCAI_PROCESSOR_ID:
        raise ValueError("DOCAI_PROCESSOR_ID environment variable is not set.")

    processor_name = docai_client.processor_path(
        PROJECT_ID, DOCAI_LOCATION, DOCAI_PROCESSOR_ID
    )

    # 1. Input Configuration
    input_config = documentai.BatchProcessRequest.BatchInputConfig(
        gcs_source=documentai.GcsSource(
            uri=gcs_uri
        ),
        mime_type="application/pdf"
    )

    # 2. Output Configuration (CRITICAL: Use doc_id in the prefix for easy retrieval)
    output_gcs_uri = f"gs://{OUTPUT_BUCKET}/{DOCAI_OUTPUT_PREFIX}{doc_id}/"
    document_output_config = documentai.DocumentOutputConfig(
        gcs_output_config=documentai.DocumentOutputConfig.GcsOutputConfig(
            gcs_uri=output_gcs_uri,
        )
    )

    # 3. Create the Batch Request
    request = documentai.BatchProcessRequest(
        name=processor_name,
        input_configs=[input_config],
        document_output_config=document_output_config,
        # FIX: Using the non-deprecated field for Imageless Mode. 
        # This is equivalent to setting imageless_mode=true and should solve the page limit error.
        process_options=documentai.ProcessOptions(
            ocr_options=documentai.ProcessOptions.OcrOptions(
                disable_native_pdf_parsing=True, # The correct modern field to enable imageless mode
            )
        )
    )

    # 4. Start the Long Running Operation (LRO) with retry logic
    max_retries = 3
    delay = 2 # seconds
    for attempt in range(max_retries):
        try:
            operation = docai_client.batch_process_documents(request=request)
            
            app.logger.info(
                f"Batch Operation started (Attempt {attempt + 1}/{max_retries}): {operation.operation.name}. "
                f"Results will be written to {output_gcs_uri}"
            )
            
            # Success, break loop
            return operation.operation.name, f"{DOCAI_OUTPUT_PREFIX}{doc_id}/"

        # Catch transient errors (like quota or network issues)
        except (gcp_exceptions.ResourceExhausted, gcp_exceptions.Aborted, gcp_exceptions.Unavailable) as e:
            if attempt < max_retries - 1:
                app.logger.warning(f"Transient error on DocAI start (Attempt {attempt + 1}): {e}. Retrying in {delay}s...")
                time.sleep(delay)
                delay *= 2  # Exponential backoff
            else:
                app.logger.error(f"DocAI start failed after {max_retries} attempts.")
                raise # Re-raise the exception if final attempt fails
        
        # Catch hard configuration errors (like InvalidArgument) immediately
        except gcp_exceptions.InvalidArgument as e:
             # Explicitly catch quota/limit errors and log them as critical
            error_message = str(e)
            if "page limit" in error_message.lower() or "quota" in error_message.lower():
                app.logger.critical(f"Critical API Error: {error_message}. Check project quota.")
            raise # Re-raise the exception for the /work handler to catch

def _read_docai_output(doc_id: str) -> str:
    """
    Reads the DocAI output JSON file(s) from the dedicated output folder
    and extracts the full text. This logic is required to get the text.
    """
    # The full prefix is the DOCAI_OUTPUT_PREFIX + doc_id
    docai_folder_prefix = f"{DOCAI_OUTPUT_PREFIX}{doc_id}/"
    app.logger.info(f"Attempting to read DocAI output for {doc_id} from bucket={OUTPUT_BUCKET} prefix={docai_folder_prefix}")

    # 1. List blobs to find the dynamically generated LRO folder
    blobs = storage_client.list_blobs(
        OUTPUT_BUCKET, prefix=docai_folder_prefix, delimiter="/"
    )
    
    # Get the LRO sub-folder prefix (e.g., docai-results/job_12345/projects_..._operations_abc/)
    lro_folder_prefixes = [p for p in blobs.prefixes if p.startswith(docai_folder_prefix)]
    
    if not lro_folder_prefixes:
        app.logger.error(f"DocAI result LRO folder not found in {OUTPUT_BUCKET}/{docai_folder_prefix}. Check bucket contents/permissions.")
        raise FileNotFoundError(f"LRO folder not found in {docai_folder_prefix}")

    # Use the first LRO folder found
    lro_folder_path = lro_folder_prefixes[0] 
    app.logger.info(f"Found LRO path: {lro_folder_path}")

    # 2. List all document JSON files inside the detected LRO folder
    # We specifically look for "output-*-to-*-shard*.json" or "document.json"
    result_blobs = list(storage_client.list_blobs(
        OUTPUT_BUCKET, prefix=lro_folder_path, match_glob="**/*.json"
    ))
    
    if not result_blobs:
        app.logger.error(f"No DocAI JSON output found for {doc_id} at {lro_folder_path}. Listing returned zero files.")
        raise FileNotFoundError(f"No result JSON found for {doc_id} under LRO path.")
    
    app.logger.info(f"Found {len(result_blobs)} JSON result files. Processing the first one: {result_blobs[0].name}")

    # Download and parse the first result JSON (assuming a single document in the batch)
    result_blob = result_blobs[0]
    json_data = result_blob.download_as_bytes()
    
    # Document AI output is in Protocol Buffer format (JSON-encoded)
    document = documentai.Document.from_json(json_data, ignore_unknown_fields=True)
    
    # Extract the text
    ocr_text = document.text or ""
    
    if not ocr_text:
        app.logger.warning(f"Extracted text was empty for doc_id {doc_id} from file {result_blob.name}. DocAI may have failed to process the document content.")
        
    return ocr_text

# --- ENDPOINTS ---

@app.get("/")
def health():
    """Simple health check endpoint."""
    return "ok", 200

@app.post("/work")
def work():
    """
    Initial Pub/Sub trigger for a new job.
    This now STARTs the Document AI batch job (asynchronously) and exits.
    """
    # Pub/Sub push envelope
    envelope = request.get_json(silent=True)
    if not envelope or "message" not in envelope:
        abort(400, "Bad Pub/Sub push")

    msg = envelope["message"]
    data_b64 = msg.get("data", "")
    doc_id = None

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
        # --- LOGIC: Start the Asynchronous Batch Job with Retries ---
        lro_name, docai_output_prefix = _start_docai_batch_job(gcs_uri, doc_id)

        # Update Firestore job status to PENDING
        _update_job(doc_id, {
            "status": "PENDING",
            "phase": "DOCAI_BATCH_SENT",
            "lroName": lro_name,
            "ocr": {
                "processorId": DOCAI_PROCESSOR_ID,
                "location": DOCAI_LOCATION
            },
            # Store routing info for the next stage to use
            "routing": routing_info, 
        })
        
        app.logger.info(f"Batch job started successfully for {doc_id}. LRO: {lro_name}")
        return ("OK", 200) # IMPORTANT: Return immediately!

    except Exception as e:
        error_message = str(e)
        app.logger.error(f"Failed to start batch job for {doc_id}: {error_message}")
        
        # Update job with failure status
        if doc_id:
            _update_job(doc_id, {
                "status": "FAILED", 
                "error": error_message, 
                "phase": "DOCAI_BATCH_START_ERROR",
                "routing": routing_info,
            })
        # Return 200 so Pub/Sub does not retry (the job will never succeed)
        return ("FAILED", 200)


@app.post("/docai_result")
def docai_result():
    """
    Endpoint triggered by the GCS Notification Pub/Sub message when
    Document AI writes the output file(s) to the output bucket.
    This completes the asynchronous job.
    """
    doc_id = None
    try:
        envelope = request.get_json(silent=True)
        if not envelope or "message" not in envelope:
            abort(400, "Bad Pub/Sub push")
        
        # 1. Parse the Cloud Storage Notification
        message = envelope.get("message", {})
        data_encoded = message.get("data")
        if not data_encoded:
            raise ValueError("No data field in GCS Pub/Sub message.")
        
        # The data contains the Cloud Storage Event JSON
        data = json.loads(base64.b64decode(data_encoded).decode("utf-8"))
        
        output_object_name = data.get("name")
        
        # We only care about the file that confirms LRO completion.
        if "operation.json" not in output_object_name:
             app.logger.info(f"Ignoring file: {output_object_name}")
             return ("OK - Ignoring partial file", 200)

        app.logger.info(f"DocAI operation.json received: {output_object_name}")

        # 2. Extract the doc_id from the GCS output path.
        # Path format: {DOCAI_OUTPUT_PREFIX}{doc_id}/LRO_FOLDER_ID/operation.json
        lro_sub_path = output_object_name.replace(DOCAI_OUTPUT_PREFIX, "", 1)
        doc_id = lro_sub_path.split("/")[0]

        if not doc_id:
             raise ValueError(f"Could not extract doc_id from path: {output_object_name}")
        
        # 3. Retrieve the job information from Firestore
        # Use FS_COLLECTION or a fallback if it was not explicitly set
        collection_name = FS_COLLECTION if FS_COLLECTION else "jobs"
        job_doc = db.collection(collection_name).document(doc_id).get()
        if not job_doc.exists:
            app.logger.warning(f"Job document {doc_id} not found. Continuing without full job data. Original source GCS URI cannot be determined.")
            routing_info = {}
            # gcsUri is stored in the job document. If the document is missing,
            # we cannot reliably construct the original file location, so we set it to None.
            gcs_uri = None 
        else:
            job_data = job_doc.to_dict()
            routing_info = job_data.get("routing", {})
            gcs_uri = job_data.get("gcsUri") # Should be available from the job start


        # 4. Read DocAI output and extract text (This step is required to get the text)
        ocr_text = _read_docai_output(doc_id)

        # 5. Write raw text to the 'txt/' GCS folder (Your desired output)
        text_uri = _write_text_to_gcs(ocr_text, doc_id)

        # 6. Update job status and artifacts
        # ONLY storing the textUri in the artifacts, as requested.
        _update_job(doc_id, {
            "status": "SUCCEEDED",
            "phase": "OCR_DONE",
            "artifacts": {"textUri": text_uri},
        })
        
        # 7. Publish to next worker (Gemini)
        _publish_next({
            "type": "ocr.done",
            "docId": doc_id,
            "textUri": text_uri,
            "gcsUri": gcs_uri,  
            "routing": routing_info,  
            "ts": _now_iso()
        })

        app.logger.info(f"Batch OCR COMPLETE and published for {doc_id} -> {text_uri}")
        return ("OK", 200)

    except Exception as e:
        app.logger.error(f"DocAI result processing failed for {doc_id}: {e}")
        # Update job with failure status
        if doc_id:
            try:
                _update_job(doc_id, {
                    "status": "FAILED", 
                    "error": str(e), 
                    "phase": "DOCAI_RESULT_ERROR"
                })
            except Exception as update_error:
                app.logger.error(f"Failed to update job status after result error: {update_error}")
        # Return 500 to signal a transient error to Pub/Sub for retry
        return (f"ERROR: {e}", 500)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
