Upload a new PDF to gs://fsmb-legal-docs/raw/

Eventarc detects the new object → sends event to Cloud Run

doc-orchestrator

Creates/updates Firestore jobs/{docId}

Publishes a message to Pub/Sub

Pub/Sub pushes the message to doc-worker

doc-worker

Marks the job RUNNING

Calls Document AI to OCR → saves txt/<docId>.txt

Fetches schema & prompt from Firestore

Calls Gemini to extract fields → saves processed/<docId>.json

Marks the job SUCCEEDED in Firestore (or FAILED on error)

Firestore & GCS now contain:

Job metadata (status, artifacts, used.promptId, used.schemaId)

Text & structured output for downstream indexing or analytics
