from google.cloud import documentai

docai_client = documentai.DocumentProcessorServiceClient()

def run_docai_ocr(project_id: str, location: str, processor_id: str, gcs_uri: str) -> str:
    name = docai_client.processor_path(project_id, location, processor_id)

    # download the PDF from GCS
    bucket_name, object_name = gcs_uri.replace("gs://", "").split("/", 1)
    pdf_bytes = storage_client.bucket(bucket_name).blob(object_name).download_as_bytes()

    raw_document = documentai.RawDocument(content=pdf_bytes, mime_type="application/pdf")
    request = documentai.ProcessRequest(name=name, raw_document=raw_document)
    result = docai_client.process_document(request=request).document

    return result.text or ""
