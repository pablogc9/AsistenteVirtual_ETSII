from src.ingestion.pdf_parser import PDFProcessor

processor = PDFProcessor()
chunks = processor.process_folder("data/raw")

print(f"Número de chunks: {len(chunks)}\n\n")

first_chunk = chunks[0]
print("\n--- Primer chunk ---")
print(first_chunk.page_content)
print("\n--- Metadatos del primer chunk ---")
print(f"Título: {first_chunk.metadata.get('title')}")
print(f"URL: {first_chunk.metadata.get('source')}")
print(f"Número de página: {first_chunk.metadata.get('page_number')}")
