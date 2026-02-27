from dotenv import load_dotenv

from src.ingestion.scraper import obtener_titulo_y_parrafos
from src.ingestion.processor import DocumentProcessor
from src.database.vector_store import VectorStoreManager
from src.core.llm_engine import ChatEngine


def main() -> None:
    # Cargar variables de entorno (incluida la API key de Groq)
    load_dotenv()

    # URL objetivo de la ETSI Informática de la UMA
    url = "https://www.uma.es/etsi-informatica/"

    # Ejecutar el scraping para obtener título y párrafos
    titulo, parrafos = obtener_titulo_y_parrafos(url)

    # Procesar los párrafos con el DocumentProcessor para obtener chunks
    processor = DocumentProcessor()
    chunks = processor.process(titulo, url, parrafos)

    if not chunks:
        print("No se ha generado ningún chunk tras el procesado.")
        return

    # Mostrar el primer chunk y sus metadatos
    first_chunk = chunks[0]
    print("\n--- Primer chunk ---")
    print(first_chunk.page_content)
    print("\n--- Metadatos del primer chunk ---")
    print(f"Título: {first_chunk.metadata.get('title')}")
    print(f"URL: {first_chunk.metadata.get('source')}")

    # Instanciar el gestor de la base de datos vectorial y añadir los chunks
    manager = VectorStoreManager()
    manager.add_documents(chunks, metadatas=None)

    # Prueba de búsqueda en el almacén vectorial
    query = "¿A qué día se cambiaron los examenes que había el 4 de febrero?"
    resultados = manager.search(query, k=3)

    print("\n=== Resultados de la búsqueda ===")
    print(f"Consulta: {query}\n")

    if not resultados:
        print("No se han encontrado resultados en la base de datos vectorial.")
        return

    for i, r in enumerate(resultados, start=1):
        print(f"\nResultado {i}:")
        print(f"Distancia: {r['distance']}")
        print(f"Metadatos: {r['metadata']}")
        print("Texto:")
        print(r['text'])

    # Generar respuesta final usando el ChatEngine y los fragmentos recuperados
    chat_engine = ChatEngine()
    respuesta = chat_engine.generate_answer(query, resultados)

    print("\n=== Respuesta del modelo ===")
    print(respuesta)


if __name__ == "__main__":
    main()

