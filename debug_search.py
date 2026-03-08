from src.database.vector_store import VectorStoreManager

vs = VectorStoreManager()
count = vs._collection.count()
print(f"Total chunks en ChromaDB: {count}\n")

query = "cuantos creditos totales tiene el grado en ingenieria del software"
results = vs._collection.query(
    query_texts=[query],
    n_results=10,
    include=["documents", "metadatas", "distances"],
)

print("=== RESULTADOS BUSQUEDA ===")
for i, (doc, meta, dist) in enumerate(
    zip(results["documents"][0], results["metadatas"][0], results["distances"][0])
):
    fuente = meta.get("filename", meta.get("url", "-"))
    print(f"\n[{i+1}] dist={dist:.4f} | fuente={fuente}")
    print(doc[:400])
    print("---")

# Buscar especificamente en datos_maestros
print("\n=== BUSQUEDA ESPECIFICA datos_maestros ===")
r2 = vs._collection.query(
    query_texts=["240 creditos ECTS grado software"],
    n_results=5,
    include=["documents", "metadatas", "distances"],
)
for i, (doc, meta, dist) in enumerate(
    zip(r2["documents"][0], r2["metadatas"][0], r2["distances"][0])
):
    fuente = meta.get("filename", meta.get("url", "-"))
    print(f"\n[{i+1}] dist={dist:.4f} | fuente={fuente}")
    print(doc[:400])
