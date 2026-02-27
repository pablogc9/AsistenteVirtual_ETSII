import requests
from bs4 import BeautifulSoup

def obtener_titulo_y_parrafos(url: str) -> tuple[str, list[str]]:
    # Hacer la petición HTTP a la URL
    respuesta = requests.get(url)
    # Lanzar excepción si la respuesta tiene código de error (4xx, 5xx)
    respuesta.raise_for_status()

    # Asegurar una decodificación correcta de caracteres (tildes, ñ, etc.).
    respuesta.encoding = respuesta.apparent_encoding or respuesta.encoding or "utf-8"

    # Crear el objeto BeautifulSoup a partir del HTML recibido
    soup = BeautifulSoup(respuesta.text, "html.parser")

    # Extraer el título de la página (<title>)
    titulo = ""
    if soup.title and soup.title.string:
        titulo = soup.title.string.strip()

    # Extraer todos los textos dentro de etiquetas <p>
    parrafos = []
    for p in soup.find_all("p"):
        texto = p.get_text(strip=True)
        if texto:  # Evitar añadir párrafos vacíos
            parrafos.append(texto)

    # Devolver título y lista de párrafos
    return titulo, parrafos

if __name__ == "__main__":
    url_objetivo = "https://www.uma.es/etsi-informatica/"
    
    try:
        print(f"--- Iniciando extracción en: {url_objetivo} ---")
        titulo, contenidos = obtener_titulo_y_parrafos(url_objetivo)
        
        print(f"\n[TÍTULO DE LA PÁGINA]: {titulo}")
        print(f"\n[PÁRRAFOS ENCONTRADOS]: {len(contenidos)}")
        print("-" * 30)
        
        for i, p in enumerate(contenidos[:5]): # Mostramos solo los 5 primeros para no saturar
            print(f"Párrafo {i+1}: {p}")
            
    except Exception as e:
        print(f"Error durante el scraping: {e}")