import requests
from bs4 import BeautifulSoup

HEADERS = {"User-Agent": "AsistenteETSI/1.0"}

def obtener_titulo_y_parrafos(url: str) -> tuple[str, list[str]]:
    # Hacer la petición HTTP a la URL
    respuesta = requests.get(url, headers=HEADERS, timeout=10)
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

    # Extraer textos de párrafos y listas
    parrafos = []
    for tag in soup.find_all(["p", "li"]):
        texto = tag.get_text(strip=True)
        if texto:
            parrafos.append(texto)

    # Devolver título y lista de párrafos
    return titulo, parrafos

# Test de la función
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