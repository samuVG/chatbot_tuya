from bs4 import BeautifulSoup
import requests
from langchain.document_loaders import WebBaseLoader
from langchain.docstore.document import Document
import utils

# funciones para hacer el web scraping de las urls

def extract_ordered_text_1(url):

    try:
        # Descargar el HTML directamente de la URL
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        response = requests.get(url, headers=headers)
        response.raise_for_status()  # Verificar errores HTTP
        
        html = response.text
        # Resto del código de procesamiento...
        
    except requests.exceptions.RequestException as e:
        print(f"Error al descargar la página: {e}")
        return []

    # Parsear el HTML
    soup = BeautifulSoup(html, 'html.parser')
    
    # Eliminar elementos <a> no deseados
    for a_tag in soup.find_all('a', attrs={'data-toggle': lambda x: x != 'collapse'}):
        a_tag.decompose()

    # Elementos a eliminar por clase predefinida
    excluded_classes = {
        'quizaEstesBuscando',  # Clase sin info relevante
        'textMenurightXs'      # Clase sin info relevante
    }
    for element in soup.find_all(class_=excluded_classes):
        element.decompose()  # Elimina el elemento y su contenido del árbol

    # Eliminar elementos no deseados de html (scripts, estilos, imagenes, botones, etc.)
    for element in soup(['script', 'style', 'noscript', 'meta', 'link', 'header', 'footer', 'img', 'button', 'nav', 'span']):
        element.decompose()
    
    # Seleccionar los elementos relevantes en orden de aparición
    relevant_tags = ['div', 'h1', 'h2', 'h3', 'p', 'li']
    elements = soup.find_all(relevant_tags)
    
    # Extraer y limpiar el texto manteniendo el orden
    ordered_text = []
    #evitar contenido duplicado al rastrear los elementos HTML que ya fueron procesados
    visited_elements = set() # colección que no permite elementos duplicados
    
    for element in elements:
        # para no repetir elementos. Si el elemento ya se añadio se pasa al siguiente
        if element in visited_elements:
            continue

        # Saltar elementos vacíos o que contienen solo whitespace
        text = element.get_text(strip=True, separator=' ')
        if not text:
            continue
        
        # Evitar contenido duplicado de elementos padres/hijos
        parents = list(element.parents)
        if any(parent in visited_elements for parent in parents):
            continue
            
        ordered_text.append(text)
        visited_elements.add(element)
    
    return ordered_text


def extract_ordered_text_2(url, elements_excluded):

    try:
        # Descargar el HTML directamente de la URL
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        response.encoding = "utf-8"  # Fuerza codificación
        
        # Limpiar y guardar
        soup = BeautifulSoup(response.text, "html.parser")
        cleaned_html = soup.prettify()
     
        
    except requests.exceptions.RequestException as e:
        print(f"Error al descargar la página: {e}")
        return []

    # Parsear el HTML
    soup = BeautifulSoup(cleaned_html, 'html.parser')
    
    # # Eliminar elementos <a> no deseados
    # for a_tag in soup.find_all('a', attrs={'data-toggle': lambda x: x != 'collapse'}):
    #     a_tag.decompose()

    # Filtrar elementos <a>: conservar data-toggle="collapse" O enlaces con texto y href (link)
    for a_tag in soup.find_all('a'):
        has_data_toggle = a_tag.get('data-toggle') == 'collapse'
        has_valid_href = a_tag.has_attr('href') and a_tag.get_text(strip=True)
        
        if not (has_data_toggle or has_valid_href):
            a_tag.decompose()

    # Elementos a eliminar por clase predefinida
    excluded_classes = {
        'quizaEstesBuscando',  # Clase sin info relevante
        'textMenurightXs'      # Clase sin info relevante
    }
    for element in soup.find_all(class_=excluded_classes):
        element.decompose()  # Elimina el elemento y su contenido del árbol

    # Eliminar elementos no deseados de html (scripts, estilos, imagenes, botones, etc.)
    for element in soup(elements_excluded):
        element.decompose()
    

    # Seleccionar los elementos relevantes en orden de aparición
    relevant_tags = ['div', 'h1', 'h2', 'h3', 'p', 'li']
    elements = soup.find_all(relevant_tags)
    
    # Extraer y limpiar el texto manteniendo el orden
    ordered_text = []
    #evitar contenido duplicado al rastrear los elementos HTML que ya fueron procesados
    visited_elements = set() # colección que no permite elementos duplicados
    
    for element in elements:
        # para no repetir elementos. Si el elemento ya se añadio se pasa al siguiente
        if element in visited_elements:
            continue

        # Saltar elementos vacíos o que contienen solo whitespace. Extraer texto con formato personalizado
        # Manejar enlaces especiales
        if element.name == 'a':
            link_text = element.get_text(strip=True)
            url_href = element.get('href', '')
            if url_href:
                text = f"{link_text} [URL: {url_href}]" if link_text else url_href
            else:
                text = link_text
        else:
            text = element.get_text(strip=True, separator=' ')
            
        if not text:
            continue
        
        # Evitar contenido duplicado de elementos padres/hijos
        parents = list(element.parents)
        if any(parent in visited_elements for parent in parents):
            continue
            
        ordered_text.append(text)
        visited_elements.add(element)
    
    return ordered_text

# Preprocesar los documentos
def preprocess_documents(docs):
    cleaned_docs = []
    for doc in docs:
        cleaned_content = utils.clean_text(doc.page_content)
        cleaned_docs.append(Document(page_content=cleaned_content, metadata=doc.metadata))
    return cleaned_docs

# web scraping de las urls usando langchain
def extract_text_langchain(urls):

    loader = WebBaseLoader(urls)
    docs = loader.load()

    # 4. Aplicar preprocesamiento
    cleaned_docs = preprocess_documents(docs)

    return cleaned_docs