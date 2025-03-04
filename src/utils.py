import pickle
import re

# UTILS.PY
def save_text_file(ruta, url, file_name, texto_extraido):
    with open(ruta + "/" + file_name, 'w', encoding='utf-8') as f:
        for item in texto_extraido:
            f.write(f"{item}\n\n")
    
    print("Extracción completada de "+url+" . Ver archivo './data/data_web_scraping/"+ file_name + "'")


def save_text_docs_langchain(ruta, url, file_name, texto_extraido):
    with open(ruta + "/" + file_name, 'w', encoding='utf-8') as f:
        for item in texto_extraido:
            f.write(f"{item}")
    
    print("Extracción completada de "+url+" . Ver archivo './data/data_langchain_webloader/"+ file_name + "'")

def save_docs_langchain(ruta,file_name,docs):
    # Guardar los documentos en un archivo .pkl
    with open(ruta + "/" + file_name, "wb") as f:
        pickle.dump(docs, f)

    print("Almacenamiento de objeto Documents. Ver archivo './data/dadata_langchain_webloader/"+ file_name + "'")

# func para limpiar texto extraido de las urls via langchain
def clean_text(text):
    text = re.sub(r'\s+', ' ', text)  # Eliminar espacios múltiples
    return text.strip()

