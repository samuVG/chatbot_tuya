#librerias con las funciones a utilizar
import utils
import scraper
import llm_rag_pipeline

def main():
    print("Comienza la ejecucion del chat bot Tuya:")

    # 1. Web Scraping. Proceso de extraccion de texto de las paginas web y almacenamiento en la carpeta data.

    print("     1. Web Scraping...")
    urls = [
        "https://www.tuya.com.co/tarjetas-de-credito",
        "https://www.tuya.com.co/credicompras",
        "https://www.tuya.com.co/como-pago-mi-tarjeta-o-credicompras",
        "https://www.tuya.com.co/otras-soluciones-financieras",
        "https://www.tuya.com.co/nuestra-compania"
    ]

    docs = scraper.extract_text_langchain(urls)

    ruta =r'S:\SAMUEL VASCO\TUYA\Cientifico_de_Datos\Prueba_cientifico_IA\data\data_langchain_webloader'
    file_name = "docs_from_urls_langchain.pkl"
    # Guardar el objeto Documents con el texto extraido de las urls. Producido por WebBaseLoader de Langchain
    utils.save_docs_langchain(ruta,file_name,docs)

    # URL 1
    url = "https://www.tuya.com.co/tarjetas-de-credito"
    file_name = "texto_extraido_url_1_tarjetas_de_credito.txt"
    texto_extraido = str(docs[0])
    utils.save_text_docs_langchain(ruta, url, file_name, texto_extraido)

    # URL 2
    url = "https://www.tuya.com.co/credicompras"
    file_name = "texto_extraido_url_2_credicompras.txt"
    texto_extraido = str(docs[1])
    utils.save_text_docs_langchain(ruta, url, file_name, texto_extraido)

    # URL 3
    url = "https://www.tuya.com.co/como-pago-mi-tarjeta-o-credicompras"
    file_name = "texto_extraido_url_3_como_pago_mi_tarjeta_o_credicompras.txt"
    texto_extraido = str(docs[2])
    utils.save_text_docs_langchain(ruta, url, file_name, texto_extraido)

    # URL 4
    url = "https://www.tuya.com.co/otras-soluciones-financieras"
    file_name = "texto_extraido_url_4_otras_soluciones_financieras.txt"
    texto_extraido = str(docs[3])
    utils.save_text_docs_langchain(ruta, url, file_name, texto_extraido)

    # URL 6
    url = "https://www.tuya.com.co/nuestra-compania"
    file_name = "texto_extraido_url_6_nuestra_compania.txt"
    texto_extraido = str(docs[4])
    utils.save_text_docs_langchain(ruta, url, file_name, texto_extraido)

    print("         Listo.")

    # 2. Dividir Texto en Chunks por oraciones completas
    print("     2. Dividir Texto en Chunks...")
    splits = llm_rag_pipeline.split_documents_langchain(docs)
    print("         Listo.")

    # 3. Generar Embeddings
    print("     3. Generar Embeddings...")
    embeddings = llm_rag_pipeline.embedding_HuggingFace()
    print("         Listo.")

    # 4. Almacenamiento Vectorial (FAISS)
    print("     4. Almacenamiento Vectorial (FAISS)...")
    vectorstore = llm_rag_pipeline.FAISS_langchain(splits, embeddings)
    print("         Listo.")

    # 5. Configuración de Llama 3.2 (usando cpu) y Pipeline
    print("     5. Configuración de Llama 3.2...")
    llm = llm_rag_pipeline.pipeline_config_llm(model_id="unsloth/Llama-3.2-1B-Instruct")
    print("         Listo.")

    # 6. Configurar el Sistema RAG
    print("     6. Configurar el Sistema RAG...")
    qa_chain = llm_rag_pipeline.config_RAG_system(llm, vectorstore)
    print("         Listo.")

    return qa_chain

if __name__ == "__main__":
# 7. Prueba del modelo con 2 preguntas
    qa_chain = main()

    print('\n\n\n',"Pregunta 1:")
    question1 = "¿Cuáles son los nombres de las tarjetas que tiene disponibles Tuya S.A.?"
    print(question1,'\n')
    response1 = qa_chain.invoke({"query": question1})
    print(response1["result"],'\n\n')

    print("Pregunta 2:")
    question2 = "¿Cuáles son los valores la tasa de interés y póliza del producto Credicompras?."
    print(question2,'\n')
    response2 = qa_chain.invoke({"query": question2})
    print(response2["result"])