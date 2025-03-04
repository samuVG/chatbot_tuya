**Samuel Vasco González**

# Chatbot de Asistencia Financiera con RAG y Llama 3.2

Este proyecto implementa un chatbot de asistencia financiera utilizando técnicas de **Retrieval-Augmented Generation (RAG)** y el modelo de lenguaje **Llama 3.2**. El chatbot está diseñado para responder preguntas sobre los productos y servicios de **Tuya S.A.**, como tarjetas de crédito y Credicomrpas (créditos no rotativos para uso dentro del retail).

---

## Tabla de Contenidos
1. [Descripción del Proyecto](#descripción-del-proyecto)
2. [Tecnologías Utilizadas](#tecnologías-utilizadas)
3. [Instalación](#instalación)
4. [Uso](#uso)
5. [Estructura del Proyecto](#estructura-del-proyecto)
6. [Contribución](#contribución)
7. [Licencia](#licencia)

---

## Descripción del Proyecto

El objetivo de este proyecto es crear un chatbot que pueda responder preguntas específicas sobre los productos financieros de **Tuya S.A.**, utilizando información extraída de su sitio web. Para lograrlo, se implementa un sistema **RAG** que combina:

- **Recuperación de información**: Extrae documentos relevantes de una base de datos vectorial (FAISS).
- **Generación de respuestas**: Usa el modelo **Llama 3.2** para generar respuestas precisas basadas en el contexto recuperado.

El chatbot es capaz de responder preguntas como:
- ¿Qué tarjetas de crédito ofrece Tuya S.A.?
- ¿Cuál es la tasa de interés y valor de la póliza de seguro de Credicompras?

---

## Tecnologías Utilizadas

- **Lenguaje de Programación**: Python 3.11.3
- **Modelo de Lenguaje**: [Llama 3.2](https://huggingface.co/unsloth/Llama-3.2-1B-Instruct)
- **Frameworks**:
  - [LangChain](https://www.langchain.com/): Para la integración de RAG y gestión de cadenas de procesamiento.
  - [Hugging Face Transformers](https://huggingface.co/transformers/): Para cargar y usar el modelo de lenguaje Llama 3.2.
- **Almacenamiento Vectorial**: 
    - [FAISS](https://github.com/facebookresearch/faiss): Almacenamiento Vectorial para realizar búsquedas de similitud eficiente en vectores.
- **Scraping**: [BeautifulSoup](https://www.crummy.com/software/BeautifulSoup/) y [Requests](https://docs.python-requests.org/)
- **Embeddings**: [Sentence Transformers](https://www.sbert.net/)

---

## Instalación

Sigue estos pasos para configurar el proyecto en tu entorno local:

1. **Clona el repositorio**:
   ```bash
   git clone https://github.com/tu-usuario/tuya_chatbot.git
   cd tuya_chatbot```

2. **Crea un entorno virtual**:

    ```python -m venv venv
    source venv/bin/activate  # Linux/Mac
    .\venv\Scripts\activate   # Windows
    ```

    ```pip install -r requirements.txt```

3. **Ejecutar main.py**:

    main.py llama todas las funciones desde el web scraping hasta la ocnfiguración del modelo LLM con RAG. Ademas, imprime en pantalla la respuesta a las 2 preguntas de la prueba.

    Adicionalmente en el notebook **prueba_ejecucion_main.ipynb** se importa la funcion $main$ de ``main.py`` para probar el modelo y hacer preguntas adicionales al chatbot.
