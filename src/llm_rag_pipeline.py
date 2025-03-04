# librerias
from langchain.text_splitter import NLTKTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline
)
from langchain.llms import HuggingFacePipeline
import torch
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# funcion para Text spliter por oraciones completas
def split_documents_langchain(docs):
    # Crear el splitter
    nltk_splitter = NLTKTextSplitter()

    # Dividir el texto
    splits = nltk_splitter.split_documents(docs)

    return splits

# funcion que retorna objeto con embedding en español
def embedding_HuggingFace():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},  # 'cuda' para GPU
        encode_kwargs={'normalize_embeddings': True}
    )
    return embeddings

# Almacenamiento Vectorial (FAISS). para realizar búsquedas de similitud eficiente en vectores
def FAISS_langchain(splits, embeddings):
    vectorstore = FAISS.from_documents(
        documents=splits, 
        embedding=embeddings
    )
    return vectorstore

# Configuración de llm Llama 3.2
def pipeline_config_llm(model_id):
    # ===== 1. Parámetros del Modelo =====
    #model_id = "unsloth/Llama-3.2-1B-Instruct"

    # Configurar dispositivo
    device = "cpu"  # Forzar CPU

    # ===== 2. Cargar Modelo y Tokenizer =====
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            padding_side="left",
            trust_remote_code=True
        )
        
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map=None,  # No usar device_map en CPU
            torch_dtype=torch.float32,  # Usar float32 en CPU
        )
    except Exception as e:
        print(f"Error al cargar el modelo: {str(e)}")
        exit()

    # ===== 3. Configuración del Pipeline =====
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,  # Reducir para CPU
        temperature=0.3, # controla la aleatoriedad de las respuestas. Un valor más bajo (cercano a 0) hace que el modelo sea más determinista
        repetition_penalty=1.15, #  Controla la penalización por repetición de palabras o frases. Valores cercanos a 1: No hay penalización por repetición. 
                     #Valores mayores a 1: El modelo evita repetir palabras o frases. Un valor de 1.15 significa que las palabras repetidas se penalizan ligeramente.
        do_sample=True,
        return_full_text=False,
        pad_token_id=tokenizer.eos_token_id,
        device=device  # Usar CPU
    )

    # ===== 4. Integración con LangChain =====
    llm = HuggingFacePipeline(
        pipeline=pipe,
        model_kwargs={
            "max_length": 1024,  # Reducir para CPU
            "truncation": True
        }
    )

    return llm

# 6. Configurar el Sistema RAG
# Ahora, integra el vectorstore con el modelo para que pueda recuperar documentos relevantes antes de generar una respuesta
def config_RAG_system(llm, vectorstore):
    # 1. Definir el prompt con contexto
    template = """
    <s>[INST] Eres un asistente especializado en productos financieros de Tuya S.A.
    Responde la pregunta usando solo la información proporcionada en el contexto.

    Contexto:
    {context}

    Pregunta: {question}
    Respuesta: [/INST]
    """
    prompt = PromptTemplate.from_template(template)

    # 2. Configurar la cadena de RAG
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,  # Tu modelo Llama 3.2
        chain_type="stuff",
        retriever=vectorstore.as_retriever(),  # Integrar el vectorstore # search_kwargs={"k": 3} para limitar fragmentos
        chain_type_kwargs={"prompt": prompt}
    )
    return qa_chain