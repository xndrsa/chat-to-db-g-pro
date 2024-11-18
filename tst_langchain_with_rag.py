# import os
# import psycopg2
# from langchain_core.prompts import PromptTemplate
# from langchain_core.output_parsers import StrOutputParser
# from langchain.schema.runnable import RunnablePassthrough
# import google.generativeai as genai
# from langchain_community.utilities.sql_database import SQLDatabase

# from langchain.vectorstores import FAISS
# from langchain.embeddings.openai import OpenAIEmbeddings
# from langchain.docstore.document import Document
# import pandas as pd

# # 1. Cargar el archivo CSV
# csv_path = r"training_data\qa_pairs.csv"
# qa_df = pd.read_csv(csv_path)

# # Verificar que las columnas 'input' y 'output' existen
# if 'input' not in qa_df.columns or 'output' not in qa_df.columns:
#     raise ValueError("El archivo CSV debe contener las columnas 'input' y 'output'.")

# # Separar en dos conjuntos: SQL Queries y Preguntas Generales
# sql_queries_df = qa_df[qa_df['output'].str.contains("SELECT", na=False)]
# general_questions_df = qa_df[~qa_df['output'].str.contains("SELECT", na=False)]

# print(f"Total de SQL Queries: {len(sql_queries_df)}")
# print(f"Total de Preguntas Generales: {len(general_questions_df)}")

# # Crear documentos para SQL Queries y Preguntas Generales
# sql_documents = [
#     Document(page_content=f"Question: {row['input']}\nAnswer: {row['output']}")
#     for _, row in sql_queries_df.iterrows()
# ]
# general_documents = [
#     Document(page_content=f"Question: {row['input']}\nAnswer: {row['output']}")
#     for _, row in general_questions_df.iterrows()
# ]

# # Inicializar embeddings y crear índices FAISS
# embeddings = OpenAIEmbeddings()
# sql_index = FAISS.from_documents(sql_documents, embeddings)
# general_index = FAISS.from_documents(general_documents, embeddings)

# # Guardar índices localmente
# sql_index.save_local("faiss_sql_index")
# general_index.save_local("faiss_general_index")

# # Cargar los índices
# sql_index = FAISS.load_local("faiss_sql_index", embeddings)
# general_index = FAISS.load_local("faiss_general_index", embeddings)

# # Función para recuperar referencias de los índices
# def retrieve_reference(question, index, top_k=1):
#     """
#     Recupera respuestas relevantes de un índice basado en la pregunta del usuario.
#     """
#     docs = index.similarity_search(question, k=top_k)
#     return docs[0].page_content if docs else "No relevant answer found."

# # 2. Configuración de la base de datos PostgreSQL
# conn = psycopg2.connect(
#     host=os.getenv("HOST"),
#     user=os.getenv("USER"),
#     password=os.getenv("PASSWORD"),
#     database=os.getenv("DATABASE"),
#     port=5432
# )

# db_uri = f"postgresql+psycopg2://{os.getenv('USER')}:{os.getenv('PASSWORD')}@{os.getenv('HOST')}:5432/{os.getenv('DATABASE')}"
# db = SQLDatabase.from_uri(db_uri, view_support=True)

# def get_table_names(db, allowed_tables=None):
#     """
#     Extrae los nombres de las tablas utilizables de la base de datos, limitando a las tablas permitidas.
#     """
#     tables_info = {}
#     all_table_names = db.get_usable_table_names()
#     table_names = [table for table in all_table_names if table in allowed_tables] if allowed_tables else all_table_names
#     print("Tablas permitidas después del filtro:", table_names)
#     try:
#         for table in table_names:
#             table_info = db.get_table_info([table])
#             columns = [
#                 {"column_name": col.split(" ")[0], "data_type": " ".join(col.split(" ")[1:])}
#                 for col in table_info.split("\n")[1:] if col.strip()
#             ]
#             tables_info[table] = columns
#     except Exception as e:
#         print(f"Error al obtener la información de las tablas: {e}")
#     return tables_info

# allowed_tables = ["product", "pricing_transparency", "llm_fact_ms_drg_test"]
# table_names = get_table_names(db, allowed_tables=allowed_tables)

# # 3. Integrar índices en el flujo
# def process_question_with_references(question):
#     """
#     Procesa preguntas utilizando referencias de los índices.
#     """
#     # Consultar índices
#     sql_reference = retrieve_reference(question, sql_index)
#     general_reference = retrieve_reference(question, general_index)

#     # Mostrar referencia SQL si relevante
#     if "No relevant answer found" not in sql_reference:
#         print("\nReferencia SQL encontrada:")
#         print(sql_reference)
#         return sql_reference

#     # Mostrar referencia general si relevante
#     if "No relevant answer found" not in general_reference:
#         print("\nReferencia General encontrada:")
#         print(general_reference)
#         return general_reference

#     # Generar nueva respuesta si no hay referencias relevantes
#     return process_question(question)

# def process_question(question):
#     """
#     Procesa una pregunta y genera una respuesta SQL si es necesario.
#     """
#     relevant_tables = get_relevant_tables(question, list(table_names.keys()))
#     relevant_tables_info = {table: table_names[table] for table in relevant_tables if table in table_names}
#     sql_query = generate_sql_query(question, relevant_tables_info)
#     with conn.cursor() as cursor:
#         cursor.execute(sql_query)
#         result = cursor.fetchall()
#     return process_answer_direct(question, sql_query, result)

# # 4. Validar preguntas y mostrar resultados
# question_sql = "What is the average charge for MS-001 by payor and plan for great lakes?"
# response_sql = process_question_with_references(question_sql)
# print("\nRespuesta SQL:")
# print(response_sql)

# question_general = "What is a CDM (Chargemaster) and what is its role in a hospital?"
# response_general = process_question_with_references(question_general)
# print("\nRespuesta General:")
# print(response_general)




import os
import psycopg2
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
import google.generativeai as genai
from langchain_community.utilities.sql_database import SQLDatabase

from langchain.vectorstores import FAISS
from langchain.embeddings import FakeEmbeddings
from langchain.docstore.document import Document
import pandas as pd


from langchain.embeddings import HuggingFaceEmbeddings

# Utiliza un modelo de Hugging Face para generar embeddings
hf_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")



# 1. Configuración de Google Gemini
api_key = os.getenv("API_KEY")
genai.configure(api_key=api_key)

# Configurar la conexión a la base de datos PostgreSQL
conn = psycopg2.connect(
    host=os.getenv("HOST"),
    user=os.getenv("USER"),
    password=os.getenv("PASSWORD"),
    database=os.getenv("DATABASE"),
    port=5432
)

db_uri = f"postgresql+psycopg2://{os.getenv('USER')}:{os.getenv('PASSWORD')}@{os.getenv('HOST')}:5432/{os.getenv('DATABASE')}"
db = SQLDatabase.from_uri(db_uri, view_support=True)

# 2. Cargar el archivo CSV
csv_path = r"training_data\qa_pairs.csv"
qa_df = pd.read_csv(csv_path)

# Verificar que las columnas 'input' y 'output' existen
if 'input' not in qa_df.columns or 'output' not in qa_df.columns:
    raise ValueError("El archivo CSV debe contener las columnas 'input' y 'output'.")

# Separar en dos conjuntos: SQL Queries y Preguntas Generales
sql_queries_df = qa_df[qa_df['output'].str.contains("SELECT", na=False)]
general_questions_df = qa_df[~qa_df['output'].str.contains("SELECT", na=False)]

# Crear índices FAISS para SQL y preguntas generales
sql_documents = [
    Document(page_content=f"Question: {row['input']}\nAnswer: {row['output']}")
    for _, row in sql_queries_df.iterrows()
]
general_documents = [
    Document(page_content=f"Question: {row['input']}\nAnswer: {row['output']}")
    for _, row in general_questions_df.iterrows()
]

# Inicializar embeddings y crear índices FAISS

# Crear una clase para los embeddings si no existe soporte directo para Gemini
class GeminiEmbeddings:
    def __init__(self, embedding_dim=128):
        """
        Inicializa los embeddings con una longitud fija.
        """
        self.embedding_dim = embedding_dim

    def embed(self, text):
        """
        Genera embeddings de longitud fija para el texto.
        """
        # Simulación: Convierte el texto a un vector fijo
        embedding = [float(ord(char)) for char in text[:self.embedding_dim]]
        # Rellena con ceros si el texto es más corto que embedding_dim
        embedding += [0.0] * (self.embedding_dim - len(embedding))
        return embedding[:self.embedding_dim]  # Asegura la longitud fija

    def embed_documents(self, texts):
        """
        Genera embeddings para una lista de textos.
        """
        return [self.embed(text) for text in texts]


class FAISSCompatibleEmbeddings:
    def __init__(self, gemini_embeddings):
        self.gemini_embeddings = gemini_embeddings

    def embed_query(self, text):
        # Implementa cómo GeminiEmbeddings genera un embedding para un texto
        return self.gemini_embeddings.generate_embedding(text)

    def embed_documents(self, texts):
        # Implementa cómo GeminiEmbeddings genera embeddings para documentos
        return [self.gemini_embeddings.generate_embedding(text) for text in texts]









# Crear embeddings con Gemini
embeddings = FAISSCompatibleEmbeddings(GeminiEmbeddings())

# Crea los índices FAISS
sql_index = FAISS.from_documents(sql_documents, hf_embeddings)
general_index = FAISS.from_documents(general_documents, hf_embeddings)

# Guardar y cargar los índices
sql_index.save_local("faiss_sql_index")
general_index.save_local("faiss_general_index")

# Cargar los índices con deserialización permitida
sql_index = FAISS.load_local("faiss_sql_index", embeddings, allow_dangerous_deserialization=True)
general_index = FAISS.load_local("faiss_general_index", embeddings, allow_dangerous_deserialization=True)

# 3. Recuperar tablas de la base de datos
def get_table_names(db, allowed_tables=None):
    """
    Extrae los nombres de las tablas utilizables de la base de datos, limitando a las tablas permitidas.
    """
    tables_info = {}
    all_table_names = db.get_usable_table_names()
    table_names = [table for table in all_table_names if table in allowed_tables] if allowed_tables else all_table_names
    print("Tablas permitidas después del filtro:", table_names)
    try:
        for table in table_names:
            table_info = db.get_table_info([table])
            columns = [
                {"column_name": col.split(" ")[0], "data_type": " ".join(col.split(" ")[1:])}
                for col in table_info.split("\n")[1:] if col.strip()
            ]
            tables_info[table] = columns
    except Exception as e:
        print(f"Error al obtener la información de las tablas: {e}")
    return tables_info

allowed_tables = ["llm_fact_ms_drg_test"]
table_names = get_table_names(db, allowed_tables=allowed_tables)

# 4. Función para consultar índices
def retrieve_reference(question, index, top_k=1):
    """
    Recupera respuestas relevantes de un índice basado en la pregunta del usuario.
    """
    docs = index.similarity_search(question, k=top_k)
    return docs[0].page_content if docs else "No relevant answer found."

# 5. Funciones para LLM

def get_gemini_reply(question, prompt):
    """
    Función para generar una respuesta con Gemini.
    - question: pregunta del usuario.
    - prompt: contexto para el modelo.
    """
    model = genai.GenerativeModel('models/gemini-1.5-pro-latest')
    print("Usando modelo:", model)
    
    token_count_prompt = model.count_tokens(prompt)
    token_count_question = model.count_tokens(question)

    print("\nTokens utilizados en el LLM:")
    print(f"Prompt (tokens): {token_count_prompt}")
    print(f"Pregunta (tokens): {token_count_question}")

    response = model.generate_content(f"{prompt}\n\n{question}")

    token_count_response = model.count_tokens(response.text)
    print(f"Respuesta generada (tokens): {token_count_response}")

    return response.text

def get_relevant_tables(question, table_names):
    """
    Identifica las tablas relevantes basándose en la pregunta y el esquema.
    """
    prompt = f"""
    Return the names of ALL the SQL tables that MIGHT be relevant to the user question. 
    The tables are:

    {table_names}

    Remember to include ALL POTENTIALLY RELEVANT tables, even if you're not sure that they're needed.
    """
    response = get_gemini_reply(question="-", prompt=prompt)
    return response.split("\n")

def generate_sql_query(question, relevant_tables_info):
    """
    Genera una consulta SQL basada en una pregunta del usuario y la información de las tablas relevantes.
    """
    relevant_tables_context = "\n".join(
        f"Table: {table}\nColumns: {', '.join([f'{col['column_name']} ({col['data_type']})' for col in info])}"
        for table, info in relevant_tables_info.items()
    )
    prompt = f"""
    Based on the following relevant tables, create an SQL query to answer the question:
    {relevant_tables_context}
    Question: {question}
    """
    return get_gemini_reply(question="-", prompt=prompt)

def process_answer_direct(question, sql_query, result):
    """
    Procesa la respuesta final directamente con Gemini.
    """
    result_text = serialize_result(result)
    prompt = f"""
    Given the following question, corresponding Query and SQL result, answer the user question:
    Question: {question}
    Query: {sql_query}
    Result: {result_text}
    """
    return get_gemini_reply(question="-", prompt=prompt)

def serialize_result(result):
    """
    Convierte el resultado de la consulta SQL en texto plano.
    """
    if not result or len(result) == 0:
        return "No results found."
    try:
        return "\n".join([", ".join(map(str, row)) for row in result])
    except Exception as e:
        print(f"Error serializando el resultado: {e}")
        return "Serialization error."

# 6. Flujo principal con referencias
def process_question_with_references(question):
    """
    Procesa preguntas utilizando referencias de los índices.
    """
    sql_reference = retrieve_reference(question, sql_index)
    general_reference = retrieve_reference(question, general_index)
    if "No relevant answer found" not in sql_reference:
        print("\nReferencia SQL encontrada:")
        print(sql_reference)
        return sql_reference
    if "No relevant answer found" not in general_reference:
        print("\nReferencia General encontrada:")
        print(general_reference)
        return general_reference
    return process_question(question)

def process_question(question):
    """
    Procesa una pregunta y genera una respuesta SQL si es necesario.
    """
    relevant_tables = get_relevant_tables(question, list(table_names.keys()))
    relevant_tables_info = {table: table_names[table] for table in relevant_tables if table in table_names}
    sql_query = generate_sql_query(question, relevant_tables_info)
    with conn.cursor() as cursor:
        cursor.execute(sql_query)
        result = cursor.fetchall()
    return process_answer_direct(question, sql_query, result)

# 7. Prueba
question = "What is the average charge for MS-001 by payor and plan for great lakes?"
response_sql = process_question_with_references(question)
print("\nRespuesta SQL:")
print(response_sql)

question_general = "What is a CDM (Chargemaster) and what is its role in a hospital?"
response_general = process_question_with_references(question_general)
print("\nRespuesta General:")
print(response_general)
