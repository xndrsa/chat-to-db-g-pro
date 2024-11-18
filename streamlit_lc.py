import os
import psycopg2
import streamlit as st
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
import google.generativeai as genai
from langchain_community.utilities.sql_database import SQLDatabase
import pandas as pd

# Configurar el LLM de Gemini usando `genai`
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

def get_table_names(db, allowed_tables=None):
    tables_info = {}
    all_table_names = db.get_usable_table_names()  # Obtiene todas las tablas utilizables
    
    if allowed_tables:
        table_names = [table for table in all_table_names if table in allowed_tables]
    else:
        table_names = all_table_names

    print("Tablas permitidas después del filtro:", table_names)

    try:
        all_tables_info = db.get_table_info(table_names)
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

total_tokens = 0
total_llm_calls = 0

def get_gemini_reply(question, prompt):
    global total_tokens, total_llm_calls

    model = genai.GenerativeModel('gemini-1.5-flash')
    token_count_prompt = model.count_tokens(prompt).total_tokens
    token_count_question = model.count_tokens(question).total_tokens
    response = model.generate_content(f"{prompt}\n\n{question}")
    token_count_response = model.count_tokens(response.text).total_tokens

    total_tokens += token_count_prompt + token_count_question + token_count_response
    total_llm_calls += 1

    return response.text

def get_relevant_tables(question, table_names):
    prompt = f"""
    Return the names of ALL the SQL tables that MIGHT be relevant to the user question. 
    The tables are:

    {table_names}

    Remember to include ALL POTENTIALLY RELEVANT tables, even if you're not sure that they're needed.

    Important: Only return the name of the table, otherwise return None
    User question: {question}
    """
    response = ['llm_fact_ms_drg_test', '']
    response = [item for item in response if item.strip()]
    return response

def generate_sql_query(question, relevant_tables_info):
    relevant_tables_context = "\n".join(
        f"Table: {table}\nColumns: {', '.join([f'{col['column_name']} ({col['data_type']})' for col in info])}"
        for table, info in relevant_tables_info.items()
    )

    prompt = f"""
    You are an expert SQL query generator, focusing on pricing transparency, procedure codes, and hospital-related data. 

    Based on the following relevant tables, generate an SQL query to answer the provided question. Follow these rules:
    1. Return only the SQL query without additional text or delimiters.
    2. Use only the provided tables and columns.
    3. Use `LIKE` with `%` and `lower()` for flexible pattern matching.
    4. Avoid rows with null values.
    5. Ensure proper `GROUP BY` usage and include aggregated columns in `SELECT`.
    6. Use `standard_charge_negotiated_dollar` for pricing-related queries.
    7. If unclear or no data is available, return: 
       "Unable to generate query: Question is unclear or no relevant data available."

    Relevant Tables and Columns:
    {relevant_tables_context}

    User Question:
    {question}

    SQL Query:
    """
    sql_query = get_gemini_reply(question, prompt)
    return sql_query.strip()

def serialize_result(result):
    """
    Convierte el resultado de la consulta SQL en un texto plano legible.
    Filtra valores inválidos como None y los formatea correctamente.
    """
    if isinstance(result, pd.DataFrame):
        if result.empty:
            return "No results found."
        # Convierte el DataFrame en una cadena legible
        return result.to_csv(index=False, sep=",")
    elif not result or len(result) == 0:
        return "No results found."
    
    try:
        # Convierte otros tipos de resultados en un formato legible
        formatted_result = "\n".join(
            [", ".join(map(str, row)) for row in result]
        )
        return formatted_result
    except Exception as e:
        print(f"Error serializando el resultado: {e}")
        return "Serialization error."


def process_question(question):
    """
    Procesa una pregunta, identifica tablas relevantes, genera una consulta SQL y ejecuta la consulta.
    """
    print("\n[1] Identificando tablas relevantes...")
    relevant_tables = get_relevant_tables(question, list(table_names.keys()))
    print("Tablas relevantes:", relevant_tables)
    relevant_tables_info = {table: table_names[table] for table in relevant_tables if table in table_names}

    print("\n[2] Generando consulta SQL...")
    sql_query = generate_sql_query(question, relevant_tables_info)
    print("Consulta SQL generada:", sql_query)

    # Validar si el LLM devolvió un mensaje en lugar de una consulta SQL
    if "Unable to generate query" in sql_query:
        return None, sql_query  # Retorna None como DataFrame y el mensaje como consulta

    print("\n[3] Ejecutando consulta en la base de datos...")
    try:
        with conn.cursor() as cursor:
            cursor.execute(sql_query)
            col_names = [desc[0] for desc in cursor.description]
            result = cursor.fetchall()
            df = pd.DataFrame(result, columns=col_names)
    except Exception as e:
        print(f"Error ejecutando la consulta: {e}")
        return None, "Error executing the query: " + str(e)

    return df, sql_query



# def process_question(question):
#     """
#     Procesa la pregunta, genera el SQL y ejecuta la consulta.
#     """
#     relevant_tables = get_relevant_tables(question, list(table_names.keys()))
#     relevant_tables_info = {table: table_names[table] for table in relevant_tables if table in table_names}
#     sql_query = generate_sql_query(question, relevant_tables_info)
#     print(sql_query)
#     try:
#         with conn.cursor() as cursor:
#             cursor.execute(sql_query)
#             col_names = [desc[0] for desc in cursor.description]
#             result = cursor.fetchall()
#             if not result:
#                 return pd.DataFrame(), sql_query  # Retorna un DataFrame vacío si no hay resultados
#             df = pd.DataFrame(result, columns=col_names)
#             return df, sql_query
#     except Exception as e:
#         error_message = f"Unable to execute the query due to: {str(e)}"
#         print(error_message)
#         return sql_query  # Retorna el mensaje de error como string y el query

def analyze_question_relevance(question):
    """
    Evalúa si la pregunta está relacionada con pricing transparency usando el modelo LLM.
    """
    prompt = f"""
    You are a specialized assistant focusing on pricing transparency, hospital billing, 
    medical procedure codes, and healthcare-related topics. Your goal is to determine if 
    the user's question is relevant to your expertise.

    User Question: "{question}"

    ### Instructions:
    - If the question is related to pricing transparency, healthcare billing, or hospital 
      data, respond: "Relevant: The question pertains to the main topic."
    - If the question is unrelated, respond with: "Irrelevant: This question is outside 
      the main topic. Please ask about pricing transparency, hospital billing, or 
      healthcare-related topics."
    - Your answer must only include "Relevant" or "Irrelevant" followed by the explanation.
    
    Answer:
    """
    try:
        relevance_response = get_gemini_reply(question, prompt)
        return relevance_response.strip()
    except Exception as e:
        print(f"Error al analizar relevancia: {e}")
        return "Error analyzing the question relevance."



def process_answer_direct(question, sql_query, result):
    result_text = serialize_result(result)
    prompt = f"""
    You are a professional analyst specializing in pricing transparency and hospital-related topics.

    Given the following:
    - **Question**: {question}
    - **SQL Query**: {sql_query}
    - **Result**: {result_text}

    Provide a clear, concise, and accurate answer.

    Answer:
    """
    try:
        response = get_gemini_reply(question="-", prompt=prompt) 
        return response
    except Exception as e:
        print(f"Error al invocar el modelo directamente: {e}")
        raise

### Integración Streamlit ###

logo = r"Correlate-Logo-Final.png"
st.set_page_config(page_title="CorrelateIQ", page_icon=logo, layout="centered")
st.header("Chat with CorrelateIQ")
st.image(logo, width=400)

with st.form(key='question_form'):
    col1, col2 = st.columns([4, 1])
    with col1:
        question = st.text_area(
            "Type your question here:",
            placeholder="Ask a question related to pricing transparency...",
            key="question_input",
            height=100
        )
    with col2:
        submit = st.form_submit_button("Submit")

# if submit:
#     st.subheader("Processing your query...")
#     df, sql_query = process_question(question)

#     # Analizar relevancia de la pregunta
#     relevance_response = analyze_question_relevance(question)
#     if relevance_response.startswith("Irrelevant"):
#         st.warning(relevance_response)
#     else:
#         # Procesar la pregunta si es relevante
#         df, sql_query = process_question(question)

#         if isinstance(df, str):  # Si se devolvió un mensaje de error
#             st.warning("The query could not be executed. Please refine your question.")
#         elif isinstance(df, pd.DataFrame) and df.empty:  # Si el DataFrame está vacío
#             st.warning("The query executed successfully, but no results were returned.")
#         else:  # Si hay datos en el DataFrame
#             final_answer = process_answer_direct(question, sql_query, df)
#             st.subheader("Answer:")
#             st.write(final_answer)
#             st.subheader("Query Results:")
#             st.dataframe(df)


if submit:
    st.subheader("Processing your query...")

    df, sql_query = process_question(question)

    if df is None:
        # Manejar casos donde no se generó o ejecutó una consulta
        if "Unable to generate query" in sql_query:
            st.warning(sql_query)  # Mostrar el mensaje del LLM
        else:
            st.error(sql_query)  # Mostrar el mensaje de error generado por `process_question`
    elif df.empty:
        # Si la consulta fue exitosa pero no devolvió resultados
        st.warning("The query executed successfully, but no results were returned.")
    else:
        # Si hay resultados, procesa la respuesta final
        final_answer = process_answer_direct(question, sql_query, df)
        st.subheader("Answer:")
        st.write(final_answer)
        st.subheader("Query Results:")
        st.dataframe(df)



st.sidebar.title("Usage Stats")
st.sidebar.markdown(f"**Total Tokens Used:** {total_tokens}")
st.sidebar.markdown(f"**Total LLM Calls:** {total_llm_calls}")
st.sidebar.markdown(f"**Query Generated:** {sql_query}")
