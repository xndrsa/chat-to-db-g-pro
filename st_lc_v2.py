import os
import psycopg2
import streamlit as st
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
    all_table_names = db.get_usable_table_names()
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

def analyze_question_relevance(question):
    """
    Evalúa si la pregunta está relacionada con pricing transparency.
    """
    prompt = f"""
    You are a professional assistant focusing on pricing transparency, hospital billing,
    medical procedure codes, and healthcare-related topics.

    User Question: "{question}"

    ### Instructions:
    - If the question is related to your domain, respond:
      "Relevant: This question pertains to the main topic."
    - If the question is unrelated, respond:
      "Irrelevant: This question is outside the main topic. Please ask about pricing transparency or hospital billing."

    Answer:
    """
    try:
        relevance_response = get_gemini_reply(question, prompt)
        return relevance_response.strip()
    except Exception as e:
        print(f"Error al analizar relevancia: {e}")
        return "Error analyzing question relevance."

def get_relevant_tables(question, table_names):

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


def process_question(question):
    """
    Procesa la pregunta, identifica tablas relevantes, genera y ejecuta la consulta SQL.
    """
    relevant_tables = get_relevant_tables(question, list(table_names.keys()))
    relevant_tables_info = {table: table_names[table] for table in relevant_tables if table in table_names}
    sql_query = generate_sql_query(question, relevant_tables_info)

    if "Unable to generate query" in sql_query:
        return None, sql_query

    try:
        with conn.cursor() as cursor:
            cursor.execute(sql_query)
            col_names = [desc[0] for desc in cursor.description]
            result = cursor.fetchall()
            df = pd.DataFrame(result, columns=col_names)
    except Exception as e:
        return None, f"Error executing query: {str(e)}"

    return df, sql_query

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

def process_answer_direct(question, sql_query, result):
    """
    Procesa la respuesta usando el modelo LLM.
    """
    result_text = serialize_result(result)
    prompt = f"""
    You are a professional analyst specializing in pricing transparency, medical procedure codes,
    and hospital-related topics.

    Given:
    - **Question**: {question}
    - **SQL Query**: {sql_query}
    - **Result**: {result_text}

    ### Instructions:
    1. Provide a clear and concise answer if relevant to pricing transparency.
    2. Redirect irrelevant questions politely to the main topic.
    3. Maintain a professional tone and avoid assumptions.

    Answer:
    """
    try:
        response = get_gemini_reply(question="-", prompt=prompt)
        return response
    except Exception as e:
        print(f"Error al invocar el modelo: {e}")
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

if submit:
    st.subheader("Processing your query...")

    # Análisis de relevancia
    relevance_response = analyze_question_relevance(question)
    if relevance_response.startswith("Irrelevant"):
        st.warning(relevance_response)
    else:
        # Procesar la pregunta si es relevante
        df, sql_query = process_question(question)

        if df is None:
            st.warning(sql_query)
        elif df.empty:
            st.warning("The query executed successfully, but no results were returned.")
        else:
            final_answer = process_answer_direct(question, sql_query, df)
            st.subheader("Answer:")
            st.write(final_answer)
            st.subheader("Query Results:")
            st.dataframe(df)

st.sidebar.title("Usage Stats")
st.sidebar.markdown(f"**Total Tokens Used:** {total_tokens}")
st.sidebar.markdown(f"**Total LLM Calls:** {total_llm_calls}")
st.sidebar.markdown(f"**Query Generated:** {sql_query}")
