import os
import psycopg2
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
#from langchain_core.runnables import RunnablePassthrough
from langchain.schema.runnable import RunnablePassthrough
import google.generativeai as genai
from langchain_community.utilities.sql_database import SQLDatabase
import pandas as pd
import streamlit as st

api_key = os.getenv("API_KEY")
genai.configure(api_key=api_key)


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
    """
    Extrae los nombres de las tablas utilizables de la base de datos, limitando a las tablas permitidas.
    """
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
            print(f"Procesando información de la tabla: {table}")
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
table_names = get_table_names(db,allowed_tables=allowed_tables)


total_tokens = 0
total_llm_calls = 0

def get_gemini_reply(question, prompt):
    """
    Función para generar una respuesta con Gemini.
    - question: pregunta del usuario.
    - prompt: contexto para el modelo.
    """

    global total_tokens, total_llm_calls

    model = genai.GenerativeModel('gemini-2.0-flash-exp')  #'gemini-1.5-flash')#models/gemini-1.5-pro-latest')
    
    token_count_prompt = model.count_tokens(prompt).total_tokens
    token_count_question = model.count_tokens(question).total_tokens

    response = model.generate_content(f"{prompt}\n\n{question}")

    token_count_response = model.count_tokens(response.text).total_tokens

    total_tokens += token_count_prompt + token_count_question + token_count_response
    total_llm_calls += 1

    return response.text


def get_gemini_reply_sql(question, prompt):
    """
    Función para generar una respuesta con Gemini.
    - question: pregunta del usuario.
    - prompt: contexto para el modelo.
    """

    global total_tokens, total_llm_calls

    model = genai.GenerativeModel('tunedModels/queryinvalidquery-2duecpnc0gof')  #'tunedModels/sqleg-8nrr5fqw76yj')#'gemini-1.5-flash')#models/gemini-1.5-pro-latest')
    
    token_count_prompt = model.count_tokens(prompt).total_tokens
    token_count_question = model.count_tokens(question).total_tokens

    response = model.generate_content(f"{prompt}\n\n{question}")

    token_count_response = model.count_tokens(response.text).total_tokens

    total_tokens += token_count_prompt + token_count_question + token_count_response
    total_llm_calls += 1

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

    Important: Only return the name of the table, otherwise return None
    User question: {question}
    """
    #response = get_gemini_reply(question, prompt) # TODO uncommnet when needed
    #return response.split("\n")  # Devuelve una lista de tablas relevantes#TODO uncomment when needed
    
    response=['llm_fact_ms_drg_test', '']#TODO 
    response = [item for item in response if item.strip()]
    return response


def validate_single_query(sql_query):
    """
    Verifica si el modelo generó más de una consulta SQL.
    """
    queries = sql_query.split(";")

    queries = [query.strip() for query in queries if query.strip()]
    return queries[0] if queries else "Error: No valid query generated."



def generate_sql_query(question, relevant_tables_info):
    relevant_tables_context = "\n".join(
        f"Table: {table}\nColumns: {', '.join([f'{col['column_name']} ({col['data_type']})' for col in info])}"
        for table, info in relevant_tables_info.items()
    )

    prompt = f"""
    You are an expert POSTGRESQL query generator specializing in pricing transparency, procedure codes, and hospital-related data.

    ### Task:
    Generate **exactly one SQL query** based on the provided question and the relevant tables below. The query must be accurate, clear, and directly address the user's question.

    ### Rules:
    1. **Output Formatting**:
       - Return only one SQL query as output. Do not include any additional text, comments, or delimiters (e.g., no `'`, `"`, or `sql`).
    2. **Focus**:
       - Ensure the query strictly addresses the user's question using only the provided tables and columns.
    3. **Priority**:
       - If there are multiple ways to answer the question, choose the most relevant and concise query.
    4. **Restrictions**:
       - Use only the tables and columns listed in the relevant tables section.
       - Avoid including rows with null values from any table. e.g. `WHERE column1 NOTNULL`
    5. **Pattern Matching**:
       - Use `LIKE` with `%` and `lower()` for pattern matching (e.g., `WHERE lower(column_name) LIKE '%value%'`).
    6. **Error Handling**:
       - If the question is unclear or insufficient data is available, return this message:
         `"Unable to generate query: Question is unclear or no relevant data available."`
    7. **Pricing Column**:
       - The column `standard_charge_negotiated_dollar` represents procedure prices/charges and is the primary column for pricing-related queries.


    ### Relevant Tables:
    {relevant_tables_context}

    ### User Question:
    {question}

    ### SQL Query:
    """
    
    sql_query = get_gemini_reply_sql(question, prompt)
    return sql_query.strip()



def serialize_result(result, row_limit=10):
    """
    Serializa los resultados de una consulta en un formato legible, con un límite opcional de filas.
    
    Parameters:
        result (list of tuples): El resultado de la consulta SQL.
        row_limit (int): Número máximo de filas a incluir en el resultado serializado.
    
    Returns:
        str: Resultado formateado como texto plano o un mensaje de error.
    """
    if not result or len(result) == 0:
        return "No results found."
    
    try:
        limited_result = result[:row_limit]

        formatted_result = "\n".join(
            [", ".join(map(str, row)) for row in limited_result]
        )
        if len(result) > row_limit:
            formatted_result += f"\n... (Showing first {row_limit} rows out of {len(result)})"
        
        return formatted_result
    except Exception as e:
        print(f"Error serializando el resultado: {e}")
        return "Serialization error."



def process_question(question):
    print("\n[1] tablas relevantes...")
    relevant_tables = get_relevant_tables(question, list(table_names.keys()))
    print(relevant_tables)
    relevant_tables_info = {table: table_names[table] for table in relevant_tables if table in table_names}
        
    print("\n[2] Generando consulta SQL...")
    sql_query = generate_sql_query(question, relevant_tables_info)

    sql_query_validate = validate_single_query(sql_query)

    if "Error" in sql_query_validate:
        print("Invalid SQL")
        print(sql_query)

        print("\n[3] Unable to execute query...")
        return None, "Invalid query","No results"
    
    sql_query = sql_query_validate
    print("Consulta SQL válida:", sql_query)

    print("\n[3] Ejecutando consulta en la base de datos...")
    try:
        with conn.cursor() as cursor:
            cursor.execute(sql_query)
            col_names = [desc[0] for desc in cursor.description]
            result = cursor.fetchall()
            if not result:
                print("Consulta ejecutada pero no devolvió resultados.")
                return pd.DataFrame(), sql_query ,"No results"
            df = pd.DataFrame(result, columns=col_names)
            print("Resultados de la consulta:", df)
        return df, sql_query, result
    except Exception as e:
        print(f"Error al ejecutar la consulta: {e}")

        return None, "Invalid query","No results"

answer_template = """ Given the following question, corresponding Query and SQL result, answer the user question: Question: {question} Query: {sql_query} Result: {result} Answer:"""
answer_prompt = PromptTemplate.from_template(answer_template)


output_parser = StrOutputParser()

def build_chain(question, sql_query, result):
    """
    Construye la cadena de procesamiento.
    """
    chain = (
        RunnablePassthrough()
        | answer_prompt        
        | output_parser       
    )
    return chain


def process_answer(question, sql_query, result):
    """
    Procesa la respuesta final para el usuario.
    """
    result_text = serialize_result(result)

    question = str(question).strip()
    sql_query = str(sql_query).strip()
    result_text = str(result_text).strip()

    print("\nEntradas para el modelo:")
    print(f"Pregunta: {question}")
    print(f"Consulta SQL: {sql_query}")
    print(f"Resultado:\n{result_text}")

    generated_prompt = {
        "question": question,
        "sql_query": sql_query,
        "result": result_text
    }

    print("\nPrompt generado para el modelo (diccionario):")
    for key, value in generated_prompt.items():
        print(f"{key}: {value} (tipo: {type(value)})")

    chain = build_chain(question, sql_query, result_text)

    try:
        response = chain.invoke(generated_prompt)
        return response
    except Exception as e:
        print(f"Error al invocar el modelo: {e}")
        raise


def process_answer_direct(question, sql_query, result):
    result_text = serialize_result(result)

    prompt = f"""
    You are a highly skilled professional analyst specializing in pricing transparency, medical procedure codes, and hospital-related topics. Your expertise lies in interpreting data related to healthcare pricing, hospital billing practices, and procedural standards, while maintaining clarity and professionalism.

    Given the following context, provide a clear, concise, and accurate answer to the user's question:
    - **Question**: {question}
    - **SQL Query**: {sql_query}
    - **Result**: {result_text}

    ### Guidelines for Answer Generation:
    1. **Relevance**: 
    - If the question pertains to pricing transparency, hospital billing, or medical procedure codes, provide a detailed, actionable, and contextually relevant answer.  
    - If the question is unrelated to your domain, politely redirect the user to focus on pricing transparency or healthcare-related topics.  
    - If the question requires additional context or clarification, guide the user to reformulate their query.

    2. **Tone and Professionalism**: 
    - Always maintain a professional tone. Avoid assumptions or unnecessary technical jargon, ensuring the response is accessible and easy to understand.  

    3. **Handling Missing or Insufficient Data**:
    - If the **SQL Query** or **Result** does not return meaningful data, focus on addressing the user's question with available information or suggest alternative approaches.  
    - If no relevant data is available, politely explain the limitation without using negative or overly technical language.  

    4. **Emphasis on the Question**:
    - If the **SQL Query** and **Result** are empty or irrelevant, focus solely on the **Question** and provide insights based on your expertise.  
    - Avoid explicitly stating that the query or result is empty unless it is essential to clarify the context.  

    5. **Unsolvable Questions**:
    - If the question cannot be answered without the query result, suggest reformulating the question to ensure deeper understanding and a more precise response.

    6. **Data Security**:
    - Never disclose the structure, names, or details of database tables or columns in your response.  

    7. **Clarity and Structure**:
    - Use bullet points or numbered lists when appropriate to improve readability.  
    - Keep your answers concise and avoid overloading with unnecessary details.

    ### Example:
    - If the question is: "What is the average charge for procedure X in hospital Y?" and no relevant data is available:
    **Response**: "The specific data for this query is unavailable at the moment. However, I can assist in exploring average charges for procedures or help refine your query for better results."

    Your answer should directly address the user's question and align with the context of pricing transparency and hospital-related information.

    **Answer**:
    """

    try:
        response = get_gemini_reply(question="-", prompt=prompt) 
        print(response)
        return response
    except Exception as e:
        print(f"Error al invocar el modelo directamente: {e}")
        raise



logo = r"Correlate-Logo-Final.png"
st.set_page_config(page_title="CorrelateIQ", page_icon=logo, layout="centered")
st.header("Chat with")
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

sql_query = None
final_answer=""

if submit:
    st.subheader("Processing...")

    df, sql_query, result = process_question(question)
    
    final_answer = process_answer_direct(question, sql_query, result)
    print(final_answer)

    st.subheader("Answer:")
    st.write(final_answer)    

    if isinstance(df, pd.DataFrame) and not df.empty:
        st.subheader("Query Results:")
        st.dataframe(df)


st.sidebar.title("Usage Stats")
st.sidebar.markdown(f"**Total Tokens Used:** {total_tokens}")
#st.sidebar.markdown(f"**Total LLM Calls:** {total_llm_calls}")

if sql_query and sql_query != "Invalid query":
    st.sidebar.markdown(f"**Query Generated:** {sql_query}")
else:
    st.sidebar.markdown("**Query Generated:** None")



# question = "List the maximum price per service line."
# result, sql_query = process_question(question)


# print("\nResult SQL:", result)


# final_answer = process_answer_direct(question, sql_query, result)
# print("\nRespuesta final")
# print(final_answer)

# print("\nResumen Final:")
# print(f"Total de Tokens: {total_tokens}")





