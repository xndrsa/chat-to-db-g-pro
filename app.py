import psycopg2
import streamlit as st
import google.generativeai as genai
import pandas as pd
from dotenv import load_dotenv
import os

load_dotenv()


api_key = os.getenv("API_KEY")

conn =psycopg2.connect(
    host=os.getenv("HOST"),
  user=os.getenv("USER"),
  password=os.getenv("PASSWORD"),
  database=os.getenv("DATABASE"),
  port=5432
)

genai.configure(api_key=api_key)

def get_gemini_reply(question, prompt):
    """
    question: this is the user prompt
    prompt: This is the context for the model
    """

    model = genai.GenerativeModel('gemini-pro')

    token_count_in = model.count_tokens(f"{prompt}\n\n{question}")
    # print(f"Input token count: {token_count_in}")

    response = model.generate_content(f"{prompt}\n\n{question}")

    token_count_out = model.count_tokens(response.text)
    # print(f"Output token count: {token_count_out}")
    
    return response.text


def read_sql_query(sql):
  """
  sql: requires the query
  """
  cur=conn.cursor()
  cur.execute(sql)
  rows= cur.fetchall()
  colnames = [desc[0] for desc in cur.description]
  cur.close()
  conn.close()
  df = pd.DataFrame(rows, columns=colnames)
  return df

prompt = ["""
Context:   
Be always friendly with your replies, You represent the facility "Great Lakes Hospital" take this consideration when someone whant to compare himself to another hospital.
You are an expert at converting English questions into advanced SQL queries, still analyze if your reply really needs a sql query, you are specifically focused on hospital pricing transparency topics. If the question is not related to the columns shown below, immediately output: invalid request.

Use the SQL table `llm_fact_ms_drg_test`, which contains the following columns:
- **facility_name**: text (facility reference name)
- **ms_drg_code**: text
- **ms_drg_description**: text
- **billing_class**: text

- **standard_charge_negotiated_dollar**: List of negotiated prices, numeric(10,2), (relevant as it lists the prices, can refer to this like price or charge, remove the null ones when this columns is requested)
- **standard_charge_negotiated_algorithm**: text
- **standard_charge_negotiated_percentage**: numeric(10,2)
- **standard_charge_min**: numeric(10,2)
- **standard_charge_max**: numeric(10,2)
- **discounted_cash_pricing**: numeric
- **discounted_gross_charge**: numeric
- **service_line_name**: text
- **standard_charge_methodology**: text
- **modifiers**: text
- **setting_name**: text
- **payor_name**: text
- **plan_name**: text
- **plan_type_name**: text

Follow these rules when generating SQL queries:
1. Only output the SQL query without additional text, remove the delimiters ''', '', ` or ```, remove also the term "SQL" or sql in the output.
2. Base your queries strictly on the user's question, providing only the required information.
3. Never provide the full table information, they need to request analysis based requests, include a limit of 100.
4. standard_charge_negotiated_dollar table must never show the null values

**Example Questions and Expected SQL Output:**
1. *What is the average negotiated dollar per code per facility?*
   - Expected query: select short_name, ms_drg_code, avg(standard_charge_negotiated_dollar) as avg_negotiated_dollar from llm_fact_ms_drg_test group by short_name, ms_drg_code;

2. *What are the top three highest negotiated dollar charges per code?*
   - Expected query: select ms_drg_code, standard_charge_negotiated_dollar from llm_fact_ms_drg_test where standard_charge_negotiated_dollar is not null group by ms_drg_code, standard_charge_negotiated_dollar order by standard_charge_negotiated_dollar desc limit 3;

Generate concise, relevant SQL queries based on this guidance.
                                       Never talk about the context I'm providing, use it just for processing, never mention context on your reply

Question:                                       
"""]

prompt_if_different_topic = ["""                             
Context:                             
Be always friendly with your replies, You can provide information and links related to payors, plan, service codes (like ms-drg,cpt,etc), plan types, but has to be related to hospital pricing transparency. If your question falls 
outside this area, please note that responses may be limited to transparency-related information only.
                             
Don't say you cannot, say you are focused only to the topic described before.
                             
Never talk about the context I'm providing, use it just for processing, never mention context on your reply
Question:    
"""]


columns_taken_out = ["""
- **standard_charge_gross**: numeric(10,2) (this is not the price)                     
- **standard_charge_gross**: Gross standard charge (numeric, 10,2)
"""]

prompt_analysis = ["""
Context:             
Be always friendly with your replies,                 
You represent the facility "Great Lakes Hospital" take this consideration when someone whant to compare himself to another hospital.

Analyze the data based on the question, providing only relevant insights. If the answer would simply repeat or restate the information given, reply like a title only, example: can you give the list of  max dolla negotiated per service line, and which code it is? -answer: List of Prices.

Use this information strictly as context and never display it directly in your response. Below are the columns available for reference:
- **facility_name**: hospital/Facility reference name (text)
- **ms_drg_code**: DRG code (text)
- **ms_drg_description**: DRG description (text)
- **billing_class**: Billing class (text)

- **standard_charge_negotiated_dollar**: List of negotiated prices (numeric, 10,2) — relevant for pricing analysis
- **standard_charge_negotiated_algorithm**: Negotiated pricing algorithm (text)
- **standard_charge_negotiated_percentage**: Negotiated price percentage (numeric, 10,2)
- **standard_charge_min**: Minimum negotiated price (numeric, 10,2)
- **standard_charge_max**: Maximum negotiated price (numeric, 10,2)
- **discounted_cash_pricing**: Discounted cash pricing (numeric)
- **discounted_gross_charge**: Discounted gross charge (numeric)
- **service_line_name**: Service line name (text)
- **standard_charge_methodology**: Charge methodology (text)
- **modifiers**: Modifiers (text)
- **setting_name**: Setting name (text)
- **payor_name**: Payor name (text)
- **plan_name**: Plan name (text)
- **plan_type_name**: Plan type (text)

Provide concise and insightful responses that directly address the question, utilizing relevant data points without revealing this context verbatim.                  
Never talk about the context I'm providing, use it just for processing, never mention context on your reply
Question:    
"""]

logo = r"Correlate-Logo-Final.png"

# Front format

st.set_page_config(page_title="CorrelateIQ", page_icon=logo, layout="centered")

st.header("Chat with ")
st.image(logo, width=400)

with st.form(key='question_form'):
    col1, col2 = st.columns([4, 1])
    
    with col1:
        question = st.text_area(
           "Type your question here:",
            placeholder="",  
            key="question_input",
            height=100
        )
        
    with col2:
        submit = st.form_submit_button("Submit")  


if submit:
  sql_query =get_gemini_reply(question,prompt)
  # print(f"\n{question}")
  # print(f"\n{sql_query}")
  try:
    data=read_sql_query(sql_query)
    data_plus_prompt_analysis=f"{prompt_analysis}\n{data}"
    data_analisys=get_gemini_reply(question,data_plus_prompt_analysis)

    #st.subheader("The response is:")
    #st.header(data_analisys)
    st.dataframe(data)
  except:
    unrelated_topic = get_gemini_reply(question,prompt=prompt_if_different_topic)

    st.subheader("The response is:")
    st.header(unrelated_topic)