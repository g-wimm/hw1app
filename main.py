import os
import streamlit as st
from dotenv import load_dotenv


from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

st.set_page_config(page_title="City Recommender", layout="centered")
st.title("City Recommender with Duration and Budget")

#os.environ["GOOGLE_API_KEY"] = "AIzaSyDubs-CxPBUKKhkwYJU2NgQA6Exxuaw1Qk" # Update this with your Google AI Studio API Key

load_dotenv()

# --- API Key check (Google AI Studio) ---
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    st.error("Missing GOOGLE_API_KEY. Set it in your environment, restart the terminal, then rerun Streamlit.")
    st.stop()

# --- Model (Google Generative AI / AI Studio) ---
model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.1,
    max_output_tokens=256
)

# --- UI ---
my_budget = st.sidebar.selectbox(
    "Your Budget is:",
    ("Less than $1000", "Between $1000 and $2000", "Between $2000 and $5000", "More than $5000"),
)

my_duration = st.sidebar.number_input(
    "Enter the Number of Weeks for Your Vacation",
    min_value=1,
    step=1,
)

col1, col2, col3 = st.sidebar.columns(3)
generate_result = col2.button("Tell Me!")

# --- Prompt + Chain (LCEL) ---
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful travel assistant."),
    ("user",
     "I want to spend a nice vacation for {duration} week(s). "
     "My budget for the entire trip is {budget}. "
     "Suggest a list of 10 cities to visit that would fit this budget. "
     "Return ONLY the city names as a comma-separated list. No explanations.")
])

chain = prompt | model | StrOutputParser()

if generate_result:
    result = chain.invoke({"budget": my_budget, "duration": my_duration})
    st.write(result)