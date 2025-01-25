import os
from dotenv import load_dotenv, find_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
_ = load_dotenv(find_dotenv())
# Text model
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL_FLASH20= os.getenv("GEMINI_MODEL_FLASH20")
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", api_key=GEMINI_API_KEY, temperature=0.5, max_tokens=2000,timeout=60,transport="rest")

response = llm.invoke("Explain quantum computing in 50 words.")
print(response.content)