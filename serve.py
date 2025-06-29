from fastapi import FastAPI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langserve import add_routes
import os
from dotenv import load_dotenv
load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")
model = ChatGroq(model="mixtral-8x7b-32768",
                 api_key=groq_api_key)

# prompt template 
generic_prompt = "Translate the following from english to {language}."
prompt = ChatPromptTemplate.from_messages([
    ("system", generic_prompt),
    ("user", "{text}"),
])

parser = StrOutputParser()

chain = prompt | model | parser

# app defintion 
app = FastAPI(title="Language Translation API",
              description="An API for translating text between languages using Groq's Gemma2 model.",
              version="1.0.0")

add_routes(
    app,
    chain,
    path="/chain"
)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)