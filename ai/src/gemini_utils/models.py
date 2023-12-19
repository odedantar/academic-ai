from langchain_google_genai import ChatGoogleGenerativeAI

from config import GOOGLE_API_KEY


def get_gemini_llm(model_name='gemini-pro') -> ChatGoogleGenerativeAI:

    return ChatGoogleGenerativeAI(model=model_name)
