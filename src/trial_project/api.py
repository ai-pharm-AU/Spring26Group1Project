# This file contains helpers for using the openai API

from openai import AsyncOpenAI, OpenAI
import asyncio
from dotenv import load_dotenv

load_dotenv()

client = AsyncOpenAI()
model = "gpt-5-mini"
instructions = "w/e"

def generate_client():
    return OpenAI() 

#def get_async_response(message)

