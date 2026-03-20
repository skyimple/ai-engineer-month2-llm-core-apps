import instructor
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv("config.env")

client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)

# Patch with instructor for structured outputs
instructor_client = instructor.from_openai(client)
