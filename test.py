from groq import Groq
from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv(dotenv_path=Path('.env'))

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

response = client.chat.completions.create(
    model='llama-3.1-8b-instant',
    messages=[{
        'role': 'user',
        'content': 'How do I prevent rice blast? Answer in 2 sentences.'
    }],
    max_tokens=100
)

print(response.choices[0].message.content)