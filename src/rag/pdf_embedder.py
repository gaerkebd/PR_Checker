from google import genai
from google.genai import types

with open('owasp-top-10.pdf', 'rb') as f:
    pdf_bytes = f.read()

client = genai.Client()