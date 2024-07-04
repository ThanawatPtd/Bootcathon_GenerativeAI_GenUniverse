import sys
import logging

logging.basicConfig(filename='/home/LogFiles/python_error.log', level=logging.ERROR)

try:
    from flask import Flask, request, abort
    from linebot import LineBotApi, WebhookHandler
    from linebot.exceptions import InvalidSignatureError
    from linebot.models import MessageEvent, TextMessage, TextSendMessage

    import os
    from openai import AzureOpenAI, DefaultHttpxClient
    from dotenv import load_dotenv, find_dotenv
    from azure.core.credentials import AzureKeyCredential  
    from azure.search.documents import SearchClient  
    from azure.search.documents.models import Vector  
    import google.generativeai as genai
except Exception as e:
    logging.error(f"Startup Error: {str(e)}", exc_info=True)
    print(f"Error: {str(e)}", file=sys.stderr)
    raise
app = Flask(__name__)

@app.route('/')
def hello():
    return "Hello, World!"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)