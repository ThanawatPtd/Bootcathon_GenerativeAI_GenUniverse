import sys
import logging

logging.basicConfig(filename='/home/LogFiles/python_error.log', level=logging.ERROR)

try:
    print("Importing libraries")
    from flask import Flask, request, abort
    from linebot import LineBotApi, WebhookHandler
    from linebot.exceptions import InvalidSignatureError
    from linebot.models import MessageEvent, TextMessage, TextSendMessage

    import os
    import textwrap
    from openai import OpenAI
    from openai import AzureOpenAI, DefaultHttpxClient
    from dotenv import load_dotenv, find_dotenv
    from azure.core.credentials import AzureKeyCredential  
    from azure.search.documents import SearchClient  
    from azure.search.documents.models import VectorQuery
    from azure.search.documents.models import VectorizedQuery

    import google.generativeai as genai
except Exception as e:
    logging.error(f"Startup Error: {str(e)}", exc_info=True)
    print(f"Error: {str(e)}", file=sys.stderr)
    raise

# Azure AI Search
service_endpoint = os.environ['AZURE_AI_SEARCH_ENDPOINT']
key = os.environ['AZURE_AI_SEARCH_KEY']
index_name = os.environ['AZURE_AI_SEARCH_INDEX_NAME']
google_key = os.environ['GOOGLE_API_KEY']
credential = AzureKeyCredential(key)

# Azure OpenAI
client = AzureOpenAI(
    api_key=os.environ['AZURE_OPENAI_API_KEY'],
    azure_endpoint=os.environ['AZURE_OPENAI_API_ENDPOINT'],
    api_version=os.environ['AZURE_OPENAI_API_VERSION'],
    http_client=DefaultHttpxClient(verify=False)
)
#Open AI
clientAI = OpenAI(
  api_key=os.environ['OPENAI_API_KEY']
)

embedding_model = os.environ['EMBEDDING_MODEL_NAME']
genai.configure(api_key=google_key)
gpt4o_model = os.environ['GPT4O_MODEL_NAME']
gemini_flash_model = genai.GenerativeModel('gemini-1.5-flash', generation_config={"response_mime_type": "application/json"})

def generate_embeddings(text):
    print("embedding_query: " + text)
    response = client.embeddings.create(input=text, model=embedding_model)
    return response.data[0].embedding


def format_retrieved_documents(data_list):
    retrieved_documents = ""
    for item in data_list:
        content = item.get('content', '')
        sourcepage = item.get('sourcepage', '')
        retrieved_documents += "sourcepage: " + sourcepage + ", content: " + content + "\n"
    return retrieved_documents

def transform_to_gemini(messages_chatgpt):
    messages_gemini = []
    system_promt = ''
    for message in messages_chatgpt:
        if message['role'] == 'system':
            system_promt = message['content']
        elif message['role'] == 'user':
            messages_gemini.append({'role': 'user', 'parts': [message['content']]})
        elif message['role'] == 'assistant':
            messages_gemini.append({'role': 'model', 'parts': [message['content']]})
    if system_promt:
        messages_gemini[0]['parts'].insert(0, f"*{system_promt}*")

    return messages_gemini

def get_chat_completion_from_gemini_pro(messages):
    print('Generating answer using Flash Pro model')
    response = gemini_flash_model.generate_content(messages)
    return response.text

def get_chat_completion_from_gpt4o(messages):
    print('Generating answer using GPT-4.0 model')
    response = clientAI.chat.completions.create(
        model=gpt4o_model,
        messages=messages
    )
    return response.choices[0].message.content

def get_search_results(query):
    search_client = SearchClient(service_endpoint,
                                 index_name, credential=credential)
    
    # vector = VectorQuery(value=generate_embeddings(query), k=3, fields="contentVector")
    vector_query = VectorizedQuery(vector=generate_embeddings(query), k_nearest_neighbors=3, fields="contentVector")
    try:
        results = search_client.search(
        search_text=query,
        vector_queries=[vector_query],
        select=["content", "sourcepage"],
        query_type="semantic",
        semantic_configuration_name="my-semantic-config",
        top=3)
        result_list = []
        for result in results:
            result_list.append(result)
        return result_list
    except Exception as e:
        # logging.error(f"zzzzzzzzz: {str(vector)}", exc_info=True)
        print(f"Error: {str(e)}", file=sys.stderr)
        raise


def get_system_promt(message):
    result_list = get_search_results(message)
    system_prompt = f"""
    ## On your role
    - You are a chatbot for ExxonMobil named Mobilly, designed to help answer customer questions based on retrieved documents and relevant knowledge to refine the answers.
    - You always respond in Thai.
    - Do not include greetings in your responses.
    - Be polite and answer as a kind, helpful assistant.
    - Mention the source of your answers.

    ## Retrieved documents
    {format_retrieved_documents(result_list)}

    ## Instructions
    - Only answer questions related to the topics covered in the retrieved documents.
    - If a question is outside the scope of the retrieved documents, try to answer based on your knowledge.
    """
    return system_prompt

app = Flask(__name__)
app.logger.info(service_endpoint)

# Line Bot credentials
line_bot_api = LineBotApi(os.environ['LINE_CHANNEL_ACCESS_TOKEN'])
handler = WebhookHandler(os.environ['LINE_CHANNEL_SECRET'])

@app.route('/', methods=['GET'])
def home():
    return '<form action="/test" method="POST"><input type="text" name="message" placeholder="Enter your message"><button type="submit">Press</button></form>'

@app.route('/test', methods=['POST'])
def test():
    try:
        message = request.form['message']
        response = process_user_message(message)
    except Exception as e:
        logging.error(f"yyyyyyyyyyyyyyyyyy: {str(e)}", exc_info=True)
        print(f"Error: {str(e)}", file=sys.stderr)
        raise

    # response = message
    return response

@app.route('/callback', methods=['POST'])
def callback():
    signature = request.headers['X-Line-Signature']
    body = request.get_data(as_text=True)
    app.logger.info("Request body: " + body)
    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        app.logger.info("Invalid signature. Please check your channel access token/channel secret.")
        abort(400)
    return 'OK'

@handler.add(MessageEvent, message=TextMessage)
def handle_message(event):
    user_message = event.message.text
    response_message = process_user_message(user_message)
    line_bot_api.reply_message(
        event.reply_token,
        TextSendMessage(text=response_message)
    )

def process_user_message(message):
    messages = [
        {
            "role": "user",
            "content": message
        },
        {
            "role": "system",
            "content": get_system_promt(message)
        }
    ]

    # messages_gemini = transform_to_gemini(messages)
    response = get_chat_completion_from_gpt4o(messages)
    # response = get_chat_completion_from_gemini_pro(messages_gemini)
    return response

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
