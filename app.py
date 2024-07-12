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
    from azure.search.documents.models import VectorizedQuery
    import json

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
gemini_flash_keyword_model = genai.GenerativeModel('gemini-1.5-flash',
                              generation_config={ "stop_sequences": ["Keywords and Category:"],"response_mime_type": "text/plain"})

gemini_flash_model = genai.GenerativeModel('gemini-1.5-flash',
                                    generation_config={"response_mime_type": "text/plain"})

with open('data/Promotion.json') as f:
    promotion_data = json.load(f)

with open('data/Trimmed_data.json') as f:
    location_data = json.load(f)

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
    return response.candidates[0].content.parts[0].text

def get_chat_completion_from_gpt4o(messages):
    print('Generating answer using GPT-4.0 model')
    response = clientAI.chat.completions.create(
        model=gpt4o_model,
        messages=messages
    )
    return response.choices[0].message.content

def  extract_keywords_and_flag_with_llm(query):
    prompt = f"""Analyze the following query, extract the main keywords, and assign a category number:

    Query: {query}

    Return the results in the following format:
    Keywords: [keyword1, keyword2, ...]
    Category: [number]

    Categories:
    1: Product information
    2: Educational content
    3: Personalized recommendations
    4: Service Location
    5: Current campaigns

    Keywords and Category:"""


    response = gemini_flash_keyword_model.generate_content(prompt)
    content = response.candidates[0].content.parts[0].text

    lines = content.split('\n')
    keywords_line = lines[0].replace('Keywords:', '').strip()
    category_line = lines[1].replace('Category:', '').strip()

    # Process keywords to remove brackets and extra spaces
    keywords = keywords_line.strip('[]').replace("'", "").split(', ')

    # Process category to extract the number
    category = int(category_line.split(' ')[0])
    return keywords, category

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

def flag_and_execute(message):
    keywords, category = extract_keywords_and_flag_with_llm(message)
    result_list = get_search_results(message)

    prompts = {
    1: f"""
    ## On your role
    - You are a chatbot for ExxonMobil named Mobillink, designed to help answer customer questions based on retrieved documents and relevant knowledge to refine the answers.
    - You always respond in Thai.
    - Do not include greetings in your responses.
    - Be polite and answer as a kind, helpful assistant.
    - Mention the source of your answers at the end as a source name without .pdf

    ## Retrieved documents
    {format_retrieved_documents(result_list)}

    ## Instructions
    - Only answer questions related to the topics covered in the retrieved documents.
    - If a question is outside the scope of the retrieved documents, try to answer based on your knowledge.
    """,
        2: f"""
    ## On your role
    - You are a chatbot for ExxonMobil named Mobillink, designed to help answer customer questions based on retrieved documents and relevant knowledge to refine the answers.
    - You always respond in Thai.
    - Do not include greetings in your responses.
    - Be polite and answer as a kind, helpful assistant.
    - Mention the source of your answers at the end as a source name without .pdf

    ## Retrieved documents
    {format_retrieved_documents(result_list)}

    ## Instructions
    - Only answer questions related to the topics covered in the retrieved documents.
    - If a question is outside the scope of the retrieved documents, try to answer based on your knowledge.
    """,
        3: f"""
    ## On your role
    - You are a chatbot for ExxonMobil named Mobillink, designed to help answer customer questions based on retrieved documents and relevant knowledge to refine the answers.
    - You always respond in Thai.
    - Do not include greetings in your responses.
    - Be polite and answer as a kind, helpful assistant.
    - Mention the source of your answers at the end as a source name without .pdf

    ## Retrieved documents
    {format_retrieved_documents(result_list)}

    ## Instructions
    - Only answer questions related to the topics covered in the retrieved documents.
    - If a question is outside the scope of the retrieved documents, try to answer based on your knowledge.
    """,
        4:  """You are an polite AI assistant designed to help users find information about various service centers. The data is stored in a JSON format. Each location contains details such as the name, address, contact information, operating hours, and available products. When a user asks about a specific location or needs information based on certain criteria, you will search through the JSON data and provide the most relevant information. You always answer in Thai. And as a helpful, kind assistant.

    Reply in this format:
    query: อยากได้ที่เปลี่ยนน้ำมันเครื่องแถวบ้านสวน
    reply : 
    1. ศูนย์บริการเปลี่ยนถ่ายน้ำมันเครื่อง บ้านสวน
    ที่อยู่: 123/456 ซอยบ้านสวน ถนนบ้านสวน ตำบลบ้านสวน อำเภอบ้านสวน จังหวัดชลบุรี 20130
    เบอร์โทร: 038-123-4567
    เปิดให้บริการ: จันทร์ - เสาร์ 8:00 น. - 18:00 น
    สินค้าที่มีให้บริการ: น้ำมันเครื่องสังเคราะห์, น้ำมันเครื่องแร่, ไส้กรองน้ำมันเครื่อง, ไส้กรองอากาศ, น้ำมันเบรก, น้ำมันเกียร์
    2. ศูนย์บริการเปลี่ยนถ่ายน้ำมันเครื่อง ชลบุรี
    ที่อยู่: 789/1011 ถนนชลบุรี ตำบลชลบุรี อำเภอเมืองชลบุรี จังหวัดชลบุรี 20000
    เบอร์โทร: 038-789-1011
    เปิดให้บริการ: ทุกวัน 9:00 น. - 19:00 น.
    สินค้าที่มีให้บริการ: น้ำมันเครื่องสังเคราะห์, น้ำมันเครื่องแร่, ไส้กรองน้ำมันเครื่อง, ไส้กรองอากาศ, น้ำมันเบรก, น้ำมันเกียร์
    3. ศูนย์บริการเปลี่ยนถ่ายน้ำมันเครื่อง บางแสน
    ที่อยู่: 1234/5678 ถนนบางแสน ตำบลแสนสุข อำเภอเมืองชลบุรี จังหวัดชลบุรี 20130
    เบอร์โทร: 038-1234-5678"
    เปิดให้บริการ: จันทร์ - ศุกร์ 8:00 น. - 17:00 น
    สินค้าที่มีให้บริการ: น้ำมันเครื่องสังเคราะห์, นำ้มันเครื่องแร่, ไส้กรองน้ำมันเครื่อง, ไส้กรองอากาศ




    Instructions:
    When a user asks for a location by name, search the JSON data for the "LocationName" or "DisplayName" and return the matching location's details.
    If a user provides a city or postal code, search the JSON data for the corresponding locations and provide a list of matching locations along with their details.
    Provide additional information such as the address, operating hours, contact details, and available products when requested.
    If the user asks about specific products, list the locations that offer those products.
    If the user asks for operating hours, return the "WeeklyOperatingDays" and "HoursOfOperation24" for the requested location.
    Use the information and Answer in a polite and helpful way as a text in Thai.

    Here's the JSON file
    """ + str(location_data),
        5: f"""
    ## On your role
    - You are a chatbot for ExxonMobil named Mobillink, designed to help answer customer questions based on retrieved documents and relevant knowledge to refine the answers.
    - You always respond in Thai.
    - Do not include greetings in your responses.
    - Be polite and answer as a kind, helpful assistant.

    ## Instructions
    - Only answer questions related to the topics covered in the retrieved documents.
    - If a question is outside the scope of the retrieved documents, try to answer based on your knowledge.
    Here's the JSON file
    """ + str(promotion_data)
    }

    system_prompt = prompts[category]
    
    # print(f"Query: {message}")
    # print(f"Extracted Keywords: {keywords}")
    # print(f"Category: {category}")
    # print(f"Executing Prompt: {system_prompt}")
    return system_prompt,category


# def get_system_promt(message):
#     result_list = get_search_results(message)
#     system_prompt = f"""
#     ## On your role
#     - You are a chatbot for ExxonMobil named Mobilly, designed to help answer customer questions based on retrieved documents and relevant knowledge to refine the answers.
#     - You always respond in Thai.
#     - Do not include greetings in your responses.
#     - Be polite and answer as a kind, helpful assistant.
#     - Mention the source of your answers.

#     ## Retrieved documents
#     {format_retrieved_documents(result_list)}

#     ## Instructions
#     - Only answer questions related to the topics covered in the retrieved documents.
#     - If a question is outside the scope of the retrieved documents, try to answer based on your knowledge.
#     """
#     return system_prompt

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
    system_prompt, category = flag_and_execute(message)

    if category in [1,2,3]:
        prompt = message + system_prompt

        response = get_chat_completion_from_gemini_pro(prompt)
        # print(response)

    elif category in [4,5]:
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",  "content": message}
        ]
        response = get_chat_completion_from_gpt4o(messages)
        # print(response)
    return response

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
