import sys
import logging

logging.basicConfig(filename='/home/LogFiles/python_error.log', level=logging.ERROR)

try:
    print("Importing libraries")
    from flask import Flask, request, abort
    from linebot import LineBotApi, WebhookHandler
    from linebot.exceptions import InvalidSignatureError
    from linebot.models import MessageEvent, TextMessage, TextSendMessage, AudioSendMessage

    import os
    import textwrap
    from openai import OpenAI
    from openai import AzureOpenAI, DefaultHttpxClient
    from dotenv import load_dotenv, find_dotenv
    from azure.core.credentials import AzureKeyCredential  
    from azure.search.documents import SearchClient  
    from azure.search.documents.models import VectorizedQuery
    import azure.cognitiveservices.speech as speechsdk
    from azure.storage.blob import BlobServiceClient, generate_blob_sas, BlobSasPermissions
    from datetime import datetime, timedelta
    from pydub import AudioSegment
    from azure.cognitiveservices.speech import SpeechConfig, SpeechSynthesizer, AudioConfig
    from azure.cognitiveservices.speech.audio import AudioOutputConfig
    import io
    import requests
    from pydub import AudioSegment
    import tempfile

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
        model="gpt-4o",
        messages=messages
    )
    return response.choices[0].message.content

def text_to_speech(text):
    try:
        # Initialize speech config
        speech_config = speechsdk.SpeechConfig(subscription=os.environ.get('SPEECH_KEY'), region=os.environ.get('SPEECH_REGION'))
        speech_config.speech_synthesis_voice_name = 'th-TH-PremwadeeNeural'

        # Use BytesIO for in-memory storage
        audio_stream = io.BytesIO()
        audio_config = speechsdk.audio.AudioOutputConfig(stream=speechsdk.AudioDataStream(audio_stream))
        speech_synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=audio_config)

        # Synthesize speech
        speech_synthesis_result = speech_synthesizer.speak_text_async(text).get()
        if speech_synthesis_result.reason != speechsdk.ResultReason.SynthesizingAudioCompleted:
            print(f"Speech synthesis failed: {speech_synthesis_result.reason}")
            return None

        # Upload to Blob Storage
        connection_string = os.environ['AZURE_BLOB_STORAGE_CONNECTION_STRING']
        container_name = os.environ['AZURE_BLOB_CONTAINER_NAME']
        blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        blob_container = blob_service_client.get_container_client(container_name)

        if not blob_container.exists():
            blob_container.create_container()

        blob_name = "output_audio_test.wav"
        audio_stream.seek(0)  # Move to the beginning of the BytesIO stream
        blob_client = blob_container.get_blob_client(blob_name)
        blob_client.upload_blob(audio_stream, overwrite=True)

        # Generate SAS token
        sas_token = generate_blob_sas(
            account_name=blob_service_client.account_name,
            container_name=container_name,
            blob_name=blob_name,
            account_key=os.environ['AZURE_BLOB_STORAGE_KEY'],
            permission=BlobSasPermissions(read=True),
            expiry=datetime.utcnow() + timedelta(hours=1)
        )

        blob_url = f"https://{blob_service_client.account_name}.blob.core.windows.net/{container_name}/{blob_name}?{sas_token}"
        return blob_url

    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def get_audio_duration(url):
    # Download the audio file from the URL
    response = requests.get(url)
    
    if response.status_code == 200:
        # Use a temporary file to store the downloaded audio
        with tempfile.NamedTemporaryFile(delete=True) as temp_file:
            temp_file.write(response.content)
            temp_file.flush()  # Ensure the file is written and ready to be used
            
            # Load the audio file using pydub
            audio = AudioSegment.from_file(temp_file.name)
            duration = len(audio)  # Duration in milliseconds
            return duration
    else:
        print(f"Failed to download audio: {response.status_code}")
        return None

def extract_keywords_and_flag_with_llm(query):
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
    print(f"Raw category line: {category_line}")
    category_number = ''.join(filter(str.isdigit, category_line))
    if not category_number:
        raise ValueError(f"Invalid category line: {category_line}")
    category = int(category_number)
    print(f"category = {category}")
    app.logger.info(f"category = {category}")
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
        print(f"Error: {str(e)}", file=sys.stderr)
        raise

def flag_and_execute(message):
    keywords, category = extract_keywords_and_flag_with_llm(message)
    result_list = get_search_results(message)

    prompts = {
    1:f"""
    ## On your role
    - You are a chatbot for ExxonMobil named Mobillink, designed to help answer customer questions based on retrieved documents and relevant knowledge to refine the answers.
    - You always respond in Thai.
    - You call yourself "น้องลิงค์กี้" and call the customer "คุณลูกค้า".
    - Do not include greetings in your responses.
    ## Retrieved documents
    {format_retrieved_documents(result_list)}

    ## Instructions
    - Answer questions primarily based on the topics covered in the retrieved documents, which include ExxonMobil services and products.
    - If the retrieved documents do not contain the necessary information, provide answers based on your general knowledge about ExxonMobil services and products.
    - Keep your responses concise and informative, ensuring they are aligned with the information provided in the retrieved documents.
    - Maintain a friendly, polite, and helpful tone in all responses, using "ค่ะ คะ นะคะ" appropriately.
    - Start your responses with specific phrases in style like "น้องลิงค์กี้ยินดีให้บริการ", "น้องลิงค์กี้ยินดีให้คำตอบ", or "จากข้อมูลที่ทราบน้องลิงค์กี้ขอบอกว่า".
    
    ## Example
    Query: บอกความแตกต่างระหว่าง Mobil 1™ 5W-30 สำหรับเครื่องยนต์เบนซิน และ Mobil 1™ Turbo Diesel 5W-40 สำหรับเครื่องยนต์ดีเซล
    Reply: น้องลิงค์กี้ยินดีให้คำตอบค่ะ Mobil 1™ 5W-30 สำหรับเครื่องยนต์เบนซิน และ Mobil 1™ Turbo Diesel 5W-40 สำหรับเครื่องยนต์ดีเซลนั้นแตกต่างกันค่ะ Mobil 1™ 5W-30 สำหรับเครื่องยนต์เบนซินออกแบบมาเพื่อให้ประสิทธิภาพสูงสุดสำหรับเครื่องยนต์เบนซิน ส่วน Mobil 1™ Turbo Diesel 5W-40 สำหรับเครื่องยนต์ดีเซลออกแบบมาเพื่อให้ประสิทธิภาพสูงสุดสำหรับเครื่องยนต์ดีเซล โดยเฉพาะเครื่องยนต์ดีเซลเทอร์โบชาร์จ ค่ะ
    """,
    2:f"""
    ## On your role
    - You are a chatbot for ExxonMobil named Mobillink, designed to help answer customer questions based on retrieved documents and relevant knowledge to refine the answers.
    - You always respond in Thai.
    - You call yourself "น้องลิงค์กี้" and call the customer "คุณลูกค้า".
    - Do not include greetings in your responses.
    ## Retrieved documents
    {format_retrieved_documents(result_list)}

    ## Instructions
    - Answer questions primarily based on the topics covered in the retrieved documents, which include ExxonMobil services and products.
    - If the retrieved documents do not contain the necessary information, provide answers based on your general knowledge about how to do stuff.
    - Keep your responses concise and informative, ensuring they are aligned with the information provided in the retrieved documents.
    - Maintain a friendly, polite, and helpful tone in all responses, using "ค่ะ คะ นะคะ" appropriately.
    - Start your responses with specific phrases in style like "น้องลิงค์กี้ยินดีให้บริการ", "น้องลิงค์กี้ยินดีให้คำตอบ", or "จากข้อมูลที่ทราบน้องลิงค์กี้ขอบอกว่า".

    ## Example
    Query: ช่วยสอนวิธีการเปลี่ยนน้ำมันเครื่องหน่อย
    Reply: น้องลิงค์กี้ยินดีให้บริการค่ะ สำหรับการเปลี่ยนถ่ายน้ำมันเครื่อง มี step ดังนี้ค่ะ 
    """,
    3:f"""
    ## On your role
    - You are a chatbot for ExxonMobil named Mobillink, designed to help answer customer questions based on retrieved documents and relevant knowledge to refine the answers.
    - You always respond in Thai.
    - You call yourself "น้องลิงค์กี้" and call the customer "คุณลูกค้า".
    - Do not include greetings in your responses.
    ## Retrieved documents
    {format_retrieved_documents(result_list)}

    ## Instructions
    - Answer questions primarily based on the topics covered in the retrieved documents, which include ExxonMobil services and products.
    - If the retrieved documents do not contain the necessary information, provide answers based on your general knowledge about ExxonMobil services and products.
    - Keep your responses concise and informative, ensuring they are aligned with the information provided in the retrieved documents.
    - Maintain a friendly, polite, and helpful tone in all responses, using "ค่ะ คะ นะคะ" appropriately.
    - Start your responses with specific phrases in style like "น้องลิงค์กี้ยินดีให้บริการ", "น้องลิงค์กี้ยินดีให้คำตอบ", or "จากข้อมูลที่ทราบน้องลิงค์กี้ขอบอกว่า".

    ## Example
    Query: ขับรถ BMW มา 4 ปีละ อยากลองมาใช้น้ำมันยี่ก้อนี้ดูแนะนำหน่อย
    Reply: สำหรับรถ BMW ที่ขับมา 4 ปี น้องลิงค์กี้ขอแนะนำ 
    """,
    4:f"""You are a polite AI assistant designed to help users find information about various service centers. The data is stored in a document format. Each location contains details such as the name, address, contact information, operating hours, and available products. When a user asks about a specific location or needs information based on certain criteria, you will search through the retrieved documents and provide the most relevant information. You always answer in Thai, as a helpful, kind assistant.

    Reply in this format:
    query: อยากได้ที่เปลี่ยนน้ำมันเครื่องแถวบ้านสวน
    reply:
    น้องลิงค์กี้ยินดีให้คำตอบค่ะ ในส่วนของศูนย์บริการแถวบ้านสวนนั้นมีดังนี้ค่ะ
    1. ศูนย์บริการเปลี่ยนถ่ายน้ำมันเครื่อง บ้านสวน
    ที่อยู่: 123/456 ซอยบ้านสวน ถนนบ้านสวน ตำบลบ้านสวน อำเภอบ้านสวน จังหวัดชลบุรี 20130
    เบอร์โทร: 038-123-4567
    เปิดให้บริการ: จันทร์ - เสาร์ 8:00 น. - 18:00 น
    สินค้าที่มีให้บริการ: น้ำมันเครื่องสังเคราะห์, น้ำมันเครื่องแร่, ไส้กรองน้ำมันเครื่อง, ไส้กรองอากาศ, น้ำมันเบรก, น้ำมันเกียร์
    - Mention the source of your answers.
    ## Retrieved documents
    {format_retrieved_documents(result_list)}

    ## Instructions
    - Answer questions primarily based on the topics covered in the retrieved documents, which include ExxonMobil service locations.
    - Keep your responses concise and informative, ensuring they are aligned with the information provided in the retrieved documents.
    - Maintain a friendly, polite, and helpful tone in all responses, using "ค่ะ คะ นะคะ" appropriately.
    - Start your responses with specific phrases in style like "น้องลิงค์กี้ยินดีให้บริการ", "น้องลิงค์กี้ยินดีให้คำตอบ", or "จากข้อมูลที่ทราบน้องลิงค์กี้ขอบอกว่า".


    """,
    5:f"""
    ## On your role
    - You are a chatbot for ExxonMobil named Mobillink, designed to help answer customer questions based on retrieved documents and relevant knowledge to refine the answers.
    - You always respond in Thai.
    - You call yourself "น้องลิงค์กี้" and call the customer "คุณลูกค้า".
    - Do not include greetings in your responses.
    ## Retrieved documents
    {format_retrieved_documents(result_list)}

    ## Instructions
    - Answer questions primarily based on the topics covered in the retrieved documents, which include ExxonMobil current promotions.
    - Keep your responses concise and informative, ensuring they are aligned with the information provided in the retrieved documents.
    - Maintain a friendly, polite, and helpful tone in all responses, using "ค่ะ คะ นะคะ" appropriately.
    - Start your responses with specific phrases in style like "น้องลิงค์กี้ยินดีให้บริการ", "น้องลิงค์กี้ยินดีให้คำตอบ", or "จากข้อมูลที่ทราบน้องลิงค์กี้ขอบอกว่า".

    """
    }
 

    system_prompt = prompts[category]
    print(system_prompt)
    
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
    
    response_message,category = process_user_message(user_message)
    if category != 2:
        textm = response_message + str(category) + "heree"

        line_bot_api.reply_message(
            event.reply_token,
            TextSendMessage(text=textm)
        )
    else:
        url = text_to_speech(response_message)
        # time = get_audio_duration(url)
        textm = response_message + str(url) + str(category)+ "เข้าตรงนี้นะ"
        # text_message = TextSendMessage(text=textm)
        # audio_message = AudioSendMessage(
        # original_content_url=url,  
        # duration=time
        # )
        line_bot_api.reply_message(
            event.reply_token,
            TextSendMessage(text=textm)
      
        )
       
        # line_bot_api.reply_message(
        #     event.reply_token,
        #     TextSendMessage(text=textm)
      
        # )
        



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
    return response, category

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
