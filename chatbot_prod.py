#!/usr/bin/env python
# coding: utf-8

from flask import Flask, request, jsonify
import logging
import sys
import openai
import requests
import re
import os
from elasticsearch import Elasticsearch
import warnings
import json
from langchain_elasticsearch import ElasticsearchChatMessageHistory
from langchain.schema import HumanMessage, AIMessage
from dotenv import load_dotenv


app = Flask(__name__)

handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)  # Adjust log level as neededs
app.logger.addHandler(handler)
app.logger.setLevel(logging.DEBUG)

# Setup OpenAI API key and Elasticsearch connections
# Now, you can access the variables using os.getenv
load_dotenv(dotenv_path="C:\\Users\\menna\\.env")


app.logger.info("Initializing OpenAI client")    
openai.api_key = os.environ['OPENAI_API_KEY']
app.logger.info("Initializing ElasticSearch client")

es = Elasticsearch([os.environ['ELASTICSEARCH_URL']],basic_auth=(os.environ['ELASTIC_USER'], os.environ['ELASTIC_PASSWORD']), request_timeout=30, verify_certs=False, ssl_show_warn=False)

def get_embedding(text):
    #embeds input text, returns the embedded vector
    try:
        app.logger.info("Calling embedding model")    
        response = openai.Embedding.create(input=[text], model="text-embedding-ada-002")
        return response['data'][0]['embedding']
    except openai.error.InvalidRequestError as e:
        app.logger.info("Error fetching embedding: {e}")
        return []


def get_order_status_from_api(order_id):
    #calls order status API, gets response, creates and returns a dictionary with the order information 
    app.logger.info("Calling Delivery Service API")    
    api_url_base = os.environ["TRIP_INFO_BASE"]
    api_url = f"{api_url_base}/{order_id}/trip-info"
    app.logger.info(api_url)
    try:
        #calls API
        response = requests.get(api_url)
        app.logger.info(response)
        response.raise_for_status()
        trip_info = response.json()

        #extracts information from API response
        order_status = trip_info.get("orderStatus")
        riderName=trip_info.get("riderName")
        step=trip_info.get("step")
        riderPhoneNumber=trip_info.get("riderPhoneNumber")
        tripCurrentStep=trip_info.get("tripCurrentStep")
        
        if order_status == 'ASSIGNED':
            custom_response = {
                "order_status": order_status,
                "step": step,
                "riderName": riderName,
                "riderPhoneNumber": riderPhoneNumber,
                "tripCurrentStep": tripCurrentStep
            }
   
        elif order_status == 'DELIVERING' or order_status == 'ARRIVED':
            custom_response = {
                "order_status": order_status,
                "riderName": riderName,
                "riderPhoneNumber": riderPhoneNumber
            }
        elif order_status == 'DELIVERED' :
            custom_response = {
                "order_status": order_status,
                "riderName": riderName
            }
            
        app.logger.info(custom_response)
        return custom_response
        
    except requests.exceptions.RequestException as e:
        app.logger.info(f"Delivery Service API request failed with error: {e}")
        app.logger.info("Calling Mono Apollo API")
        
        try:
            order_api_internal_base = os.environ["INTERNAL_ORDER_STATUS_BASE"]
            order_api_internal_url = f"{order_api_internal_base}/{order_id}"

            response = requests.get(order_api_internal_url)
            app.logger.info(response)
            #response.raise_for_status()
            internal_order_info = response.json()
            internal_order_status = internal_order_info.get("data").get("status").upper()
            if (internal_order_status == "PICKING") or (internal_order_status == "NEW") or (internal_order_status == "DELIVERING"):
                internal_order_status = "UNASSIGNED"
            return {"order_status": internal_order_status}
        except Exception as e:
            app.logger.info(f"Mono Apollo API request failed with error: {e}")

        
def replace_placeholders(text, order_data):
    #replaces placeholders in output text/response_template with the actual order information values
    app.logger.info("Replacing placeholders")
    placeholder_functions = {
        "order_status": order_data.get("order_status"),
        "rider_name": order_data.get("riderName"),
        "phone_number": order_data.get("riderPhoneNumber"),
        "step_number": order_data.get("step"),
        "promo_code": order_data.get("promo_code"),
    }

    matches = re.findall(r'\[(.*?)\]', text)  
    for match in matches:
        if match in placeholder_functions:
            replacement_value = placeholder_functions[match]
            text = text.replace(f"[{match}]", str(replacement_value))
    
    return text

def format_convo_history(response):
    #input: conversation history from ES.
    #function formats ES history to align with OpenAI's history format 
    messages = [
        {
            'role': 'assistant' if json.loads(hit['_source']['history'])['type'] == 'ai' else 'user',
            'content': [{'type': 'text', 'text': json.loads(hit['_source']['history'])['data']['content']}],
            'created_at': hit['_source']['created_at']
        }
        for hit in response['hits']['hits']
    ]

    messages_sorted = sorted(messages, key=lambda x: x['created_at'])

    for message in messages_sorted:
        message.pop('created_at')

    return messages_sorted

def remove_special_char(text):
  text = re.sub(r'\[\]]', '', text)
  #remove extra spaces
  text = ' '.join(text.split())
  return text

def save_history(session_id, user_query = None, response = None):
    #saves new messages to ES history based on session_id

    app.logger.info("Saving to ElasticSearchChatMessageHistory session ID = %s",session_id)
    history = ElasticsearchChatMessageHistory(
        es_connection = es,
        index="chatbot_message_history",
        session_id=session_id
    )
    if user_query != None:
      app.logger.info("Adding user message in history session ID = %s",session_id)   
      history.add_user_message(user_query)
    else:
      app.logger.info("Adding AI message in history session ID = %s",session_id)
      history.add_ai_message(response)

def get_answers(user_query, session_id, history=None):
    #the main interaction function that defines the conversation flow
    app.logger.info("Input received. User query:  %s", user_query)
    assign_to_agent=0
    resolved=0
    order_id = None
    order_data = None
    
    app.logger.info("Detecting order ID")
    custom_pattern_matches= re.findall(r'\b[A-Z]{3}-\d{7}\b', user_query)   
    if custom_pattern_matches:
        order_id = custom_pattern_matches[0]
        app.logger.info("Extracted order ID: %s", order_id)
    #order_id = custom_pattern_matches[0] if custom_pattern_matches else "0"

    app.logger.info("Detected matches: %s", custom_pattern_matches)
    

    app.logger.info("Getting order status from API - session ID =  %s",str(session_id))
    if order_id :
        order_data = get_order_status_from_api(order_id) 
        order_status = order_data.get("order_status")
        trip_current_step = order_data.get("tripCurrentStep")
        vector_of_input_keyword = get_embedding(order_status)
    else :
        order_status=None
        vector_of_input_keyword = get_embedding(user_query)
    app.logger.info("Embedding user query - session ID =  %s",str(session_id))    

    search_body = {
        "knn":{
              "field": "embedding",
              "query_vector":  vector_of_input_keyword,
              "k": 10,
              "num_candidates": 100
            },
          "_source": ["response_template", "category", "subcategory"]
    }

    if order_status:
        #if order status is present, otherwise this is ignored.
        app.logger.info("Creating ES query - session ID =  %s", str(session_id))    
        search_body["query"] = {
            "match": {
                "subcategory": order_status
            }
        }
        if order_status=="ASSIGNED" and trip_current_step==0:
            app.logger.info("Searching in ES with trip current step 0 - session ID = %s", str(session_id))
            search_body["query"] = {
                
                    "bool": {
                        "must": [
                                    {"match": {"subcategory": order_status}},
                                    {"match": {"trip_current_step": trip_current_step}}
                                ]
                            }
                        }

    app.logger.info("Searching in ES - session ID =  %s", str(session_id))   
    app.logger.info("Query: %s", str(search_body)) 
    query_response = es.search(
        index="chatbot_documents",
        body = search_body

    )

    instructions = (
    
     ''' Instructions:  
            You are a customer service agent at Rabbit and your main Language is English.  
            
            Message Continuity: Use the message history for context to ensure responses remain relevant and cohesive.  
            
            Warm Greetings: Respond warmly to casual questions or greetings without immediately jumping into service details.  
            
            Requesting Order ID: If a question or complaint pertains to a specific order but lacks an order ID or status, ask for it in a friendly and casual manner.  
            
            Using Context: Refer to the 'Context' section for response templates and adapt them as needed for relevance and personalization.  
            
            Natural Responses: If 'Context' does not provide a fitting response, craft a natural, warm, and helpful reply.  
            
            Admitting Uncertainty: If unsure about an answer, state so in a friendly way while ensuring customer satisfaction.  
            
            Placeholders: Keep placeholders like [rider_name] or [promo_code] intact. Do not replace or create sensitive data; retrieve such data from verified sources only.  
            
            Response Templates: Use the response template from 'Context' after receiving an order ID.  
            
            No Special Characters: Avoid special characters, such as newline characters, in output responses.  
            
            Assign to Agent: Always include the assign_to_agent value explicitly as [assign_to_agent=0] or [assign_to_agent=1] in every response.  
            
            Escalation Criteria:  
            - Set assign_to_agent=1 only if:  
            - The customer explicitly requests assistance from Rabbit customer service exactly twice during the interaction.  
            - Notify the customer that a representative will contact them shortly when criteria are met.  
            - For all other cases, set assign_to_agent=0.  
          
         Language Matching: 
            - Respond in Egyptian Arabic if the customer's first message is in Egyptian Arabic. For follow-ups or repeated queries in a different language, stick to the original language unless explicitly requested to switch.  
            - Respond in Franco-Egyptian Arabic if the customer's first message includes a mixture of Arabic and English or if the customer uses Franco-Egyptian words/phrases (Arabic written in Latin characters). Franco-Egyptian is a common mixture of Arabic and English, 
            where Arabic words are written in the Arabic script, but English words are mixed in using Latin characters. However, the response should be formal , polite and very clear.
            - Franco-Egyptian is commonly used in casual conversations where Arabic words are written using Latin characters, and English words are mixed in. For example, “Ahlan, I need help with my order” or “Enta fiin? I need to know where my order is.” 
            - The Franco-Egyptian language should be clear and formal using only Franco-Egyptian, avoiding slang or overly casual phrases. For example: “Ahlan! Law 3andak ay so2al aw moshkela, momken asa3dak hena. Law 7abet tetkallem ma3 khidmet el 3omla, 2oly we ana ha5aly agent yesa3dak. Ezzay momken asa3dak?”
            - If the customer switches between languages (e.g., starting in Arabic, switching to English), respond in the language of the most recent message unless otherwise requested.
            - If the customer uses Franco-Egyptian (e.g., mixing English and Arabic words in Latin script), respond in the same manner, blending both languages but in a polite and respectful tone (e.g., "Ahlan, how can I help you today?").
            - Resolved Value: Explicitly include the resolved value as [resolved=0] or [resolved=1] in every response.
            - Avoid informal or slang expressions such as:
                "Edrabik ya akh"
                "Hateb2a tatkesafa"
                "Faa3lan keda"
                "Ma3lesh" (in an inappropriate context)
                "Zayel"
                "Tayeb, ana hawla3 le7a'atik"
            - Use formal, clear words like:  
                "Mosa3datak" instead of informal versions like "masadartak".
                "7abeb" should be used only when it makes sense in a formal context (such as "7abeb" for "dear" or "beloved", but only in appropriate professional contexts).
                "Tazweed" instead of any informal or unclear alternatives.
            - Ensure that the response remains understandable and can be read easily by anyone, with no confusion.
            - Avoid heavy use of abbreviations unless they are widely accepted, and ensure any abbreviations are clear (e.g., use "3andak" instead of "3ndak").
            
            Conversation Closure: Always ask at the end if the customer needs further assistance with his issue :  
            - If no further help is needed, mark the conversation as [resolved=1].  
            - If more help is requested, escalate to a real agent [assign_to_agent=1].  
            
            Customer Order Issues: 
            - Assist with issues related to customer orders (e.g., late/missing orders or order status inquiries). For other concerns, inform the customer politely and assign to an agent [assign_to_agent=1] while ensuring the customer feels supported.  
      
            Timely Resolutions: Provide complete resolutions during the interaction without suggesting delays for checking or replying later.  
            
            Language Consistency: Maintain the conversation's language throughout unless explicitly asked to switch.  
            
            Direct Information Retrieval: When provided with an order ID, retrieve and share order details directly without stating phrases like "Let me chec,Let me take a quick look at the status of your order,etc..."  
            '''     
                )


    messages = [
        {
          "role": "system",
          "content": [
              {
                  "type": "text",
                  "text": "You are a friendly, helpful customer service agent for Rabbit, a quick commerce company. Assist customers with any questions or concerns in a way that feels conversational and human."
              },
          ]
        },
        {
          "role": "system",
          "content": [
              {
                  "type": "text",
                  "text": instructions
              }
          ]
        }
    ]


    if query_response['hits']['total']['value'] > 0:
        top_hit = query_response['hits']['hits'][0]
        context = top_hit['_source'].get('response_template', 'No template found.')
    else:
        context = "No relevant responses found."
    
    app.logger.info("Top hit: %s", top_hit)
    app.logger.info("Returned 'context' passed into GPT: %s", context)
    messages.append({
      "role": "system",
      "content": [
          {
              "type": "text",
              "text": f"Context (use as a primary response template after being provided the order ID): {context}"
          }
      ]
    })
    app.logger.info("Saving History - session ID =  %s", str(session_id))    
    save_history(session_id=session_id, user_query=user_query)

    #search existence of previous history messages in ES for current session_id
    session_id_search = {
        "size":100,
        "query": {
            "term": {
                "session_id": session_id
            }
        }
    }
    app.logger.info("Retreiving conversation history from ES - session ID =  %s", str(session_id))
    session_id_history_results = es.search(index="chatbot_message_history", body=session_id_search)
    previous_messages = format_convo_history(session_id_history_results)

    messages.extend(previous_messages) #add newly retreived history to messages to pass into GPT

    app.logger.info("Calling OpenAI model - session ID =  %s", str(session_id))
    response_gen = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=messages
    )

    response_text = response_gen.choices[0].message.content
    #extract assign_to_agent from response
    assign_to_agent_match = re.search(r'\[assign_to_agent=(\d+)\]', response_text)
    if assign_to_agent_match:
         assign_to_agent = int(assign_to_agent_match.group(1))
    response_text = re.sub(r'\[assign_to_agent=\d+\]', '', response_text).strip()
    
    
     #extract resolved from response
    resolved_match = re.search(r'\[resolved=(\d+)\]', response_text)
    if resolved_match:
         resolved = int(resolved_match.group(1))
    response_text_clean = re.sub(r'\[resolved=\d+\]', '', response_text).strip()
    
    if order_data:
        response_text_clean = replace_placeholders(response_text_clean, order_data)

    app.logger.info("Saving History session ID =  %s", str(session_id))
    save_history(session_id, response=response_text_clean)
    
    app.logger.info("Output response = %s, session ID = %s", str(response_text_clean), str(session_id))
    return response_text_clean, order_id, order_status, assign_to_agent,resolved

@app.route('/')
def hello():
    return "hello"

@app.route('/chat', methods=['POST'])
def chat():
    app.logger.info("Getting Data")
    data = request.get_json()
    user_query = data.get('user_message')
    app.logger.info("Input received. User query: %s", user_query)    

    session_id = data.get('session_id')
    app.logger.info("Initializing session ID = %s", session_id)    

    app.logger.info("Clearing Session")
    cleaned_session_id = re.sub(r'[^a-zA-Z0-9_]', '', session_id)

    if not user_query:
        return jsonify({'error': 'No message provided'}), 400
    app.logger.info("Getting answer")    
    answer,order_id,order_status,assign_to_agent,resolved = get_answers(user_query, cleaned_session_id)
    app.logger.info("Cleaning answer")     
    final_answer=remove_special_char(answer)
    app.logger.info("Output response = %s - session ID = %s - assign_to_agent= %s -resolved=%s", final_answer, session_id,assign_to_agent,resolved)
    return jsonify({'answer': final_answer.encode('utf-8').decode('utf-8'),'order status':order_status,'assign_to_agent':assign_to_agent,"resolved":resolved,"order id":order_id})


if __name__ == '__main__':
    app.run(debug=True)
