# -*- coding: utf-8 -*-
"""
Created on Sun May 21 19:50:57 2023

@author: willi
"""
#imports
import pandas as pd
import os
import streamlit as st
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.chat_models import ChatOpenAI

#setting up env vars and directories
base_dir = os.path.dirname(os.path.abspath('__file__'))

if 'openai_api_key' not in st.session_state:
    st.session_state['openai_api_key'] = 'you wish'

super_db_df = pd.read_csv(base_dir + '/total_super_scrape_results.csv')
persist_directory = base_dir + "/chromadb_persist_dr"

# Setting up vectordb
texts = list(super_db_df.body)
try:
    embeddings = OpenAIEmbeddings(openai_api_key = st.session_state['openai_api_key'])
except:
    st.write('Your OpenAI API key seems to be invalid. Please check the textbox on the left and make sure you have input your API key correctly')

if not os.path.exists(persist_directory + "/chroma-embeddings.parquet"):
    docsearch = Chroma.from_texts(texts, 
                                  embeddings, ## Don't need this because results are worse with search by vector
                                  metadatas = [{'source': super_db_df.loc[i]['URL']} for i in super_db_df.index],
                                  persist_directory = persist_directory)
    docsearch.persist()
else:
    docsearch = Chroma(persist_directory = persist_directory, embedding_function = embeddings)

chain = load_qa_with_sources_chain(ChatOpenAI(model = 'gpt-3.5-turbo', 
                                          temperature=0,
                                          openai_api_key = st.session_state['openai_api_key']), 
                                   chain_type="map_reduce")

#Streamlit layout
st.set_page_config(page_title="SuperGPT", page_icon="📖", layout="wide")
st.title('SuperGPT')
st.text('Ask any of your Australian Super related questions here!')
query = st.text_input('Write your question here!', value = ' ')
# query_vec = embeddings.embed_query(query)
askbutton = st.button('ask!')

with st.sidebar:
    openai_api_key = st.text_input('put your OpenAI API key here and hit "enter"', key = 'openai_api_key')
    st.markdown('')
    st.markdown('')
    st.markdown('''
                **NOTE:** This app is a proof of concept and does NOT guarantee accurate results.
                This is purely an exercise in demonstrating the power of retrieval-augmented LLM-driven QA.
                For more accurate super advice, please speak to a tax agent or financial advisor.
                '''
                )

#streamlit logic
if askbutton:
    try:
        if query.strip() != '' or query.strip() is not None or query.strip() != ['']:
            docs = docsearch.similarity_search(query) #Could swap this out for simlarity search by vector but won't
            answer = chain({"input_documents": docs, "question": query}, 
                           return_only_outputs=True)['output_text']
            answers = answer.split('SOURCES:')
            answers[1] = answers[1].replace(',', ' \n ').replace(' ', ' \n ')
            st.markdown(answers[0])
            st.write(f'For more information, see here:  \n  {str(answers[1])}', unsave_allow_html = False)
        elif query.strip() == '' or query.strip() is None:
            st.write('You need to write words!')
        else:
            st.write('You need to write words!')
    except:
        st.write('You need to write words! Either that, or your API key has broken, or something else.')