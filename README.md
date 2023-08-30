# SuperGPT
 A proof-of-concept retrieval-augmented QA engine running on GPT 3.5 Turbo using the OpenAI API

## Background
This is a quick proof of concept I threw together to demonstrate how quickly and easily a retrieval-augmented LLM solution could be put to gether.

I scraped the data from [this website](https://www.ato.gov.au/Business/Super-for-employers/) to create a database, which I passed through Chromadb and OpenAI to create a vector database with embeddings.

I then created a UI and prompt-engineered QA engine with the OpenAI API, LangChain, and Streamlit.

Initial results of this PoC are promising, but it is obvious more prompt engineering, vector database finagling, and perhaps even fine-tuning (oh my!) will be necessary before it becomes a Super genius.

## How to run
You will need a local copy of this repo, as well as Python (this app was built on 3.10, but I think 3.8+ will do), and an OpenAI API Key (and some money, if your free trial has run out).

In terms of funky packages, there's:
- streamlit
- langchain
- openai
- chromadb

(apologies - I didn't build this in a venv and pipreqs refuses to work)

I'm sure there is a way for this to be run remotely, or perhaps I could publish this "dashboard" on streamlit, but I am a silly little boy.
