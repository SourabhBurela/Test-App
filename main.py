import streamlit as st
import openai
import requests
from openai import Image
from PIL import Image
import base64
from io import BytesIO
import multiprocessing
from multiprocessing import Pool
import os
from dotenv import load_dotenv

from src.Image_Generation import page1
from src.Image_Variation import page2
from src.Image_Edits import page3
load_dotenv()

# OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

st.set_page_config(page_title="Multipage Streamlit App")
pages = {
    "Image Generation": page1,
    "Image Variation": page2,
    "Image Edits": page3,
}

st.sidebar.title("Navigation")
selection = st.sidebar.radio("Go to", list(pages.keys()))

page_func = pages[selection]
page_func()
