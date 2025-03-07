import os
import time
import platform
import psutil
import multiprocessing
import torch
import re
import zipfile
from tqdm import tqdm
from gensim.models import Word2Vec


def extract_and_load_data(zip_path, extract_path="comments_data"):
    # zip path = assignment2/comments2k.zip
    """Extracts a zip file and loads text from all .txt files inside."""
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)

    comments = []
    for filename in os.listdir(extract_path):
        if filename.endswith(".txt"):  # Read only text files
            with open(os.path.join(extract_path, filename), "r", encoding="utf-8") as file:
                comments.append(file.read())  # Read entire file as a string
    return comments

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    return text.split()

def get_sys_specs():
    specs = {
        "CPU": platform.processor(),
        "Cores": multiprocessing.cpu_count(),
        "RAM (GB)": round(psutil.virtual_memory().total/(1024**3), 2),
        "GPU": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None"
    }
    return specs

zip_files = "Assignment2/comments2k.zip"
comments = extract_and_load_data(zip_files)

tokenized_comments = [preprocess_text(comment) for comment in tqdm(comments, desc="Preprocessing")]

start_time = time.time()

model = Word2Vec(
    sentences = tokenized_comments,
    vector_size = 100,
    window = 5,
    min_count = 1,
    skip_g = 0,
    cores = multiprocessing.cpu_count()
)