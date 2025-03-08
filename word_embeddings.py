import os
import re
import numpy as np
from nltk.tokenize import word_tokenize
import nltk
from gensim.models import Word2Vec, KeyedVectors
import time
import threading
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

class SentimentClassifier(nn.Module):
    def __init__(self, input_size):
        super(SentimentClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, 10)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(10, 1)
        self.sigmoid = nn.Sigmoid() 

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x

"""
    Stuff for preprocessing and tokenization. Also subroutines for reading zip.
"""
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt_tab')

def load_and_preprocess_data(data_dir="comments2k"):
    pos_dir = os.path.join(data_dir, "comments1k_pos")
    neg_dir = os.path.join(data_dir, "comments1k_neg")
    
    all_comments = []
    
    # load positive comments
    if os.path.exists(pos_dir):
        for filename in os.listdir(pos_dir):
            if filename.endswith('.txt'):
                with open(os.path.join(pos_dir, filename), 'r', encoding='utf-8', errors='ignore') as f:
                    all_comments.append(f.read())
        print(f"\nLoaded {len(all_comments)} positive comments")
    else:
        print(f"Warning: Directory {pos_dir} not found")
    
    # load negative comments
    neg_count = 0
    if os.path.exists(neg_dir):
        for filename in os.listdir(neg_dir):
            if filename.endswith('.txt'):
                with open(os.path.join(neg_dir, filename), 'r', encoding='utf-8', errors='ignore') as f:
                    all_comments.append(f.read())
                neg_count += 1
        print(f"Loaded {neg_count} negative comments")
    else:
        print(f"Warning: Directory {neg_dir} not found")
    
    print(f"Total: Loaded {len(all_comments)} comments")
    
    tokenized_comments = []
    for comment in all_comments:
        comment = comment.lower()
        comment = re.sub(r'[^a-z\s]', ' ', comment)
        comment = re.sub(r'\s+', ' ', comment).strip()

        tokens = word_tokenize(comment)

        tokens = [token for token in tokens if len(token) > 1]
        
        tokenized_comments.append(tokens)
    
    return tokenized_comments


def monitor_cpu_usage(cpu_usage_list, training_flag):
    while training_flag[0]:
        cpu_usage_list.append(psutil.cpu_percent(interval=0.5))
"""
    Question 1) Function to train Word2Vec CBOW with CPU monitoring
"""
def train_word2vec_cbow(tokenized_comments, vector_size, window=5, min_count=1):
    training_flag = [True] 
    cpu_usage_list = []

    cpu_thread = threading.Thread(target=monitor_cpu_usage, args=(cpu_usage_list, training_flag))
    cpu_thread.start()

    print("\nStarting Word2Vec CBOW training...")
    start_time = time.time()

    model = Word2Vec(
        sentences=tokenized_comments,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=4,
        sg=0,       
        epochs=5   
    )
    
    training_time = time.time() - start_time
    print(f"Word2Vec CBOW model trained in {training_time:.2f} seconds")
    
    training_flag[0] = False
    cpu_thread.join() 

    avg_cpu_usage = sum(cpu_usage_list) / len(cpu_usage_list) if cpu_usage_list else 0
    print(f"Average CPU Usage During Training: {avg_cpu_usage:.2f}%")

    return model, training_time, avg_cpu_usage

"""
    Question 2) Train word embeddings vectors using word2vec
"""
def prepare_glove_corpus(data_dir="comments2k", output_file="comments2k_corpus.txt"):
    pos_dir = os.path.join(data_dir, "comments1k_pos")
    neg_dir = os.path.join(data_dir, "comments1k_neg")

    with open(output_file, 'w', encoding='utf-8') as outfile:
        # Process positive comments
        for filename in os.listdir(pos_dir):
            if filename.endswith('.txt'):
                with open(os.path.join(pos_dir, filename), 'r', encoding='utf-8', errors='ignore') as infile:
                    text = infile.read().lower()
                    text = re.sub(r'[^a-z\s]', ' ', text)
                    text = re.sub(r'\s+', ' ', text).strip()
                    outfile.write(text + '\n')

        # Process negative comments
        for filename in os.listdir(neg_dir):
            if filename.endswith('.txt'):
                with open(os.path.join(neg_dir, filename), 'r', encoding='utf-8', errors='ignore') as infile:
                    text = infile.read().lower()
                    text = re.sub(r'[^a-z\s]', ' ', text)
                    text = re.sub(r'\s+', ' ', text).strip()
                    outfile.write(text + '\n')

    print(f"Corpus saved as {output_file}")

def load_glove_vectors(file_path):
    word_vectors = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        try:
            for line in f:
                values = line.strip().split()
                word = values[0]
                vector = np.array(values[1:], dtype='float32')
                word_vectors[word] = vector
        except:
            print("Couldnt load GloVe vectors")
    print("Loaded successfully")
    return word_vectors

"""
    Question 3) Similar words
"""
def find_most_similar(word, word_vectors, top_n=10):
    if word not in word_vectors:
        return []
    
    word_vec = word_vectors[word]
    similarities = {}
    
    for w, vec in word_vectors.items():
        if w != word:
            # Compute cosine similarity
            dot_product = np.dot(word_vec, vec)
            norm_product = np.linalg.norm(word_vec) * np.linalg.norm(vec)
            similarity = dot_product / norm_product
            similarities[w] = similarity
    sorted_words = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
    
    return sorted_words[:top_n]

"""
    Question 5) Neural network
"""
def sentence_to_vector(sentence, model, vector_size):
    vectors = [model.wv[word] for word in sentence if word in model.wv]
    if len(vectors) == 0:
        return np.zeros(vector_size)
    return np.mean(vectors, axis=0) 

def convert_to_vectors(data, model, vector_size):
    return np.array([sentence_to_vector(sentence, model, vector_size) for sentence in data])

# Training function
def train_model(X_train, y_train, X_val, y_val, vector_size, num_epochs=10):
    
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)  # Reshape for binary classification
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)

    train_data = TensorDataset(X_train_tensor, y_train_tensor)
    val_data = TensorDataset(X_val_tensor, y_val_tensor)
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=32, shuffle=False)

    # Initialize model, loss function, and optimizer
    model = SentimentClassifier(vector_size)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            predictions = model(X_batch)
            loss = criterion(predictions, y_batch)
            loss.backward()
            optimizer.step()
        
        # Validation loss
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                predictions = model(X_batch)
                val_loss += criterion(predictions, y_batch).item()
        
        print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {loss.item():.4f}, Validation Loss: {val_loss / len(val_loader):.4f}")

    return model

def evaluate_model(model, X_test, y_test):
    model.eval()
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)
    
    with torch.no_grad():
        predictions = model(X_test_tensor)
        predictions = (predictions.numpy() > 0.5).astype(int)
    
    accuracy = accuracy_score(y_test, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, predictions, average="binary")
    
    return accuracy, precision, recall, f1

def generate_random_embeddings(shape):
    return np.random.uniform(-1, 1, shape)


if __name__ == "__main__":
    import psutil

    tokenized_comments = load_and_preprocess_data("comments2k")  # all 2k comments should be here 
    print(f"Preprocessed {len(tokenized_comments)} comments")

    memory_info = psutil.virtual_memory()
    print(f"\nTotal RAM: {memory_info.total / (1024**3):.2f} GB")
    print(f"Used RAM: {memory_info.used / (1024**3):.2f} GB")
    print(f"Memory Usage: {memory_info.percent}%")

    model, training_time, cpu_usage = train_word2vec_cbow(tokenized_comments, vector_size=100)
    
    model.save("word2vec_cbow.model")
    print("Model saved as 'word2vec_cbow.model'")
    
    print(f"Vocabulary size: {len(model.wv)}")
    print(f"Vector size: {model.wv.vector_size}")

    print("")
    prepare_glove_corpus()

    glove_vectors = load_glove_vectors("/Users/ramonsaturnino/csc-446/assignment2/GloVe/vectors.txt")
    words = ["movie", "music", "woman", "christmas"]
    
    print("\nWord2Vec CBOW Results:")
    try:
        for word in words:
            print(f"\nMost similar words to {word}:")
            similar_words = model.wv.most_similar(word, topn=10)
            for word, similarity in similar_words:
                print(f"  {word}: {similarity:.4f}")
    except KeyError:
        print("Word 'movie' not in vocabulary, check your dataset content")

    print("\nGloVe Results:")
    for test_word in words:
        print(f"\nMost similar words to '{test_word}':")
        similar_words = find_most_similar(test_word, glove_vectors)
        for word, similarity in similar_words:
            print(f"  {word}: {similarity:.4f}")

    """
        Question 4) Train models with different vector sizes
    """
    # Train models with different vector sizes
    try:
        m1, time_m1, cpu_m1 = train_word2vec_cbow(tokenized_comments, vector_size=1)
        m2, time_m2, cpu_m2 = train_word2vec_cbow(tokenized_comments, vector_size=10)
        m3, time_m3, cpu_m3 = train_word2vec_cbow(tokenized_comments, vector_size=100)

        m1.save("word2vec_cbow_m1.model")
        m2.save("word2vec_cbow_m2.model")
        m3.save("word2vec_cbow_m3.model")

        print("\nModels trained finished!")
    except:
        print("\nCouldnt train m1, m2, and m3")

    # Generate labels (1 for positive, 0 for negative)
    labels = np.array([1] * 1000 + [0] * 1000)

    # Split into training (80%), validation (10%), and testing (10%)
    X_train, X_temp, y_train, y_temp = train_test_split(tokenized_comments, labels, test_size=0.2, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    print(f"Dataset Split: {len(X_train)} train, {len(X_val)} val, {len(X_test)} test")
    # Convert all datasets using each model
    X_train_m1 = convert_to_vectors(X_train, m1, 1)
    X_val_m1 = convert_to_vectors(X_val, m1, 1)
    X_test_m1 = convert_to_vectors(X_test, m1, 1)

    X_train_m2 = convert_to_vectors(X_train, m2, 10)
    X_val_m2 = convert_to_vectors(X_val, m2, 10)
    X_test_m2 = convert_to_vectors(X_test, m2, 10)

    X_train_m3 = convert_to_vectors(X_train, m3, 100)
    X_val_m3 = convert_to_vectors(X_val, m3, 100)
    X_test_m3 = convert_to_vectors(X_test, m3, 100)

    # Train models with different vector sizes
    model_m1 = train_model(X_train_m1, y_train, X_val_m1, y_val, vector_size=1)
    model_m2 = train_model(X_train_m2, y_train, X_val_m2, y_val, vector_size=10)
    model_m3 = train_model(X_train_m3, y_train, X_val_m3, y_val, vector_size=100)

    metrics_m1 = evaluate_model(model_m1, X_test_m1, y_test)
    metrics_m2 = evaluate_model(model_m2, X_test_m2, y_test)
    metrics_m3 = evaluate_model(model_m3, X_test_m3, y_test)

    print("\nEvaluation Results:")
    print(f"Vector Size 1: Accuracy={metrics_m1[0]:.4f}, Precision={metrics_m1[1]:.4f}, Recall={metrics_m1[2]:.4f}, F1-score={metrics_m1[3]:.4f}")
    print(f"Vector Size 10: Accuracy={metrics_m2[0]:.4f}, Precision={metrics_m2[1]:.4f}, Recall={metrics_m2[2]:.4f}, F1-score={metrics_m2[3]:.4f}")
    print(f"Vector Size 100: Accuracy={metrics_m3[0]:.4f}, Precision={metrics_m3[1]:.4f}, Recall={metrics_m3[2]:.4f}, F1-score={metrics_m3[3]:.4f}")


    """
        Question 2.1: C1 vs C2
    """
    # C1
    best_vector_size = 100
    X_train_best = X_train_m3
    X_val_best = X_val_m3
    X_test_best = X_test_m3

    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

    model_C1 = train_model(X_train_best, y_train, X_val_best, y_val, vector_size=best_vector_size)


    # C2
    X_train_random = generate_random_embeddings(X_train_best.shape)
    X_val_random = generate_random_embeddings(X_val_best.shape)
    X_test_random = generate_random_embeddings(X_test_best.shape)

    model_C2 = train_model(X_train_random, y_train, X_val_random, y_val, vector_size=best_vector_size)

    # Evaluate both models
    metrics_C1 = evaluate_model(model_C1, X_test_best, y_test)
    metrics_C2 = evaluate_model(model_C2, X_test_random, y_test)

    # Print results
    print("\nEvaluation Results:")
    print(f"C1 (Pre-trained Embeddings): Accuracy={metrics_C1[0]:.4f}, Precision={metrics_C1[1]:.4f}, Recall={metrics_C1[2]:.4f}, F1-score={metrics_C1[3]:.4f}")
    print(f"C2 (Random Embeddings): Accuracy={metrics_C2[0]:.4f}, Precision={metrics_C2[1]:.4f}, Recall={metrics_C2[2]:.4f}, F1-score={metrics_C2[3]:.4f}")