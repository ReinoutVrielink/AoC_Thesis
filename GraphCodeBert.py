import json
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel # https://huggingface.co/microsoft/graphcodebert-base/tree/main?library=transformers
 
tokenizer = AutoTokenizer.from_pretrained("microsoft/graphcodebert-base")
model = AutoModel.from_pretrained("microsoft/graphcodebert-base")
 
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval() 
 
def get_embedding(code):
    # generating embeddings on a solution
    inputs = tokenizer(code, return_tensors="pt", max_length=512, truncation=True, padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()[0]
    return embedding
 
 
def generate_all_embeddings(input_file, output_file):
    # generating all embeddings + saving them to a file
    print(f"\nLoading solutions from {input_file}...")
    with open(input_file, 'r') as f:
        solutions = json.load(f)
    print(f"Generating embeddings for {len(solutions)} solutions...")
    embeddings_data = []
    for sol in solutions:
        embedding = get_embedding(sol['code'])
        embeddings_data.append({
            "sol_id": sol['sol_id'],
            "author": sol['author'],
            "year": sol['year'],
            "day": sol['day'],
            "part": sol['part'],
            "embedding": embedding.tolist()
        })
    print(f"\nSaving {len(embeddings_data)} embeddings to {output_file}")
    with open(output_file, 'w') as f:
        json.dump(embeddings_data, f)
    return embeddings_data
 
 
def load_embeddings(filepath):
    # loading the embeddings
    with open(filepath, 'r') as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    df['embedding'] = list(np.stack(df['embedding'].values))
    return df
 