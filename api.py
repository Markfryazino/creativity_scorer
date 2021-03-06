import flask
from flask import Flask, jsonify, request
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel, GPT2LMHeadModel
from typing import Dict
import numpy as np
import scipy.spatial.distance
import json
import shutil
import traceback
import wandb


def perform_nli(premise, hypothesis):
    with torch.inference_mode():
        out = nli_model(**nli_tokenizer(premise, hypothesis, return_tensors='pt'))
        proba = torch.softmax(out.logits, -1).cpu().numpy()[0]
    return {v: proba[k] for k, v in nli_model.config.id2label.items()}

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask

def get_embedding(sentences):
    encoded_input = sbert_tokenizer(sentences, padding=True, truncation=False, return_tensors='pt')

    with torch.inference_mode():
        model_output = sbert_model(**encoded_input)

    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
    return sentence_embeddings

def estimate_perplexity(text):
    inputs = gpt_tokenizer(text, return_tensors="pt")
    with torch.inference_mode():
        loss = gpt_model(**inputs, labels=inputs["input_ids"]).loss
    return loss.item()

def get_mean_dist(text, embeds, k=10):
    embed = get_embedding(text).numpy()
    dists = scipy.spatial.distance.cdist(embed, embeds, metric="cosine")[0]
    return np.partition(dists, k)[:k].mean()

def score(premise: str, hypothesis: str, test: Dict[str, np.array], top_k=10, perplexity_threshold=6., beta=1.):
    nli_scores = perform_nli(premise, hypothesis)
    perplexity_score = estimate_perplexity(hypothesis)
    
    if premise not in test:
        raise Exception("There are no ready embeddings for this hypothesis!")
    else:
        distance_score = get_mean_dist(hypothesis, test[premise])
    
    divisor = max(1., np.exp(perplexity_score - perplexity_threshold))
    nli_quality = 1 - nli_scores["contradiction"]
    
    f_score = (1 + beta ** 2) * nli_quality * distance_score / (beta ** 2 * nli_quality + distance_score)
    return {
        "nli_quality": nli_quality,
        "distance_score": distance_score,
        "perplexity_divisor": divisor,
        "final": f_score / divisor
    }


app = Flask(__name__)

with open("data/embeds_fin.json") as f:
    hypothesis_files = json.load(f)
embeddings = {text[22:]: torch.load(file).numpy() for text, file in hypothesis_files.items()}

wandb.init(project="creativity_scorer", name="api-run")

try:
    shutil.rmtree("/home/broccoliman/.cache/huggingface")
except:
    pass
nli_tokenizer = AutoTokenizer.from_pretrained('cointegrated/rubert-base-cased-nli-threeway')
nli_model = AutoModelForSequenceClassification.from_pretrained('cointegrated/rubert-base-cased-nli-threeway')
shutil.rmtree("/home/broccoliman/.cache/huggingface")

sbert_tokenizer = AutoTokenizer.from_pretrained("sberbank-ai/sbert_large_nlu_ru")
sbert_model = AutoModel.from_pretrained("sberbank-ai/sbert_large_nlu_ru")
shutil.rmtree("/home/broccoliman/.cache/huggingface")

gpt_tokenizer = AutoTokenizer.from_pretrained('sberbank-ai/rugpt3small_based_on_gpt2')
gpt_model = GPT2LMHeadModel.from_pretrained('sberbank-ai/rugpt3small_based_on_gpt2')
shutil.rmtree("/home/broccoliman/.cache/huggingface")

print("MODELS READY")

count = 0

user_id = 4200000

@app.route('/id', methods=['POST'])
def get_id():
    global user_id
    user_id += 1
    print(f"User {user_id} has been created!")
    return user_id

@app.route('/predict', methods=['POST'])
def predict():
    global count
    try:
        params = request.get_json()
        print(f"Got request!")
        print(params)
        res = score(params["premise"], params["hypothesis"], embeddings)
        print(f"Result:")
        print(res)

        count += 1
        wandb.log({"count": count, **params, **res})

        return res
    except:
        return {"error": traceback.format_exc()}

@app.route('/save', methods=['POST'])
def save():
    try:
        params = request.get_json()
        print(f"Saving parameters...")
        print(params)
        wandb.log(params})

        return {"success": "success"}
    except:
        return {"error": traceback.format_exc()}

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
