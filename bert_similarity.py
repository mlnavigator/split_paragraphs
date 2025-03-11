import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np

tokenizer = AutoTokenizer.from_pretrained("sergeyzh/LaBSE-ru-turbo")
model = AutoModel.from_pretrained("sergeyzh/LaBSE-ru-turbo")
model.eval()
model.to('cpu')


def get_bert_embedding(text: str) -> np.array:
    """
    Get BERT embedding for a text
    :param text: text to embed
    :return: np.array embedding with norm 1
    """
    global model
    global tokenizer
    text_tokenized = tokenizer(text, return_tensors='pt', truncation=True)
    for k,v in text_tokenized.items():
        text_tokenized[k] = v.to(model.device)
    with torch.no_grad():
        model_output = model(**text_tokenized)
    embs = model_output['last_hidden_state']
    emb = embs[0][0]
    emb = emb.to('cpu')
    emb = emb.numpy()
    emb = emb / np.linalg.norm(emb)
    return emb


def calc_bert_similarity(text1: str, text2: str) -> float:
    """
    Calculate BERT similarity between two texts
    :param text1: first text
    :param text2: second text
    :return: similarity between two texts
    """
    emb1 = get_bert_embedding(text1)
    emb2 = get_bert_embedding(text2)
    return sum(emb1 * emb2)


def calc_similarity_bert_v(vect1: np.array, vect2: np.array):
    """
    Calculate similarity between two vectors
    :param vect1: first vector with norm 1
    :param vect2: second vector with norm 1
    :return: similarity between two vectors
    """
    return sum(vect1 * vect2)


