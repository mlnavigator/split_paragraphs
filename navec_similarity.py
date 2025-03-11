import numpy as np
import os
import re

from navec import Navec

path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'navec_hudlit_v1_12B_500K_300d_100q.tar')
navec = Navec.load(path)


def tokenize1(word: str) -> str:
    w = str(word).lower().strip().split()[0]
    w = re.sub(r'[^а-яё]', '', w)
    return w


def get_v(word):
    w = tokenize1(word)
    if len(w) == 0:
        return None
    for i in range(len(w), 0, -1):
        t = w[:i]
        v = navec.get(t)
        if v is not None:
            # print(word, '  --->   ', t)
            return v / np.linalg.norm(v)
    # print(word, '  --->   ', 'None')
    return None


def get_navec_embedding(text: str) -> np.array:
    text = str(text).lower().strip()
    tokens = text.split()
    tokens = [tokenize1(w) for w in tokens]
    tokens = [w for w in tokens if len(w) > 3]
    vects = [get_v(w) for w in tokens]
    vects = [v for v in vects if v is not None]
    if len(vects) == 0:
        return None
    mv = sum(vects)
    mv /=np.linalg.norm(mv)
    return mv


def calc_navec_similarity(text1: str, text2: str) -> float:
    """
    Calculate Navec similarity between two texts
    :param text1: first text
    :param text2: second text
    :return: similarity between two texts
    """
    emb1 = get_navec_embedding(text1)
    emb2 = get_navec_embedding(text2)
    if emb1 is None or emb2 is None:
        return 0
    return sum(emb1 * emb2)


def calc_similarity_navec_v(vect1: np.array, vect2: np.array):
    """
    Calculate similarity between two vectors
    :param vect1: first vector with norm 1
    :param vect2: second vector with norm 1
    :return: similarity between two vectors
    """
    return sum(vect1 * vect2)
