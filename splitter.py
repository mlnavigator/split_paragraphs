from typing import List, Dict, Any
from copy import deepcopy
from functools import partial
import re

from .jaccar_splitter import split_text, calc_sim_texts_jaccar
# from .bert_similarity import calc_bert_similarity
from .navec_similarity import calc_navec_similarity
from .collect import aggregate_parts_sim


def calc_sim_texts(text1: str, text2: str, cut_vect_n=500, cut_n_jaccar=1200, vect_weight=0.5, jaccar_weight=0.5) -> float:
    """
    Calculate similarity between two sequential texts using Vector and Jaccar similarity
    text1 - first text
    text2 - second text
    cut_vect_n - number of characters in texts to cut for vectorization
    vect_weight - weight of vect similarity
    jaccar_weight - weight of Jaccar similarity

    return - similarity between two texts with formula:
    (bert_weight * vect_sim + jaccar_weight * jaccar_sim) / (vect_weight + jaccar_weight)
    """
    # print('text1: ', text1[:cut_bert_n], '\n\n', 'text2: ', text2[:cut_bert_n])
    vect_sim = calc_navec_similarity(text1[-cut_vect_n:], text2[:cut_vect_n])
    jaccar_sim = calc_sim_texts_jaccar(text1[-cut_n_jaccar:], text2[:cut_n_jaccar])
    return (vect_weight * vect_sim + jaccar_weight * jaccar_sim) / (vect_weight + jaccar_weight)


def split_paragraphs(text: str, n_min: int, n_max: int, sep='\n\n', cut_vect_n=300, cut_n_jaccar=1200, vect_weight=0.6, jaccar_weight=0.4) -> List[str]:
    """
    text - str
    n_min: int - min len in letters.
    n_max: int - max len in letters.
    cut_bert_n: int - len of text to cut for calculating bert embedding
    bert_weight: float - weight of BERT similarity
    jaccar_weight: float - weight of Jaccar similarity

    return: list of paragraphs

    all paragraphs shorter than n letters will be joined with previous or next paragraphs depending on common words similarity
    """
    parts = split_text(text, sep=sep)
    calc_sim = partial(calc_sim_texts, cut_vect_n=cut_vect_n, cut_n_jaccar=cut_n_jaccar,
                       vect_weight=vect_weight, jaccar_weight=jaccar_weight)
    parts = aggregate_parts_sim(parts, n_min=n_min, n_max=n_max, sep=sep, calc_sim=calc_sim)
    return parts


def construct_series(text: str, n_max: int) -> List[Dict[str, Any]]:
    series = []
    parts1 = split_text(text, sep='\n\s*\n')
    s = {'sep': '\n\n', 'parts': []}
    for p1 in parts1:
        if len(p1) <= n_max:
            s['parts'].append(p1)
        else:
            series.append(s)
            parts2 = split_text(p1, sep='\n')
            s = {'sep': '\n', 'parts': []}
            for p2 in parts2:
                if len(p2) <= n_max:
                    s['parts'].append(p2)
                else:
                    series.append(s)
                    sentence_pattern = r'[а-яёa-z]{2,}\s*[\.!?](\s+)[А-ЯЁA-Z]+\w{2,}'
                    p2_ = re.sub(sentence_pattern, lambda x: x.group(0).replace(x.group(1), '<sep>'), p2)
                    parts3 = p2_.split('<sep>')
                    parts3 = [p3.strip() for p3 in parts3 if len(p3.strip()) > 0]
                    s = {'sep': ' ', 'parts': parts3}
                    series.append(s)
                    s = {'sep': '\n', 'parts': []}
            series.append(s)
            s = {'sep': '\n\n', 'parts': []}
    series.append(s)

    series = [s for s in series if len(s['parts']) > 0]
    return series


def split_rec(text: str, n_min: int, n_max: int, cut_vect_n=500, cut_n_jaccar=1200, vect_weight=0.5, jaccar_weight=0.5) -> List[str]:
    """
    text - str
    n_min: int - min len in letters.
    n_max: int - max len in letters.
    cut_vect_n: int - len of text to cut for calculating vect embedding
    vect_weight: float - weight of vect similarity
    jaccar_weight: float - weight of Jaccar similarity

    return: list of parts

    All parts shorter than n letters will be joined with previous or next paragraphs depending on common words similarity if possible.
    If part is longer than n_max, it will be splitted to several parts if possible.
    Order of splitting: paragraphs - '\n\n', lines - '\n', sentences - '[\.!?](\s*)
    """

    calc_sim = partial(calc_sim_texts, cut_vect_n=cut_vect_n, cut_n_jaccar=cut_n_jaccar,
                       vect_weight=vect_weight, jaccar_weight=jaccar_weight)
    series = construct_series(text, n_max)
    parts = []
    for s in series:
        aggregated = aggregate_parts_sim(s['parts'], n_min=n_min, n_max=n_max, sep=s['sep'], calc_sim=calc_sim)
        parts.extend(aggregated)
    return parts
