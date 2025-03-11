import re
from typing import List
from functools import partial
from .collect import aggregate_parts_sim


def split_text(text: str, sep: str=r'\n\s*\n') -> List[str]:
    text = text.strip()
    text = re.sub(sep, '<sep>', text)
    parts = text.split('<sep>')
    parts = [p.strip() for p in parts if len(p.strip()) > 0]
    return parts


def jaccar_tokenize(text: str) -> List[str]:
    toks = re.findall('\w{4,}', text.lower())
    return [t[:4] for t in toks]


def calc_sim_texts_jaccar(text1: str, text2: str) -> float:
    s1 = set(jaccar_tokenize(text1))
    s2 = set(jaccar_tokenize(text2))
    if len(s1) == 0 or len(s2) == 0:
        return 0
    return len(s1.intersection(s2)) / len(s1.union(s2))


aggregate_parts_sim_jaccar = partial(aggregate_parts_sim, calc_sim=calc_sim_texts_jaccar)


def split_paragraphs_jaccar(text: str, n_min: int, n_max: int, sep: str=r'\n\s*\n') -> List[str]:
    """
    text - str
    n_min: int - len in letters.
    max_len: int - max len of paragraph
    all paragraphs shorter than n letters will be joined with previous or next paragraphs depending on common words similarity
    """
    parts = split_text(text, sep=sep)
    parts = aggregate_parts_sim_jaccar(parts, n_min=n_min, n_max=n_max, sep='\n\n', calc_sim=calc_sim_texts_jaccar)
    return parts
