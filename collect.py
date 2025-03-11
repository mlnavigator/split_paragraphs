from typing import List
from copy import deepcopy


def aggregate_parts_sim(parts: List[str], n_min: int, n_max: int, sep='\n\n',
                        calc_sim: callable = None,
                        ) -> List[str]:
    """
    All parts shorter than n letters are aggregated to another paragraph.
    parts - paragraphs
    n - len in letters.
    calc_sim - function for calculating similarity between two texts calc_sim(text1: str, text2: str) -> float
    """
    if calc_sim is None:
        raise ValueError('calc_sim argument is not defined')

    if len(parts) <= 1:
        return parts

    new_parts = deepcopy(parts)

    while len(new_parts[-1]) <= n_min and len(new_parts) > 1 and len(new_parts[-1]) + len(new_parts[-2]) <= n_max:
        p = new_parts.pop()
        new_parts[-1] += sep + p

    while len(new_parts[0]) <= n_min and len(new_parts) > 1 and len(new_parts[0]) + len(new_parts[1]) <= n_max:
        p = new_parts.pop(0)
        new_parts[0] = p + sep + new_parts[0]

    i = 1
    while i < len(new_parts) - 1:
        if len(new_parts[i]) >= n_min:
            i += 1
            continue
        # print(i, len(new_parts))
        prev = i - 1
        next_ = i + 1

        t_prev = new_parts[prev][-n_min:]
        t = new_parts[i]
        t_next = new_parts[next_][:n_min]

        s_prev = calc_sim(t_prev, t)
        s_next = calc_sim(t, t_next)

        if s_prev >= s_next:
            if len(t_prev) <= n_max:
                p = new_parts.pop(i)
                new_parts[prev] += sep + p
            else:
                i += 1
        else:
            if len(t_next) <= n_max:
                p = new_parts.pop(i)
                new_parts[i] = p + sep + new_parts[i]
            else:
                i += 1

    return new_parts
