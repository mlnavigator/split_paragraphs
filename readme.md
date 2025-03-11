Модуль представляет собой способ семантической сегментации текстов на русском или ином языке, где токенизация по словам и слова превращаются в стем формы обрезанием первых 4-х символов.

В модуле реализовано 2 способа сегментации.
- на основе общих слов - можно использовать для английского и других языков
- на основе семантической близости, учитывающей как общие слова, так и семантику фрагментов на основе векторных эмбеддингов - только для русскоязычных текстов

Самый простой и быстрый способ делать сегментацию - на основе общих слов - Jaccar similarity.

- Алгоритм бьет текст на параграфы.
- затем маленькие параграфы присоединяются к соседним (если размер меньше n_min символов присоединяются к предыдущему или следующему, при условии что целевой фрагмент не более n_max)
- близость считается как доля общих слов (в стем форме) по мере Жаккара

```python3

from split_paragraphs import split_paragraphs_jaccar

text = """Lorem Ipsum is simply dummy text of the printing and typesetting industry.\n\nLorem Ipsum has been the industry's standard dummy text"""

parts = split_paragraphs_jaccar(text, n_min=10, n_max=200)

print(parts) 
>> ['Lorem Ipsum is simply dummy text of the printing and typesetting industry.',  "Lorem Ipsum has been the industry's standard dummy text"]
```

Можно делать более умную нарезку.
- Сначала бьем на параграфы
- Затем большие параграфы (более n_max) бьются по строкам
- Затем большие строки (более n_max) бьются по предложениям
- Затем полученные части объединяются по близости - маленькие части (менее n_min символов) присоединяются в соседним если они не более n_max
- Близость вычисляется как взвешенная на основе векторной и пословной (Жаккаровой) близости. Веса можно задавать

```python3

from split_paragraphs import split_rec

text = """Lorem Ipsum is simply dummy text of the printing and typesetting industry.\n\nLorem Ipsum has been the industry's standard dummy text"""

# def split_rec(text: str, n_min: int, n_max: int, cut_vect_n=500, cut_n_jaccar=1200, vect_weight=0.5, jaccar_weight=0.5) -> List[str]:
parts = split_rec(text, n_min=10, n_max=200)

print(parts) 
>> ['Lorem Ipsum is simply dummy text of the printing and typesetting industry.',  "Lorem Ipsum has been the industry's standard dummy text"]
```

- text - текст для сегментации
- n_min - длина меньше которой алгоритм постарается не создавать сегментов
- n_max - длина, больше которой алгоритм постарается не создавать сегментов
- в реальных примерах хорошо работают константы n_min = 1000, n_max = 6000 (не менее трети страницы, не более 2-х страниц. Обычно этого хватает чтобы разделить на смысловые блоки)
- начните с констант n_min = 1000, n_max = 6000 и далее подберите на нескольких примеров ваших текстов значения
- cut_vect_n - сколько символов брать для вычисления векторной близости. По умолчанию 500 символов конца предыдущего фрагмента и 500 начала следующего для вычисления векторной (семантической) близости
- cut_n_jaccar - сколько символов брать для вычисления пословной близости. По умолчанию 1200 символов конца предыдущего фрагмента и 1200 начала следующего для вычисления жаккаровой (пословной) близости
- vect_weight - вес векторной близости
- jaccar_weight - вес пословной близости, лучше ставить так, чтобы сумма jaccar_weight + vect_weight == 1
- можно сделать только пословную близость или только векторную, выставив коэффициенты 0 и 1

Векторная близость вычисляется на основе векторов navec [https://github.com/natasha/navec](https://github.com/natasha/navec)

Работает быстро на CPU, по памяти не требовательно.

В модуле есть возможность считать BERT эмбеддинги на основе маленького берта sergeyzh/LaBSE-ru-turbo [https://huggingface.co/sergeyzh/LaBSE-ru-turbo](https://huggingface.co/sergeyzh/LaBSE-ru-turbo)

Если дополнительно установить torch и transformers

```python3
from split_paragraphs.bert_similarity import model, tokenizer, get_bert_embedding

text = """Lorem Ipsum is simply dummy text of the printing and typesetting industry.\n\nLorem Ipsum has been the industry's standard dummy text"""

vect = get_bert_embedding(text)
print(len(vect))
>> 768

# Если есть видеокарта, то можно вычислять быстрее
model.to('cuda')
# после этого станет быстрее
vect = get_bert_embedding(text)
```
Но на CPU даже маленькие берты считаются достаточно долго и их не имеет смысла использовать при сегментации, 
т.к. векторов придется считать довольно много. 
поэтому векторная близость на бертах для сегментации в модуле не реализована
