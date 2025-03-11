from .jaccar_splitter import split_text, jaccar_tokenize, calc_sim_texts_jaccar, aggregate_parts_sim_jaccar, split_paragraphs_jaccar
from .splitter import construct_series


def test_split_text():
    # Тест на разделение текста на части
    text = "First paragraph.\n\nSecond paragraph.\n\n\nThird paragraph."
    expected = ["First paragraph.", "Second paragraph.", "Third paragraph."]
    assert split_text(text) == expected

    # Тест на пустой текст
    assert split_text("") == []

    # Тест на текст без разделителей
    assert split_text("Single paragraph.") == ["Single paragraph."]


def test_tokenize():
    # Тест на токенизацию текста
    text = "This is a test string with some words."
    expected = ["this", "test", "stri", "with", "some", "word"]
    assert jaccar_tokenize(text) == expected

    # Тест на текст с короткими словами
    assert jaccar_tokenize("a an the") == []

    # Тест на пустой текст
    assert jaccar_tokenize("") == []


def test_calc_sim_texts():
    # Тест на расчет схожести текстов
    text1 = "This is a test string."
    text2 = "This is another test string."
    # print(tokenize(text1))
    # print(tokenize(text2))
    assert calc_sim_texts_jaccar(text1, text2) == 0.75

    # Тест на полностью одинаковые тексты
    assert calc_sim_texts_jaccar(text1, text1) == 1.0

    # Тест на полностью разные тексты
    assert calc_sim_texts_jaccar(text1, "Completely different text.") == 0.0


def test_aggregate_parts_sim():
    # Тест на агрегацию частей текста
    parts = ["Short.", "This is a longer paragraph.", "Another short one.", "Yet another paragraph."]
    expected = ["Short.\n\nThis is a longer paragraph.", "Another short one.\n\nYet another paragraph."]
    assert aggregate_parts_sim_jaccar(parts, 20, 100) == expected

    # Тест на все части короче n
    parts = ["Short.", "Very short.", "Tiny."]
    expected = ["Short.\n\nVery short.\n\nTiny."]
    assert aggregate_parts_sim_jaccar(parts, 20, 100) == expected

    # Тест на все части длиннее n
    parts = ["This is a very long paragraph.", "Another long paragraph."]
    assert aggregate_parts_sim_jaccar(parts, 20, 100) == parts


def test_split_paragraphs():
    # Тест на разделение текста на параграфы
    text = "Short.\n\nThis is a longer paragraph.\n\nAnother short one.\n\nYet another paragraph."
    expected = ["Short.\n\nThis is a longer paragraph.", "Another short one.\n\nYet another paragraph."]
    # expected = ['Short.\n\nThis is a longer paragraph.', 'Another short one.', 'Yet another paragraph.']
    assert split_paragraphs_jaccar(text, 20, 100) == expected

    # Тест на пустой текст
    assert split_paragraphs_jaccar("", 10, 100) == []

    # Тест на текст без разделителей
    assert split_paragraphs_jaccar("Single paragraph.", 20, 100) == ["Single paragraph."]


def test_construct_series():
    text1 = "Привет мир"
    assert construct_series(text1, 10) == [{'sep': '\n\n', 'parts': ['Привет мир']}]

    # Тест 2: Текст с несколькими частями, разделенными двойным переводом строки
    text2 = "Первая часть\n\nВторая часть"
    assert construct_series(text2, 10) == [
        {'sep': ' ', 'parts': ['Первая часть']},
        {'sep': ' ', 'parts': ['Вторая часть']}
    ]

    assert construct_series(text2, 20) == [
        {'sep': '\n\n', 'parts': ['Первая часть', 'Вторая часть']}
    ]

    # Тест 3: Текст с длинной частью, которая разбивается по одинарному переводу строки
    text3 = "Очень длинная часть текста, которая не помещается в лимит\nВторая часть"
    assert construct_series(text3, 20) == [
        {'sep': ' ', 'parts': ['Очень длинная часть текста, которая не помещается в лимит']},
        {'sep': '\n', 'parts': ['Вторая часть']}
        ]

    assert construct_series(text3, 10) == [
        {'sep': ' ', 'parts': ['Очень длинная часть текста, которая не помещается в лимит']},
        {'sep': ' ', 'parts': ['Вторая часть']}
        ]

    # Тест 4: Текст с предложением, которое разбивается по предложениям
    text4 = "Это первое предложение. Это второе предложение! Или третье."
    assert construct_series(text4, 10) == [{'sep': ' ',
                                           'parts': ['Это первое предложение.',
                                                     'Это второе предложение!',
                                                     'Или третье.'
                                                     ]
                                           }
                                          ]

    assert construct_series(text4, 100) == [{'sep': '\n\n',
                                            'parts': ['Это первое предложение. Это второе предложение! Или третье.']}]

    # Тест 5: Пустой текст
    text5 = ""
    assert construct_series(text5, 10) == []

    # Тест 6: Текст с пробелами и пустыми строками
    text6 = "\n\n   \n\nТолько одна часть\n\n   "
    assert construct_series(text6, 100) == [{'sep': '\n\n', 'parts': ['Только одна часть']}]

    assert construct_series(text6, 10) == [{'sep': ' ', 'parts': ['Только одна часть']}]

    # Тест 7: Текст, где все части превышают лимит
    text7 = "Слишкомдлиннаястрокабезпробелов"
    assert construct_series(text7, 10) == [{'sep': ' ', 'parts': ['Слишкомдлиннаястрокабезпробелов']}]

def main():
    test_split_text()
    test_tokenize()
    test_calc_sim_texts()
    test_aggregate_parts_sim()
    test_split_paragraphs()
    test_construct_series()
    print("All tests passed.")

if __name__ == "__main__":
    main()
