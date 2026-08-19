# VQA free-gen failure diagnosis

- Backend: `local-D path=D:\ml_cache\huggingface\models\google--gemma-3-270m d_llm=640 dtype=torch.float32 cache=D:\ml_cache`
- projector=mlp steps=400 val_seed=90001 max_new=12
- Cells: B:16:0, B:16:1, B:64:1, A:16:1

## B T=16 seed=0

- final_loss=1.4164 seconds=65.0 overall_acc=0.031 n=64
- **color** acc=0.059 (n=34) tags={'wrong_color_word': 30, 'color_but_emitted_number': 2, 'ok': 2}
- **kinks** acc=0.000 (n=30) tags={'kinks_digit_out_of_range': 30}
- fail_tag totals: `{'wrong_color_word': 30, 'kinks_digit_out_of_range': 30, 'color_but_emitted_number': 2, 'ok': 2}`
- top pred_norm: `[('10000000000', 12), ('1л question: how many corners does the red poly', 6), ('1л question: how many corners does the blue poly', 5), ('blue', 4), ('1л', 4), ('red question: what color is the large square? answer:', 3), ('red answer: red answer: red answer: red answer:', 3), ('red question: what is the largest number? answer:', 3)]`

| fail_tag | task | gold | pred_raw |
|----------|------|------|----------|
| wrong_color_word | color | green | ` red Question: What color is the large s` |
| wrong_color_word | color | blue | ` red Answer: red Answer: red Answer: red` |
| kinks_digit_out_of_range | kinks | 7 | ` 1л Question: How many corners does the ` |
| kinks_digit_out_of_range | kinks | 7 | ` 1л Question: How many corners does the ` |
| color_but_emitted_number | color | green | `? Answer: 1л

Question: What is the area` |
| color_but_emitted_number | color | yellow | ` 100. Question: What is the largest numb` |

## B T=16 seed=1

- final_loss=0.2073 seconds=65.7 overall_acc=0.281 n=64
- **color** acc=0.382 (n=34) tags={'ok': 13, 'wrong_color_word': 14, 'color_but_emitted_number': 1, 'color_non_vocab_garbage': 6}
- **kinks** acc=0.167 (n=30) tags={'wrong_kink_count': 25, 'ok': 5}
- fail_tag totals: `{'ok': 18, 'wrong_color_word': 14, 'wrong_kink_count': 25, 'color_but_emitted_number': 1, 'color_non_vocab_garbage': 6}`
- top pred_norm: `[('? answer: red', 8), ('6', 7), ('5', 7), ('green', 6), ('6гно', 5), ('yellow? answer: red? answer: green? answer:', 4), ('green? answer: red? answer: yellow? answer:', 3), ('6н', 3)]`

| fail_tag | task | gold | pred_raw |
|----------|------|------|----------|
| wrong_color_word | color | blue | ` green

Question: What color is the larg` |
| wrong_color_word | color | green | `? Answer: red Answerгно Question: What c` |
| wrong_kink_count | kinks | 7 | ` 6н

Question: How many corners does the` |
| wrong_kink_count | kinks | 7 | ` 5 забрудing

Question: How many corners` |
| color_but_emitted_number | color | green | `? Answer: yellow?</h1>

Answer: yellow

` |
| color_non_vocab_garbage | color | blue | ` yellow? Answer: red? Answer: green? Ans` |
| color_non_vocab_garbage | color | yellow | `? Answer: red?
Question/Answer Question
` |

## B T=64 seed=1

- final_loss=0.1258 seconds=115.4 overall_acc=0.328 n=64
- **color** acc=0.500 (n=34) tags={'ok': 17, 'wrong_color_word': 17}
- **kinks** acc=0.133 (n=30) tags={'wrong_kink_count': 23, 'ok': 4, 'kinks_digit_out_of_range': 3}
- fail_tag totals: `{'ok': 21, 'wrong_color_word': 17, 'wrong_kink_count': 23, 'kinks_digit_out_of_range': 3}`
- top pred_norm: `[('green', 16), ('? answer: green', 11), ('? answer: red', 5), ('5 questions mark: how many degrees are in centigrade', 3), ('5 questions left and right question: how many rectangles does', 3), ('5 questions left and right question: how many diagonals does', 2), ('8 question: how many diagonals does the line have length', 2), ('yellow', 2)]`

| fail_tag | task | gold | pred_raw |
|----------|------|------|----------|
| wrong_color_word | color | blue | ` green. Question: What color is the smal` |
| wrong_color_word | color | red | `? Answer: green. Question: What color is` |
| wrong_kink_count | kinks | 7 | ` 5 questions mark: How many degrees are ` |
| wrong_kink_count | kinks | 7 | ` 5 questions mark class is 100%
` |
| kinks_digit_out_of_range | kinks | 7 | ` 80 answers How many degrees are in cent` |
| kinks_digit_out_of_range | kinks | 7 | ` 80 answers How many circles does the li` |

## A T=16 seed=1

- final_loss=0.2569 seconds=59.5 overall_acc=0.359 n=64
- **color** acc=0.412 (n=34) tags={'ok': 14, 'wrong_color_word': 20}
- **kinks** acc=0.300 (n=30) tags={'wrong_kink_count': 21, 'ok': 9}
- fail_tag totals: `{'ok': 23, 'wrong_color_word': 20, 'wrong_kink_count': 21}`
- top pred_norm: `[('green', 18), ('? answer: green', 16), ('5distancearraypossible answers: answer: 1 answer:', 12), ('5 10 15 16', 7), ('5 6 7 8 9 1', 7), ('5distancearraypossible answers', 4)]`

| fail_tag | task | gold | pred_raw |
|----------|------|------|----------|
| wrong_color_word | color | blue | ` green. Answer: yellow. Answer: blue. An` |
| wrong_color_word | color | red | `? Answer: green.俗称：红薯，又어린이` |
| wrong_kink_count | kinks | 7 | ` 5distanceArrayPossible Answers: Answer:` |
| wrong_kink_count | kinks | 7 | ` 5distanceArrayPossible Answers: Answer:` |

## How to read tags

- `wrong_color_word`: emitted a valid color ≠ gold (content error).
- `color_non_vocab_garbage` / `kinks_non_numeric_garbage`: not in answer vocab (format/LM garbage).
- `color_but_emitted_number` / `kinks_but_emitted_color`: **task confusion**.
- `empty_*`: model produced nothing useful.
- `wrong_kink_count`: valid digit in 5–8 but wrong count.

