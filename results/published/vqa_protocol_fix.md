# VQA protocol fix: answer-only loss + constrained ranking

- Backend: `local-D path=D:\ml_cache\huggingface\models\google--gemma-3-270m d_llm=640 dtype=torch.float32 cache=D:\ml_cache`
- projector=mlp steps=400 val_seed=90001
- **full_text**: CE on entire `Question:… Answer:…` (old)
- **answer_only**: CE only on answer token span
- **free**: greedy free-gen exact-match (early-stop heuristics)
- **cons**: constrained ranking over {red,green,blue,yellow} / {5,6,7,8}

| kind | T | seed | train | free_acc | free_color | free_kinks | cons_acc | cons_color | cons_kinks | loss |
|------|---|------|-------|----------|------------|------------|----------|------------|------------|------|
| B | 16 | 0 | full_text | 0.031 | 0.059 | 0.000 | 0.125 | 0.147 | 0.100 | 1.416 |
| B | 16 | 0 | answer_only | 0.062 | 0.059 | 0.067 | 0.203 | 0.294 | 0.100 | 0.953 |
| B | 16 | 1 | full_text | 0.156 | 0.147 | 0.167 | 0.312 | 0.441 | 0.167 | 0.207 |
| B | 16 | 1 | answer_only | 0.109 | 0.118 | 0.100 | 0.141 | 0.118 | 0.167 | 1.224 |
| B | 64 | 1 | full_text | 0.250 | 0.382 | 0.100 | 0.328 | 0.500 | 0.133 | 0.126 |
| B | 64 | 1 | answer_only | 0.141 | 0.176 | 0.100 | 0.266 | 0.412 | 0.100 | 0.980 |
| A | 16 | 1 | full_text | 0.297 | 0.412 | 0.167 | 0.266 | 0.412 | 0.100 | 0.257 |
| A | 16 | 1 | answer_only | 0.328 | 0.412 | 0.233 | 0.328 | 0.412 | 0.233 | 1.349 |

## Takeaways

- **B@T16 s0**: free 0.031→0.062 (color 0.059→0.059); cons 0.125→0.203 (color 0.147→0.294)
- **B@T16 s1**: free 0.156→0.109 (color 0.147→0.118); cons 0.312→0.141 (color 0.441→0.118)
- **B@T64 s1**: free 0.250→0.141 (color 0.382→0.176); cons 0.328→0.266 (color 0.500→0.412)
- **A@T16 s1**: free 0.297→0.328 (color 0.412→0.412); cons 0.266→0.328 (color 0.412→0.412)
- B@T16 s0 full_text: cons−free overall = +0.094
- B@T16 s0 answer_only: cons−free overall = +0.141
- B@T16 s1 full_text: cons−free overall = +0.156
- B@T64 s1 full_text: cons−free overall = +0.078
- B@T64 s1 answer_only: cons−free overall = +0.125

