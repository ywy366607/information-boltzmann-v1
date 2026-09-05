from types import SimpleNamespace

import torch

from scripts import train_real256_capacity as runner
from test_sharegpt4o_data import TinyTokenizer


def test_graph_decode_keeps_observed_question_and_never_passes_gold(monkeypatch):
    class Tokenizer(TinyTokenizer):
        def decode(self, ids, **kwargs):
            return "stub"

    tokenizer = Tokenizer()
    sample = dict(image=torch.zeros(3, 8, 8), target_rgb=torch.zeros(3, 8, 8),
                  image_precision=1.0, target_image_precision=0.0, target_text_precision=1.0,
                  need_pix=False, need_text=True, need_seg=False, task="it2t", id="qa",
                  prompt="Is a cat visible?", answer="Yes.", text_missing=False,
                  task_id=1, target_time=0.0, target_seg_precision=0.0,
                  history_images=torch.zeros(2, 3, 8, 8), history_precision=torch.zeros(2),
                  target_seg=torch.zeros(8, 8, dtype=torch.long))
    calls = []

    def forward(model, batch):
        assert batch["labels"] is None
        calls.append({k: batch[k].clone() for k in ("input_ids", "text_precision", "visual_prompt_mask")})
        logits = torch.zeros(1, batch["input_ids"].shape[1], 40)
        logits[..., 7 if len(calls) % 2 else tokenizer.eos_token_id] = 1
        return {"token_logits": logits, "n_vis_tokens": 0}

    monkeypatch.setattr(runner, "forward_real_capacity", forward)
    model = SimpleNamespace(lm_tok=tokenizer)
    runner.decode_caption(model, sample, "cpu", max_tokens=2)
    prefix = tokenizer.encode(sample["prompt"])
    assert calls[0]["input_ids"].tolist() == [prefix]
    assert calls[1]["input_ids"].tolist() == [prefix + [7]]
    assert calls[1]["visual_prompt_mask"].tolist() == [[True] * len(prefix) + [False]]
    assert calls[1]["text_precision"].eq(1).all()
    runner.decode_caption(model, {**sample, "answer": "No."}, "cpu", max_tokens=2)
    torch.testing.assert_close(calls[0]["input_ids"], calls[2]["input_ids"])
    runner.decode_caption(model, {**sample, "text_missing": True}, "cpu", max_tokens=2)
    assert calls[4]["input_ids"].tolist() == [[tokenizer.bos_token_id]]
    assert calls[4]["text_precision"].tolist() == [[0.0]]
    assert not calls[4]["visual_prompt_mask"].any()
