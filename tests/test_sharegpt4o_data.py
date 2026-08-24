from __future__ import annotations

import io
import json
import tarfile
import zipfile

import torch
from PIL import Image

from fine_grain.sharegpt4o_data import (
    collate_real_multimodal,
    counterfactual_real_samples,
    extract_referenced_tar,
    letterbox_tensor,
    load_freedom_manifest,
    load_opengv_manifest,
    materialize_record,
    fetch_zip_member,
    read_zip_directory,
)


class TinyTokenizer:
    bos_token_id = 1
    eos_token_id = 2
    pad_token_id = 0

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return [3 + (ord(char) % 31) for char in text]


def _png_bytes(color=(255, 0, 0), size=(6, 4)) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", size, color).save(buffer, format="PNG")
    return buffer.getvalue()


def test_freedom_manifests_normalize_and_filter(tmp_path):
    t2i_path = tmp_path / "t2i.json"
    t2i_path.write_text(json.dumps([{
        "input_prompt": "draw a bird",
        "output_image": "image/1.png",
        "output_image_resolution": [1024, 1024],
    }]), encoding="utf-8")
    edit_path = tmp_path / "edit.json"
    edit_path.write_text(json.dumps([
        {
            "input_prompt": "remove the fence",
            "input_image": ["image/a.png"],
            "input_image_resolution": [1024, 1024],
            "output_image": "image/b.png",
            "output_image_resolution": [1024, 1024],
        },
        {
            "input_prompt": "change aspect ratio",
            "input_image": ["image/c.png"],
            "input_image_resolution": [1536, 1024],
            "output_image": "image/d.png",
            "output_image_resolution": [1024, 1024],
        },
    ]), encoding="utf-8")

    t2i = load_freedom_manifest(t2i_path, "t2i")
    edit = load_freedom_manifest(edit_path, "it2i", same_resolution=True)
    assert t2i[0]["task"] == "t2i"
    assert t2i[0]["target_member"] == "image/1.png"
    assert len(edit) == 1
    assert edit[0]["source_members"] == ["image/a.png"]


def test_opengv_first_turn_and_image_only_boundary(tmp_path):
    path = tmp_path / "gpt-4o.jsonl"
    rows = [
        {
            "image": "images/a.jpg",
            "conversations": [
                {"from": "human", "value": "<image> Describe this image."},
                {"from": "gpt", "value": "A red bird."},
            ],
        },
        {
            "images": ["images/b.jpg"],
            "messages": [
                {"role": "user", "content": "<image> What color is the car?"},
                {"role": "assistant", "content": "Blue."},
            ],
        },
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    parsed = load_opengv_manifest(path)
    assert [row["task"] for row in parsed] == ["i2t", "it2t"]
    assert parsed[0]["prompt"] == ""
    assert parsed[1]["prompt"] == "What color is the car?"


def test_opengv_caption_paraphrases_do_not_become_it2t(tmp_path):
    path = tmp_path / "paraphrases.jsonl"
    prompts = [
        "Please explain in detail the scene depicted in the picture.",
        "Can you describe all the objects and characters in the picture?",
        "What is compelling about this image?",
        "What color is the car in the image?",
        "How many people are visible?",
    ]
    rows = [
        {
            "image": f"images/{index}.jpg",
            "conversations": [
                {"from": "human", "value": f"<image> {prompt}"},
                {"from": "gpt", "value": "answer"},
            ],
        }
        for index, prompt in enumerate(prompts)
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    parsed = load_opengv_manifest(path)
    assert [row["task"] for row in parsed] == [
        "i2t", "i2t", "i2t", "it2t", "it2t",
    ]
    assert all(not row["prompt"] for row in parsed[:3])

    balanced = load_opengv_manifest(path, max_per_task=2)
    assert [row["task"] for row in balanced] == ["i2t", "i2t", "it2t", "it2t"]


def test_tar_extract_materialize_and_letterbox(tmp_path):
    archive = tmp_path / "pilot.tar"
    payloads = {
        "image/source.png": _png_bytes((255, 0, 0)),
        "image/target.png": _png_bytes((0, 255, 0)),
    }
    with tarfile.open(archive, "w") as handle:
        for name, payload in payloads.items():
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            handle.addfile(info, io.BytesIO(payload))
    records = [{
        "id": "edit-0",
        "dataset": "test",
        "task": "it2i",
        "prompt": "make it green",
        "answer": "",
        "source_members": ["image/source.png"],
        "target_member": "image/target.png",
    }]
    extracted = extract_referenced_tar(archive, records, tmp_path / "images", max_complete=1)
    assert len(extracted) == 1
    sample = materialize_record(extracted[0], 8)
    assert sample["image"].shape == sample["target_rgb"].shape == (3, 8, 8)
    assert not torch.allclose(sample["image"], sample["target_rgb"])
    tensor = letterbox_tensor(extracted[0]["source_files"][0], 8)
    assert torch.equal(tensor[:, 0], torch.zeros_like(tensor[:, 0]))


def test_remote_zip_range_directory_and_member(tmp_path):
    archive = tmp_path / "images.zip"
    expected = _png_bytes((0, 0, 255))
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as handle:
        handle.writestr("images/blue.png", expected)
        handle.writestr("images/note.txt", b"hello")
    payload = archive.read_bytes()

    def fetch(start, stop):
        return payload[start:stop]

    directory = read_zip_directory(fetch, len(payload))
    assert set(directory) == {"images/blue.png", "images/note.txt"}
    assert fetch_zip_member(fetch, directory["images/blue.png"]) == expected


def test_real_collator_masks_answer_from_visual_queries():
    base = {
        "id": "row",
        "image": torch.zeros(3, 8, 8),
        "target_rgb": torch.ones(3, 8, 8),
        "image_precision": 1.0,
        "target_image_precision": 0.0,
        "target_text_precision": 1.0,
        "need_pix": False,
        "need_text": True,
        "answer": "bird",
    }
    i2t = {**base, "task": "i2t", "prompt": "", "text_missing": True}
    it2t = {
        **base, "id": "row-2", "task": "it2t", "prompt": "What is shown?",
        "text_missing": False,
    }
    batch = collate_real_multimodal(TinyTokenizer(), [i2t, it2t])
    supervised = batch["labels"].ne(-100)
    assert not bool((batch["visual_prompt_mask"] & supervised).any())
    assert batch["text_precision"][0, 0] == 0
    assert not batch["visual_prompt_mask"][0, 0]
    assert bool(batch["visual_prompt_mask"][1].any())


def test_real_collator_reserves_answer_supervision_after_long_prompt():
    sample = {
        "id": "long", "task": "it2t", "prompt": "p" * 400,
        "answer": "a" * 100, "text_missing": False,
        "image": torch.zeros(3, 8, 8), "target_rgb": torch.zeros(3, 8, 8),
        "image_precision": 1.0, "target_image_precision": 0.0,
        "target_text_precision": 1.0, "need_pix": False, "need_text": True,
    }
    batch = collate_real_multimodal(
        TinyTokenizer(), [sample], max_text_tokens=128, min_answer_tokens=32,
    )
    assert int(batch["labels"].ne(-100).sum()) == 32
    assert int(batch["visual_prompt_mask"].sum()) == 96

    capped = collate_real_multimodal(
        TinyTokenizer(), [sample], max_text_tokens=128,
        min_answer_tokens=32, max_answer_tokens=16,
    )
    assert int(capped["labels"].ne(-100).sum()) == 16


def test_real_collator_can_supervise_answer_termination():
    sample = {
        "id": "eos", "task": "i2t", "prompt": "", "answer": "short",
        "text_missing": True, "image": torch.zeros(3, 8, 8),
        "target_rgb": torch.zeros(3, 8, 8), "image_precision": 1.0,
        "target_image_precision": 0.0, "target_text_precision": 1.0,
        "need_pix": False, "need_text": True,
    }
    tokenizer = TinyTokenizer()
    batch = collate_real_multimodal(
        tokenizer, [sample], max_answer_tokens=4, append_eos=True,
    )
    supervised = batch["labels"][0]
    supervised = supervised[supervised.ne(-100)]
    assert len(supervised) == 4
    assert int(supervised[-1]) == tokenizer.eos_token_id


def test_real_counterfactual_preserves_targets():
    samples = [
        {
            "id": str(index), "task": "it2i", "prompt": "edit",
            "image": torch.full((3, 2, 2), float(index)),
            "target_rgb": torch.full((3, 2, 2), float(index + 4)),
        }
        for index in range(2)
    ]
    controls = counterfactual_real_samples(samples)
    assert torch.equal(controls[0]["image"], samples[1]["image"])
    assert torch.equal(controls[0]["target_rgb"], samples[0]["target_rgb"])


def test_gaussian_pixel_likelihood_is_precision_weighted():
    from fine_grain.omni_model import DualStreamOmni

    model = DualStreamOmni(
        d_model=16, n_slices=4, n_layers=1, res=4, n_heads=4,
        pixel_loss_mode="gaussian_nll", s0_acc_coef=0.0, prior_write=0.0,
        prior_loss_coef=0.0, vfe_coef=0.0,
    )
    target = torch.ones(2, 3, 4, 4)
    out = {
        "rgb": torch.zeros_like(target),
        "rgb_lv": torch.zeros_like(target),
        "x_pred": None,
        "logits": torch.zeros(2, 10),
        "pred_loss": None,
        "vfe_loss": None,
    }
    batch = {
        "need_text": [False, False],
        "need_pix": [True, True],
        "need_seg": [False, False],
        "target_rgb": target,
        "target_image_precision": torch.tensor([1.0, 0.0]),
    }
    loss, meta = model.omni_loss(out, batch, torch.device("cpu"))
    assert torch.allclose(loss, torch.tensor(0.25))
    assert meta["gaussian_nll"] == float(loss)
