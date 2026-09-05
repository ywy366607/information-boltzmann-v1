import sys

import pytest
import torch

from scripts import train_mixed_native as runner


def test_resume_preserves_adam_schedule_and_example_position(tmp_path, monkeypatch):
    """Same four updates, interrupted after a saved second update, are identical."""
    class Graph(torch.nn.Module):
        def __init__(self, **kwargs):
            super().__init__()
            self.w = torch.nn.Parameter(torch.tensor(0.1))
            self.lm = torch.nn.Module()
            self.lm_tok = None

        def omni_loss(self, out, batch, device):
            return out["loss"], {}

        def non_lm_state_dict(self):
            return self.state_dict()

        def language_meta(self):
            return {}

    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner, "DualStreamOmni", Graph)
    monkeypatch.setattr(runner, "build_real_bank", lambda a, b, r: [dict(id="sample", task="t2i", target=r/256)])
    monkeypatch.setattr(runner, "add_zero_horizon_controls", lambda b: b)
    monkeypatch.setattr(runner, "collate_real_capacity", lambda tok, samples, device: samples[0])
    monkeypatch.setattr(runner, "write_gallery", lambda *args: None)
    monkeypatch.setattr(runner, "evaluate", lambda *args: (
        {"summary": {}, "samples": [dict(id="sample", task="t2i", psnr=40, edge_correlation=1, control_mse_gap=1)]}, []))
    calls = {"count": 0, "stop": None}

    def forward(model, batch, **kwargs):
        calls["count"] += 1
        if calls["count"] == calls["stop"]:
            raise RuntimeError("simulated interruption")
        return {"loss": (model.w - batch["target"]).square()}

    monkeypatch.setattr(runner, "forward_real_capacity", forward)
    (tmp_path / "checkpoints").mkdir()
    torch.save({"config": {}, "state_dict": Graph().state_dict(), "step": 0}, tmp_path / "checkpoints/start.pt")

    def run(tag, init="checkpoints/start.pt", resume=False):
        argv = ["train", "--init", init, "--tag", tag, "--device", "cpu", "--steps", "4",
                "--eval-every", "2", "--sharegpt-manifest", "share", "--davis-manifest", "davis"]
        if resume:
            argv.append("--resume-training")
        monkeypatch.setattr(sys, "argv", argv)
        runner.main()

    run("uninterrupted")
    calls.update(count=0, stop=5)  # two resolutions per update
    with pytest.raises(RuntimeError, match="simulated interruption"):
        run("interrupted")
    saved = torch.load(tmp_path / "checkpoints/interrupted.pt", weights_only=False)
    assert saved["step"] == 2
    calls.update(count=0, stop=None)
    run("resumed", "checkpoints/interrupted.pt", resume=True)
    complete = torch.load(tmp_path / "checkpoints/uninterrupted.pt", weights_only=False)
    resumed = torch.load(tmp_path / "checkpoints/resumed.pt", weights_only=False)
    torch.testing.assert_close(complete["state_dict"]["w"], resumed["state_dict"]["w"], rtol=0, atol=0)
    assert complete["seen"] == resumed["seen"] == {"64": {"sample": 4}, "256": {"sample": 4}}
    assert complete["optimizer"]["param_groups"] == resumed["optimizer"]["param_groups"]
    for key, value in complete["optimizer"]["state"][0].items():
        torch.testing.assert_close(value, resumed["optimizer"]["state"][0][key], rtol=0, atol=0)
