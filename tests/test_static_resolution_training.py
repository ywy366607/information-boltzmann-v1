import torch

from fine_grain.gen_metrics import paired_digit_scores
from scripts.train_static_resolution import (
    BalancedStream, CASES, Raster, evaluation_bank, scene_bank,
    is_joint_candidate,
)


def test_descriptor_bank_covers_all_modes_colors_digits_and_addresses():
    scenes = scene_bank()
    assert len(scenes) == len(set(scenes)) == 360
    bank = evaluation_bank(scenes)
    assert len(bank) == len(set(bank)) == 1080
    stream = BalancedStream(scenes, 42)
    drawn = stream.take(1080)
    assert set(drawn) == set(bank)
    assert [d[0] for d in drawn[:6]] == list(CASES) * 2
    assert all(d[0] == CASES[0] for d in stream.take(12, generation_only=True))


def test_native_256_raster_and_metric_share_exact_geometry():
    torch.set_num_threads(4)
    raster = Raster()
    # Includes all digits, all four colors, and each of the nine addresses.
    scenes = scene_bank()
    for i in range(10):
        scene = (str(i), ("red", "green", "blue", "yellow")[i % 4], scenes[i % 9][2])
        sample = raster.sample((CASES[0], *scene))
        assert sample["target_rgb"].shape == (1, 3, 256, 256)
        assert sample["image"].count_nonzero() == 0
        assert sample["history_precision"].count_nonzero() == 0
        identity = paired_digit_scores(
            sample["target_rgb"], scene[0], scene[1], scene[2],
            box=raster.box, stroke_px=raster.stroke_px,
        )
        assert identity["paired_digit_top1"] == identity["paired_digit_iou"] == 1


def test_accumulation_matches_full_batch_for_independent_sample_means():
    torch.manual_seed(3)
    layer = torch.nn.Linear(5, 2)
    x, y = torch.randn(6, 5), torch.randn(6, 2)
    (layer(x) - y).square().mean().backward()
    expected = [p.grad.clone() for p in layer.parameters()]
    layer.zero_grad(set_to_none=True)
    for chunk in range(3):
        sl = slice(2 * chunk, 2 * chunk + 2)
        ((layer(x[sl]) - y[sl]).square().mean() / 3).backward()
    for parameter, gradient in zip(layer.parameters(), expected):
        torch.testing.assert_close(parameter.grad, gradient)


def test_warmup_cannot_win_joint_checkpoint_selection():
    assert not is_joint_candidate(80, 80, 1.0, -float("inf"))
    assert is_joint_candidate(81, 80, -0.6, -float("inf"))
    assert not is_joint_candidate(160, 80, -0.7, -0.6)
