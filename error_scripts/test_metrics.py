"""
test_metrics.py — self-tests for the measurement scripts. A wrong metric is more dangerous than a wrong model, because nobody suspects it.
    python -m pytest -q test_metrics.py      (or: python test_metrics.py)
"""
import numpy as np
import pandas as pd

import calibration_metrics as cm
import episode_metrics as em
from align import label_slot, normalize, wer_counts
from audio_checks import measured_snr, mix_at_snr
from stats_tests import holm
from u2r import ActionParams, choose, expected_cost, spell_type_threshold

rng = np.random.default_rng(0)


def test_normalize():
    assert normalize("Seven thirty") == normalize("7:30") == "7 30"
    assert normalize("twenty one") == "21"
    assert normalize("Center!") == "centre"


def test_wer():
    assert wer_counts("a b c", "a b c")["wer"] == 0
    c = wer_counts("a b c", "a x c d")
    assert (c["S"], c["D"], c["I"]) == (1, 0, 1)
    assert wer_counts("a", "a b c d")["wer"] == 3  # WER can exceed 1


def test_slot_labels():
    assert label_slot("for four people", "for people", "4", "")["error_type"] == "deletion"
    assert label_slot("the cambridge lodge", "the came bridge lodge", "cambridge lodge", "came bridge lodge")["error_type"] == "substitution"
    assert label_slot("", "thank you", None, "thank you")["error_type"] == "insertion"
    assert label_slot("in the centre", "in the centre", "centre", "north")["error_type"] == "extraction"


def test_calibration_perfect_vs_overconfident():
    p = rng.random(200_000)
    y = (rng.random(p.size) < p).astype(int)          # perfectly calibrated
    assert cm.ece(y, p) < 0.01
    assert cm.ece(y, np.clip(p * 1.5, 0, 1)) > 0.08    # systematic overestimate
    # same ranking -> same AUROC, but worse ECE: calibration != discrimination
    assert abs(cm.discrimination(y, p)["auroc"] - cm.discrimination(y, p ** 3)["auroc"]) < 1e-9


def test_brier_decomposition_adds_up():
    p = rng.random(50_000); y = (rng.random(p.size) < p).astype(int)
    d = cm.brier_decomposition(y, p, n_bins=50)
    assert abs(d["brier"] - (d["reliability"] - d["resolution"] + d["uncertainty"])) < 0.01


def test_u2r_threshold_matches_bruteforce():
    S, T = ActionParams(8, .5, .1), ActionParams(15, .97, .02)
    p = spell_type_threshold(S, T, 60)
    assert abs(expected_cost(p, S, 60) - expected_cost(p, T, 60)) < 1e-9
    P = {"ACCEPT": ActionParams(0, 0, 0), "CONFIRM": ActionParams(4, .6, .02), "SPELL": S, "TYPE": T}
    assert choose(0.0, P, 60)[0] == "ACCEPT" and choose(0.99, P, 60)[0] == "TYPE"
    assert choose(0.99, P, 60, ("ACCEPT", "CONFIRM", "SPELL"))[0] != "TYPE"  # user who cannot type


def test_holm():
    h = holm({"a": .01, "b": .02, "c": .04})
    assert [round(h[k]["p_holm"], 3) for k in "abc"] == [0.03, 0.04, 0.04]


def test_snr_mixing():
    clean = np.sin(np.linspace(0, 2000, 16000)) * .3
    noise = rng.normal(0, 1, 40000)
    for snr in (20, 5, 0, -5):
        assert abs(measured_snr(clean, mix_at_snr(clean, noise, snr)) - snr) < 0.2


def test_episode_metrics():
    goal = {"d1": {"people": "4", "day": "monday"}}
    turns = pd.DataFrame([
        dict(dialogue_id="d1", seed=0, condition="c", policy="U2R", turn_id=0,
             gold_state={"people": "4"}, belief_state={"people": "for"}, commits={"people": "for"},
             repairs=[{"slot": "people", "action": "TYPE", "effort_s": 12}]),
        dict(dialogue_id="d1", seed=0, condition="c", policy="U2R", turn_id=1,
             gold_state={"people": "4", "day": "monday"}, belief_state={"people": "4", "day": "monday"},
             commits={"people": "four"}, repairs=[]),
    ])
    r = em.per_dialogue(turns, goal).iloc[0]
    assert r.task_success == 1 and r.jga == 0.5 and r.wrong_commit_rate == 0.5
    assert r.first_error_turn == 0 and r.effort_s == 12
    km = em.km_survival([0, None, 3], [5, 5, 5])
    assert abs(km.survival.iloc[-1] - (2 / 3) * (1 / 2)) < 1e-9


if __name__ == "__main__":
    for k, f in list(globals().items()):
        if k.startswith("test_"):
            f(); print("ok", k)
