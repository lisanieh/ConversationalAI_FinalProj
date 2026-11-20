"""
audio_checks.py — measures whether the conditions themselves are right: does the actual SNR after mixing equal the nominal SNR?
A quality check for the measurement owner: if noise_5 is really 9 dB, every per-condition conclusion is shifted.

    mix_at_snr(clean, noise, snr_db)  ->  mixed waveform (to cross-check the speech pipeline's implementation)
    measured_snr(clean, mixed)        ->  true SNR when the clean signal is known
"""
from __future__ import annotations

import numpy as np


def _power(x: np.ndarray) -> float:
    return float(np.mean(np.asarray(x, float) ** 2) + 1e-12)


def mix_at_snr(clean: np.ndarray, noise: np.ndarray, snr_db: float, rng=None) -> np.ndarray:
    """SNR = 10*log10(P_speech / P_noise). Noise gain g = sqrt(P_s / (P_n * 10^(SNR/10))).
    Loop the noise if it is shorter than the speech; otherwise take a random excerpt (log the start for reproducibility)."""
    rng = rng or np.random.default_rng(0)
    if len(noise) < len(clean):
        noise = np.tile(noise, int(np.ceil(len(clean) / len(noise))))
    start = rng.integers(0, len(noise) - len(clean) + 1)
    n = noise[start:start + len(clean)]
    g = np.sqrt(_power(clean) / (_power(n) * 10 ** (snr_db / 10)))
    mixed = clean + g * n
    peak = np.max(np.abs(mixed))
    return mixed / peak * 0.99 if peak > 1 else mixed   # prevent clipping; global scaling does not change SNR


def measured_snr(clean: np.ndarray, mixed: np.ndarray) -> float:
    """Least-squares fit of the speech share a in the mix; the residual is the noise. Insensitive to global anti-clipping scaling."""
    clean, mixed = np.asarray(clean, float), np.asarray(mixed, float)
    a = float(np.dot(mixed, clean) / np.dot(clean, clean))
    resid = mixed - a * clean
    return 10 * np.log10(_power(a * clean) / _power(resid))


def check_condition_table(rows, tol_db: float = 0.5) -> list[dict]:
    """rows: [{"id", "clean": array, "mixed": array, "nominal_snr": float}]; returns the entries outside tolerance."""
    bad = []
    for r in rows:
        m = measured_snr(r["clean"], r["mixed"])
        if abs(m - r["nominal_snr"]) > tol_db:
            bad.append({"id": r["id"], "nominal": r["nominal_snr"], "measured": round(m, 2)})
    return bad
