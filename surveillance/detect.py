"""Temporal outbreak detectors on region-level series (EWMA and one-sided CUSUM)."""
import numpy as np


def zscore_baseline(x, base_days):
    mu, sd = np.nanmean(x[:base_days]), np.nanstd(x[:base_days])
    return (x - mu) / (sd if sd > 0 else 1.0)


def ewma_alarm(z, lam=0.3, L=3.0, base_days=60):
    e, out = 0.0, np.zeros(len(z), bool)
    sigma = np.sqrt(lam / (2 - lam))                 # asymptotic EWMA std for unit-variance z
    for t, v in enumerate(z):
        e = lam * v + (1 - lam) * e
        out[t] = (t >= base_days) and (e > L * sigma)
    return out


def cusum_alarm(z, k=0.5, h=4.0, base_days=60):
    s, out = 0.0, np.zeros(len(z), bool)
    for t, v in enumerate(z):
        s = max(0.0, s + v - k)
        out[t] = (t >= base_days) and (s > h)
        if out[t]:
            s = 0.0
    return out


def combined_z(*zs):
    return np.mean(zs, axis=0)
