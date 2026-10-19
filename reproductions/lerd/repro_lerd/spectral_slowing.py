"""Independent check of the AD spectral-slowing premise on ds004504 (Cohort A).

WHAT THIS TESTS (and what it does NOT)
--------------------------------------
LERD claim 6 / Figure 2 reports that the model's *inferred dLIF frequency*
slows with disease severity (HC > MCI > AD) and states this is "consistent with
established AD EEG slowing". The Figure 2 quantity is a MODEL OUTPUT (dLIF firing
frequency ~ 1/E[tau], with tau the MELP lognormal inter-event interval,
Algorithm 1). It cannot be reproduced without training LERD, and the paper gives
no raw-EEG central-frequency formula.

This script therefore verifies only the BIOMARKER PREMISE the claim rests on,
directly from the raw EEG, using standard measures:
  * spectral centroid over 4-12 Hz (theta+alpha), the closest standard analog to
    the paper's ~6.6-7.8 Hz central frequencies;
  * relative band powers (delta/theta/alpha/beta), the canonical AD signature
    (increased delta/theta, reduced alpha/beta; Jeong 2004; Dauwels 2010).

ds004504 groups are A=AD (36), F=FTD (23), C=HC (29). There is NO MCI group here
(the paper's MCI numbers come from the unavailable Cohort B), so we test the
AD-vs-HC ordering only, plus report FTD for context.

Deps: mne (read .set), numpy, scipy.signal.welch. CPU only.
"""

from __future__ import annotations

import csv
import glob
import os

import numpy as np
from scipy.signal import welch

import mne

DATA = os.path.join(os.path.dirname(__file__), "data", "ds004504")
DERIV = os.path.join(DATA, "derivatives")

BANDS = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 25.0),
}
CENTROID_BAND = (4.0, 12.0)  # theta+alpha; matches the paper's central-freq range


def load_labels() -> dict[str, str]:
    """participant_id -> Group (A/F/C) from participants.tsv."""
    path = os.path.join(DATA, "participants.tsv")
    with open(path) as fh:
        return {r["participant_id"]: r["Group"] for r in csv.DictReader(fh, delimiter="\t")}


def spectral_centroid(freqs: np.ndarray, psd: np.ndarray, band: tuple) -> float:
    """PSD-weighted mean frequency within `band` (the 'central frequency')."""
    lo, hi = band
    m = (freqs >= lo) & (freqs <= hi)
    p = psd[m]
    f = freqs[m]
    return float(np.sum(f * p) / np.sum(p))


def rel_band_power(freqs: np.ndarray, psd: np.ndarray) -> dict[str, float]:
    """Relative power per band (fraction of 0.5-25 Hz total)."""
    total_m = (freqs >= 0.5) & (freqs <= 25.0)
    total = np.trapz(psd[total_m], freqs[total_m])
    out = {}
    for name, (lo, hi) in BANDS.items():
        m = (freqs >= lo) & (freqs <= hi)
        out[name] = float(np.trapz(psd[m], freqs[m]) / total)
    return out


def analyze_subject(set_path: str) -> dict:
    """Return per-channel centroid and per-subject mean relative band powers."""
    raw = mne.io.read_raw_eeglab(set_path, preload=True, verbose="ERROR")
    sf = raw.info["sfreq"]
    data = raw.get_data()  # (n_ch, n_times), volts
    ch = raw.ch_names
    # Welch per channel; 4 s segments give ~0.25 Hz resolution
    nper = int(4 * sf)
    freqs, psd = welch(data, fs=sf, nperseg=nper, axis=1)
    centroids = {
        ch[c]: spectral_centroid(freqs, psd[c], CENTROID_BAND) for c in range(len(ch))
    }
    # subject-mean relative band power (mean PSD across channels first)
    rbp = rel_band_power(freqs, psd.mean(axis=0))
    return {"centroids": centroids, "rbp": rbp, "channels": ch}


def main():
    labels = load_labels()
    sets = sorted(glob.glob(os.path.join(DERIV, "sub-*", "eeg", "*_eeg.set")))
    if not sets:
        raise SystemExit("no .set files downloaded yet under " + DERIV)

    per_group_centroid: dict[str, dict[str, list]] = {g: {} for g in "AFC"}
    per_group_rbp: dict[str, dict[str, list]] = {g: {b: [] for b in BANDS} for g in "AFC"}
    channels = None

    for sp in sets:
        sid = os.path.basename(sp).split("_")[0]  # sub-001
        g = labels.get(sid)
        if g not in "AFC":
            continue
        res = analyze_subject(sp)
        channels = res["channels"]
        for c, v in res["centroids"].items():
            per_group_centroid[g].setdefault(c, []).append(v)
        for b, v in res["rbp"].items():
            per_group_rbp[g][b].append(v)

    n = {g: len(per_group_rbp[g]["theta"]) for g in "AFC"}
    print(f"Subjects analyzed: AD={n['A']} FTD={n['F']} HC={n['C']}")
    print()

    # --- spectral-slowing test: AD centroid < HC centroid, per channel ---
    if not (n["A"] and n["C"]):
        raise SystemExit(f"need both AD and HC subjects; have {n}. "
                         "Wait for the full download to finish, then rerun.")

    ad_lower = 0
    gmean = lambda g, c: np.mean(per_group_centroid[g][c]) if per_group_centroid[g].get(c) else float("nan")
    print("Per-channel spectral centroid 4-12 Hz (mean Hz)  [AD  FTD  HC]  AD<HC?")
    for c in channels:
        a = gmean("A", c)
        f = gmean("F", c)
        h = gmean("C", c)
        ok = a < h
        ad_lower += ok
        print(f"  {c:>4}  {a:5.2f} {f:5.2f} {h:5.2f}   {'yes' if ok else 'NO'}")
    print(f"\nAD centroid < HC in {ad_lower}/{len(channels)} channels "
          f"(paper's model-derived claim: HC highest in 18/19).")

    # --- band-power signature: AD theta up, alpha down vs HC ---
    print("\nRelative band power (group mean):")
    print(f"  {'band':>6}  {'AD':>6} {'FTD':>6} {'HC':>6}")
    for b in BANDS:
        a = np.mean(per_group_rbp["A"][b])
        f = np.mean(per_group_rbp["F"][b])
        h = np.mean(per_group_rbp["C"][b])
        print(f"  {b:>6}  {a:6.3f} {f:6.3f} {h:6.3f}")
    print("\nExpected AD signature: theta higher, alpha lower vs HC.")


if __name__ == "__main__":
    main()
