"""PSG 데이터 전처리 파이프라인."""

import os
import xml.etree.ElementTree as ET

import numpy as np
import matplotlib.pyplot as plt
from pyedflib import highlevel

EDF_DIR = XML_DIR = "data/edf"
ECG_FS, EMG_FS = 250, 125
WIN_SEC = 10
TRIM_MIN = 15
EPOCH_SEC = 30
STAGE_NAME = {0: "Wake", 1: "N1", 2: "N2", 3: "N3", 4: "N3", 5: "REM"}


def load_channels(nsrrid, targets=("ECG", "EMG")):
    """EDF에서 지정 채널만 {라벨: (신호, fs)} 로 반환."""
    path = f"{EDF_DIR}/shhs2-{nsrrid}.edf"
    signals, headers, _ = highlevel.read_edf(path, verbose=False)
    out = {
        h["label"]: (
            np.asarray(s, float),
            int(h.get("sample_frequency", h.get("sample_rate"))),
        )
        for s, h in zip(signals, headers)
    }
    return {k: out[k] for k in targets if k in out}, [h["label"] for h in headers]


def read_hypnogram(nsrrid, epoch_sec=EPOCH_SEC):
    """XML에서 수면단계 hypnogram 배열 복원."""
    root = ET.parse(f"{XML_DIR}/shhs2-{nsrrid}-nsrr.xml").getroot()
    hyp = []
    for ev in root.iter("ScoredEvent"):
        et = ev.find("EventType")
        if et is None or et.text != "Stages|Stages":
            continue
        stage = int(ev.find("EventConcept").text.split("|")[-1])
        dur = float(ev.find("Duration").text)
        hyp += [stage] * int(dur // epoch_sec)
    return np.array(hyp)


def hypnogram_metrics(hyp, epoch_sec=EPOCH_SEC):
    """hypnogram에서 수면 지표 계산."""
    n = len(hyp)
    sleep = np.isin(hyp, [1, 2, 3, 5])
    tib = n * epoch_sec / 3600
    tst = sleep.sum() * epoch_sec / 3600
    onset = int(np.argmax(sleep)) if sleep.any() else n
    waso = int((hyp[onset:] == 0).sum()) * epoch_sec / 60
    rem = np.where(hyp == 5)[0]
    rl = (rem[0] - onset) * epoch_sec / 60 if len(rem) else float("nan")
    return dict(
        TST_h=round(float(tst), 1),
        SleepEff_pct=round(float(tst / tib * 100), 1) if tib else 0,
        SleepLatency_min=round(float(onset * epoch_sec / 60), 1),
        WASO_min=round(float(waso), 1),
        REMLatency_min=round(float(rl), 1),
        REM_pct=round(float((hyp == 5).sum() / max(sleep.sum(), 1) * 100), 1),
    )


def make_windows(sig, fs, win_sec=WIN_SEC, trim_min=TRIM_MIN):
    """신호를 10초 윈도우로 자르기 (앞뒤 trim_min 분 제거)."""
    trim = trim_min * 60 * fs
    sig = sig[trim:-trim] if trim > 0 else sig
    win = win_sec * fs
    n = len(sig) // win
    return sig[: n * win].reshape(n, win)


def window_stages(nsrrid, win_sec=WIN_SEC, trim_min=TRIM_MIN, epoch_sec=EPOCH_SEC):
    """각 10초 윈도우에 해당하는 수면단계 반환."""
    hyp = read_hypnogram(nsrrid)
    ch, _ = load_channels(nsrrid)
    n = min(
        len(make_windows(ch["ECG"][0], ECG_FS)),
        len(make_windows(ch["EMG"][0], EMG_FS)),
    )
    idx = ((trim_min * 60 + np.arange(n) * win_sec) // epoch_sec).astype(int)
    return hyp[np.clip(idx, 0, len(hyp) - 1)]


def build_dataset(subjects):
    """여러 피험자 ECG·EMG·라벨을 멀티모달 배열로 구성."""
    ecg_list, emg_list, labels = [], [], []
    for nsrrid, label in subjects:
        ch, _ = load_channels(nsrrid)
        ecg = make_windows(ch["ECG"][0], ECG_FS)
        emg = make_windows(ch["EMG"][0], EMG_FS)
        n = min(len(ecg), len(emg))
        ecg_list.append(ecg[:n])
        emg_list.append(emg[:n])
        labels.extend([label] * n)
        print(f"{nsrrid} (PD={label}) -> {n} windows")
    return np.vstack(ecg_list), np.vstack(emg_list), np.array(labels, dtype=np.int8)


def plot_hypnogram(nsrrid):
    """배정 환자의 수면단계 hypnogram 시각화."""
    hyp = read_hypnogram(nsrrid)
    ymap = {0: 4, 1: 3, 2: 2, 3: 1, 5: 0}
    time_h = np.arange(len(hyp)) * EPOCH_SEC / 3600
    y = np.array([ymap.get(s, 2) for s in hyp])

    fig, ax = plt.subplots(figsize=(12, 4))
    rem_mask = hyp == 5
    if rem_mask.any():
        rem_starts = np.where(np.diff(rem_mask.astype(int)) == 1)[0]
        rem_ends = np.where(np.diff(rem_mask.astype(int)) == -1)[0]
        if rem_mask[0]:
            rem_starts = np.insert(rem_starts, 0, 0)
        if rem_mask[-1]:
            rem_ends = np.append(rem_ends, len(rem_mask) - 1)
        for s, e in zip(rem_starts, rem_ends):
            ax.axvspan(
                s * EPOCH_SEC / 3600,
                (e + 1) * EPOCH_SEC / 3600,
                alpha=0.15,
                color="mediumpurple",
            )

    ax.step(time_h, y, where="post", color="steelblue", linewidth=1.2)
    ax.set_xlabel("Time (h)")
    ax.set_ylabel("Sleep Stage")
    ax.set_yticks([4, 3, 2, 1, 0])
    ax.set_yticklabels(["Wake", "N1", "N2", "N3", "REM"])
    ax.set_title(f"Hypnogram — subject {nsrrid}")
    ax.set_xlim(0, time_h[-1])
    ax.invert_yaxis()
    plt.tight_layout()
    return fig, ax
