"""Load the common run-based export format and apply explicit split policies."""
import json
from pathlib import Path
import numpy as np
from ..split import split_data, split_indices
from .utils import get_proc_name

_ALIASES = {"dreyer_2023": "Dreyer2023", "lee_2019": "Lee2019_MI"}

DEFAULT_TMAX = {"Dreyer2023": 5.0, "Lee2019_MI": 4.0, "BNCI2014_004": 7.5, "GrosseWentrup2009": 7.0, "Schirrmeister2017": 4.0}


def _root(base):
    return (Path.home() if base is None else Path(base)) / "datasets"


def _name(name):
    return _ALIASES.get(name, name)


def labels_from_epochs(epochs, mappings=None):
    descriptions = {value: name for name, value in epochs.event_id.items()}
    markers = [descriptions[value] for value in epochs.events[:, 2]]
    if mappings is None:
        return np.asarray(markers)
    labels = []
    for marker in markers:
        matches = [value for key, value in mappings.items() if key in marker.split("/")]
        if len(matches) != 1:
            raise ValueError(f"Expected exactly one label mapping for {marker!r}")
        labels.append(matches[0])
    return np.asarray(labels)


def _session(name, subject, session, base, resample, tmin, tmax, l_freq, h_freq):
    path = (_root(base) / name / f"sub-{subject}" / f"ses-{session}" /
            get_proc_name(resample, tmin, tmax, l_freq, h_freq))
    prefix = f"sub-{subject}_ses-{session}"
    with open(path / f"{prefix}_meta.json") as stream:
        info = json.load(stream)
    if "runs" in info:
        runs = [item["id"] for item in info["runs"]]
    else:
        files = list(path.glob(f"{prefix}_run-*_y.npy"))
        runs = sorted(int(file.stem.split("_run-")[1].split("_")[0]) for file in files)
    if not runs:
        raise ValueError(f"No exported runs in {path}")
    return path, prefix, runs, info


def _read(path, prefix, runs, available, ea, online):
    if online and not ea:
        raise ValueError("online=True requires ea=True")
    if not runs or len(set(runs)) != len(runs) or not set(runs) <= set(available):
        raise ValueError(f"Select unique runs from {available}")
    # Respect recorded order even if callers supply runs in a different order.
    runs = [run for run in available if run in runs]
    labels = [np.load(path / f"{prefix}_run-{run}_y.npy") for run in available]
    for run, y in zip(available, labels):
        if y.ndim != 1:
            raise ValueError(f"Labels for run {run} must be one-dimensional")
    if ea:
        suffix = "_ea_online" if online else "_ea"
        X = np.load(path / f"{prefix}_X{suffix}.npy")
        if len(X) != sum(map(len, labels)):
            raise ValueError("Session EA and run labels have different trial counts")
        offsets = np.cumsum([0] + [len(y) for y in labels])
        arrays = [X[offsets[i]:offsets[i + 1]] for i, run in enumerate(available) if run in runs]
    else:
        arrays = [np.load(path / f"{prefix}_run-{run}_X.npy") for run in runs]
    ys = [labels[available.index(run)] for run in runs]
    if any(len(X) != len(y) for X, y in zip(arrays, ys)):
        raise ValueError("X and y must have equal trial counts in every run")
    return {"eeg": np.concatenate(arrays), "label": np.concatenate(ys)}


def get_data(name="Dreyer2023", subject=None, session=1, valid=False,
             train_test_split=0.8, train_valid_split=0.8,
             strategy="stratified_chronological", runs=None, ea=False, online=False,
             tmin=None, tmax=None, l_freq=7, h_freq=30, resample=128,
             base=None, return_info=False):
    """Load any common-format dataset.

    `runs={'train': [...], 'test': [...]}` selects disjoint run groups.
    Otherwise split all trials by class and time. `strategy='preset'` uses
    Dreyer's original runs 1/2 versus 3/4/5/6. Validation always comes from
    training trials. EA is read from the export cache, never recomputed.
    `base` denotes the directory containing `datasets`.
    """
    name = _name(name)
    tmin = 0.5 if tmin is None else tmin
    tmax = DEFAULT_TMAX.get(name, 5.0) if tmax is None else tmax
    path, prefix, available, info = _session(
        name, subject, session, base, resample, tmin, tmax, l_freq, h_freq)
    if strategy == "preset":
        if runs is not None:
            raise ValueError("Specify either preset or runs")
        runs = {"train": [1, 2], "test": [3, 4, 5, 6]} if name == "Dreyer2023" else None
        strategy = "stratified_chronological"
    if strategy not in ("stratified_chronological", "chronological"):
        raise ValueError(f"Unknown split strategy: {strategy}")
    if runs is not None:
        if set(runs) != {"train", "test"}:
            raise ValueError("runs must define train and test")
        if set(runs["train"]) & set(runs["test"]):
            raise ValueError("Train and test runs must be disjoint")
        result = {key: _read(path, prefix, selected, available, ea, online)
                  for key, selected in runs.items()}
    else:
        data = _read(path, prefix, available, available, ea, online)
        indices = split_indices(data["label"], (train_test_split, 1 - train_test_split), strategy)
        result = {key: {field: value[index] for field, value in data.items()}
                  for key, index in zip(("train", "test"), indices)}
    if valid:
        train = result["train"]
        indices = split_indices(train["label"], (train_valid_split, 1 - train_valid_split), strategy)
        for key, index in zip(("train", "valid"), indices):
            result[key] = {field: value[index] for field, value in train.items()}
    return (result, info) if return_info else result


def get_data_cross(name="Dreyer2023", subject=None, sessions=None, ea=False,
                   online=False, tmin=None, tmax=None, l_freq=7, h_freq=30,
                   resample=128, base=None, return_info=False):
    """Load all selected sessions without splitting; preserve session metadata."""
    name = _name(name)
    sessions = sessions_list(name, subject, base) if sessions is None else list(sessions)
    if not sessions or len(set(sessions)) != len(sessions):
        raise ValueError("Select at least one session without duplicates")
    tmin = 0.5 if tmin is None else tmin
    tmax = DEFAULT_TMAX.get(name, 5.0) if tmax is None else tmax
    data, metadata = [], {}
    for session in sessions:
        path, prefix, available, info = _session(
            name, subject, session, base, resample, tmin, tmax, l_freq, h_freq)
        data.append(_read(path, prefix, available, available, ea, online))
        metadata[str(session)] = info
    result = {key: np.concatenate([item[key] for item in data]) for key in ("eeg", "label")}
    return (result, {"sessions": metadata}) if return_info else result


def subject_list(name="Dreyer2023", base=None):
    """Discover exported subjects, including previously unknown datasets."""
    root = _root(base) / _name(name)
    return sorted(int(path.name[4:]) for path in root.glob("sub-*")
                  if path.is_dir() and path.name[4:].isdigit())


def sessions_list(name="Dreyer2023", subject=None, base=None):
    root = _root(base) / _name(name)
    subjects = [subject] if subject is not None else subject_list(name, base)
    return sorted({int(path.name[4:]) for item in subjects
                   for path in (root / f"sub-{item}").glob("ses-*")
                   if path.is_dir() and path.name[4:].isdigit()})
