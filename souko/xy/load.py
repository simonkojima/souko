"""Load the common run-based export format and apply explicit split policies."""
import json
from pathlib import Path
import numpy as np
from ..split import split_indices


def _root(base):
    return Path.home() / "datasets" if base is None else Path(base)


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


def _session(name, subject, session, base):
    path = _root(base) / name / f"sub-{subject}" / f"ses-{session}"
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


def _read(path, prefix, runs, available, alignment, alignment_mode, online):
    if online and not alignment:
        raise ValueError("online=True requires alignment=True (or ea=True)")
    if alignment_mode not in ("euclidean", "channel_norm"):
        raise ValueError(f"Unknown alignment mode: {alignment_mode}")
    if not runs or len(set(runs)) != len(runs) or not set(runs) <= set(available):
        raise ValueError(f"Select unique runs from {available}")
    # Respect recorded order even if callers supply runs in a different order.
    runs = [run for run in available if run in runs]
    labels = [np.load(path / f"{prefix}_run-{run}_y.npy") for run in available]
    for run, y in zip(available, labels):
        if y.ndim != 1:
            raise ValueError(f"Labels for run {run} must be one-dimensional")
    if alignment:

        if alignment_mode == "euclidean":
            suffix = "_ea_online" if online else "_ea"
        elif alignment_mode == "channel_norm":
            suffix = "_cn_online" if online else "_cn"


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
             base=None, return_info=False, alignment=None, alignment_mode="euclidean"):

    alignment = ea if alignment is None else alignment
    path, prefix, available, info = _session(name, subject, session, base)
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
        result = {key: _read(path, prefix, selected, available, alignment, alignment_mode, online)
                  for key, selected in runs.items()}
    else:
        data = _read(path, prefix, available, available, alignment, alignment_mode, online)
        indices = split_indices(data["label"], (train_test_split, 1 - train_test_split), strategy)
        result = {key: {field: value[index] for field, value in data.items()}
                  for key, index in zip(("train", "test"), indices)}
    if valid:
        train = result["train"]
        indices = split_indices(train["label"], (train_valid_split, 1 - train_valid_split), strategy)
        for key, index in zip(("train", "valid"), indices):
            result[key] = {field: value[index] for field, value in train.items()}
    return (result, info) if return_info else result


def get_data_cross(name="Dreyer2023", subject=None, sessions=None, alignment=False, alignment_mode="euclidean",
                   online=False, base=None, return_info=False):
    """Load all selected sessions without splitting; preserve session metadata."""
    sessions = sessions_list(name, subject, base) if sessions is None else list(sessions)
    if not sessions or len(set(sessions)) != len(sessions):
        raise ValueError("Select at least one session without duplicates")
    data, metadata = [], {}
    for session in sessions:
        path, prefix, available, info = _session(
            name, subject, session, base)
        data.append(_read(path=path, prefix=prefix, runs=available, available=available, alignment=alignment, alignment_mode=alignment_mode, online=online))
        metadata[str(session)] = info
    result = {key: np.concatenate([item[key] for item in data]) for key in ("eeg", "label")}
    return (result, {"sessions": metadata}) if return_info else result


def subject_list(name="Dreyer2023", base=None):
    """Discover exported subjects, including previously unknown datasets."""
    root = _root(base) / name
    return sorted(int(path.name[4:]) for path in root.glob("sub-*")
                  if path.is_dir() and path.name[4:].isdigit())


def sessions_list(name="Dreyer2023", subject=None, base=None):
    root = _root(base) / name
    subjects = [subject] if subject is not None else subject_list(name, base)
    return sorted({int(path.name[4:]) for item in subjects
                   for path in (root / f"sub-{item}").glob("ses-*")
                   if path.is_dir() and path.name[4:].isdigit()})
