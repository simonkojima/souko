"""Load the common run-based export format and apply explicit split policies."""

import json
from pathlib import Path
import numpy as np
from souko.split import split_indices
from mne_bids import get_entity_vals


def get_data(
    name="Dreyer2023",
    subject=None,
    session=1,
    valid=False,
    train_test_split=0.8,
    train_valid_split=0.8,
    strategy="stratified_chronological",
    runs=None,
    base=None,
    return_meata=False,
):

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
    else:
        indices = split_indices(
            data["label"], (train_test_split, 1 - train_test_split), strategy
        )
        result = {
            key: {field: value[index] for field, value in data.items()}
            for key, index in zip(("train", "test"), indices)
        }
    if valid:
        train = result["train"]
        indices = split_indices(
            train["label"], (train_valid_split, 1 - train_valid_split), strategy
        )
        for key, index in zip(("train", "valid"), indices):
            result[key] = {field: value[index] for field, value in train.items()}


def read_data(base, name, subject, session):

    fname = (
        base
        / name
        / f"sub-{subject}"
        / f"ses-{session}"
        / f"sub-{subject}_ses-{session}_X.npy"
    )
    X = np.load(fname)

    fname = (
        base
        / name
        / f"sub-{subject}"
        / f"ses-{session}"
        / f"sub-{subject}_ses-{session}_y.npy"
    )
    y = np.load(fname)

    fname = (
        base
        / name
        / f"sub-{subject}"
        / f"ses-{session}"
        / f"sub-{subject}_ses-{session}_meta.json"
    )
    with open(fname) as f:
        meta = json.load(f)

    return X, y, meta


def get_data_cross(
    name="Dreyer2023",
    subject=None,
    sessions=None,
    base=None,
    return_meta=False,
):

    if base is None:
        base = Path.home() / "datasets"

    if sessions is None:
        sessions = get_entity_vals(base / name / f"sub-{subject}", entity_key="session")

    X_list, y_list = [], []
    metadata = {}
    for session in sessions:
        X, y, meta = read_data(base, name, subject, session)
        metadata[str(session)] = meta
        X_list.append(X)
        y_list.append(y)

    data = {
        "eeg": np.concatenate(X_list, axis=0),
        "label": np.concatenate(y_list, axis=0),
    }

    if return_meta:
        return data, metadata
    else:
        return data


if __name__ == "__main__":
    print(get_data_cross(name="Dreyer2023", subject=1))
