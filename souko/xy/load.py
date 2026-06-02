import json
import mne
from pathlib import Path
import numpy as np
from .utils import get_proc_name


def _markers_from_events(events, event_id):
    event_desc = {v: k for k, v in event_id.items()}

    samples = np.array(events)[:, 0]

    markers = list()
    for val in np.array(events)[:, 2]:
        markers.append(str(event_desc[val]))

    return samples, markers


def labels_from_epochs(epochs, mappings=None):
    y = list()

    _, markers = _markers_from_events(epochs.events, epochs.event_id)

    if mappings is None:
        return np.array(markers)

    for marker in markers:
        for key, val in mappings.items():
            if "/" in marker:
                if key in marker.split("/"):
                    y.append(val)
            else:
                if key in marker:
                    y.append(val)

    if len(epochs) != len(y):
        raise RuntimeError(
            f"lenth of epochs is not match with length of y.\n len(epochs): {len(epochs)}, len(y): {len(y)}"
        )

    return np.array(y)


def get_channel_pos(ch_name, layout_name="EEG1005", ref="Cz"):
    layout = mne.channels.read_layout("EEG1005")

    pos = layout.copy().pick(picks=ch_name).pos

    pos_ref = layout.pick(picks=ref).pos

    if pos.shape[0] != 1:
        raise ValueError("pos.shape[0] != 1")

    pos = pos[0, 0:2]
    pos_ref = pos_ref[0, 0:2]

    return pos - pos_ref


def split_data(X, y, ratio=0.8):
    I_0 = np.where(y == 0)[0]
    I_1 = np.where(y == 1)[0]

    N_0 = int(len(I_0) * ratio)
    N_1 = int(len(I_1) * ratio)

    I_0_p = I_0[:N_0]
    I_1_p = I_1[:N_1]

    I_0_s = I_0[N_0:]
    I_1_s = I_1[N_1:]

    I_p = np.concatenate([I_0_p, I_1_p])
    I_s = np.concatenate([I_0_s, I_1_s])

    X_p = X[I_p]
    y_p = y[I_p]

    X_s = X[I_s]
    y_s = y[I_s]

    return X_p, y_p, X_s, y_s


def _get_dreyer_2023(
        subject,
        session=1,
        valid=False,
        train_valid_split=0.8,
        tmin=0.5,
        tmax=5,
        l_freq=7,
        h_freq=30,
        resample=128,
        base=None,
        return_info=False,
):
    if base is None:
        base = Path.home()

    base = (
            base
            / "datasets"
            / "Dreyer2023"
            / f"sub-{subject}"
            / f"ses-{session}"
            / get_proc_name(
        resample=resample,
        tmin=tmin,
        tmax=tmax,
        l_freq=l_freq,
        h_freq=h_freq,
    )
    )

    runs = {"train": [1, 2], "test": [3, 4, 5, 6]}

    data_dict = {}

    with open(base / f"sub-{subject}_ses-{session}_meta.json", "r") as f:
        info = json.load(f)

    for split, run_list in runs.items():

        X_list, y_list = [], []
        for run in run_list:
            X = np.load(base / f"sub-{subject}_ses-{session}_run-{run}_X.npy")
            y = np.load(base / f"sub-{subject}_ses-{session}_run-{run}_y.npy")

            X_list.append(X)
            y_list.append(y)

        X = np.concatenate(X_list, axis=0)
        y = np.concatenate(y_list, axis=0)

        data_dict[split] = {
            "eeg": X,
            "label": y,
            "pos": [get_channel_pos(ch) for ch in info["ch_names"]],
        }

        if valid:
            X_train, y_train, X_valid, y_valid = split_data(X, y, train_valid_split)

            data_dict["valid"] = {}

            data_dict["train"]["eeg"] = X_train
            data_dict["train"]["label"] = y_train
            data_dict["valid"]["eeg"] = X_valid
            data_dict["valid"]["label"] = y_valid
            data_dict["valid"]["pos"] = [get_channel_pos(ch) for ch in info["ch_names"]]

    if return_info:
        return data_dict, info
    else:
        return data_dict


def _get_dreyer_2023_cross(
        subject,
        ea=False,
        online=False,
        tmin=0.5,
        tmax=5,
        l_freq=7,
        h_freq=30,
        resample=128,
        base=None,
        return_info=False,
):
    session = 1

    if base is None:
        base = Path.home()

    base = (
            base
            / "datasets"
            / "Dreyer2023"
            / f"sub-{subject}"
            / f"ses-{session}"
            / get_proc_name(
        resample=resample,
        tmin=tmin,
        tmax=tmax,
        l_freq=l_freq,
        h_freq=h_freq,
    )
    )

    with open(base / f"sub-{subject}_ses-{session}_meta.json", "r") as f:
        info = json.load(f)

    if ea is False:
        X = np.load(base / f"sub-{subject}_ses-{session}_X.npy")
    else:
        if online is False:
            X = np.load(base / f"sub-{subject}_ses-{session}_X_ea.npy")
        else:
            X = np.load(base / f"sub-{subject}_ses-{session}_X_ea_online.npy")

    y = np.load(base / f"sub-{subject}_ses-{session}_y.npy")

    data_dict = {
        "eeg": X,
        "label": y,
        "pos": [get_channel_pos(ch) for ch in info["ch_names"]],
    }

    if return_info:
        return data_dict, info
    else:
        return data_dict


def _get_lee_2019(
        subject,
        session,
        train_test_split=0.8,
        train_valid_split=0.8,
        valid=False,
        ea=False,
        online=False,
        base=None,
):
    if base is None:
        base = Path.home()

    base = base / "datasets" / "lee2019" / "x_y" / f"sub-{subject}"

    data_dict = {}

    with open(base / f"sub-{subject}_info.json", "r") as f:
        info = json.load(f)
    data_dict["info"] = info

    if ea is False:
        X = np.load(base / f"sub-{subject}_ses-{session}_x.npy")
    else:
        if online is False:
            X = np.load(base / f"sub-{subject}_ses-{session}_x_ea.npy")
        else:
            X = np.load(base / f"sub-{subject}_ses-{session}_x_ea_online.npy")

    y = np.load(base / f"sub-{subject}_ses-{session}_y.npy")

    X_train, y_train, X_test, y_test = split_data(X, y, train_test_split)

    if valid:
        X_train, y_train, X_valid, y_valid = split_data(
            X_train, y_train, train_valid_split
        )

    data_dict["train"] = {
        "eeg": X_train,
        "label": y_train,
        "pos": [get_channel_pos(ch) for ch in info["ch_names"]],
    }

    if valid:
        data_dict["valid"] = {
            "eeg": X_valid,
            "label": y_valid,
            "pos": [get_channel_pos(ch) for ch in info["ch_names"]],
        }

    data_dict["test"] = {
        "eeg": X_test,
        "label": y_test,
        "pos": [get_channel_pos(ch) for ch in info["ch_names"]],
    }

    return data_dict


def _get_lee_2019_mi_cross(
        subject,
        ea=False,
        online=False,
        tmin=0.5,
        tmax=4,
        l_freq=7,
        h_freq=30,
        resample=128,
        base=None,
        return_info=False,
):
    if base is None:
        base = Path.home()

    X_list, y_list = [], []
    for session in [1, 2]:

        base_session = (
                base
                / "datasets"
                / "Lee2019_MI"
                / f"sub-{subject}"
                / f"ses-{session}"
                / get_proc_name(
            resample=resample,
            tmin=tmin,
            tmax=tmax,
            l_freq=l_freq,
            h_freq=h_freq,
        )
        )

        with open(base_session / f"sub-{subject}_ses-{session}_meta.json", "r") as f:
            info = json.load(f)

        if ea is False:
            X = np.load(base_session / f"sub-{subject}_ses-{session}_X.npy")
        else:
            if online is False:
                X = np.load(base_session / f"sub-{subject}_ses-{session}_X_ea.npy")
            else:
                X = np.load(base_session / f"sub-{subject}_ses-{session}_X_ea_online.npy")

        y = np.load(base_session / f"sub-{subject}_ses-{session}_y.npy")

        X_list.append(X)
        y_list.append(y)

    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)

    data_dict = {
        "eeg": X,
        "label": y,
        "pos": [get_channel_pos(ch) for ch in info["ch_names"]],
    }

    return data_dict


MAPPING = {
    "Dreyer2023": _get_dreyer_2023,
    "Lee2019_MI": _get_lee_2019,
}

MAPPING_CROSS = {
    "Dreyer2023": _get_dreyer_2023_cross,
    "Lee2019_MI": _get_lee_2019_mi_cross,
}


def get_data_cross(name="Dreyer2023", **kwargs):
    return MAPPING_CROSS[name](**kwargs)


def get_data(name="dreyer_2023", **kwargs):
    return MAPPING[name](**kwargs)


def subject_list(name="dreyer_2023"):
    match name:
        case "Dreyer2023":
            subject_list = list(range(1, 88))
            subject_list.remove(40)
            subject_list.remove(59)
            return subject_list
        case "Lee2019_MI":
            return list(range(1, 55))
        case _:
            raise NotImplementedError(f"Available datasets: {list(MAPPING.keys())}")


def sessions_list(name="dreyer_2023"):
    match name:
        case "dreyer_2023":
            return [1]
        case "lee_2019":
            return [1, 2]
        case _:
            raise NotImplementedError


if __name__ == "__main__":
    # data = get_data("dreyer_2023", subject=1, resample=200)
    data = get_data("dreyer_2023_cross", subject=1, resample=200)

    print(data)
