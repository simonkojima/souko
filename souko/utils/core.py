import argparse
import re
import tag_mne as tm
import numpy as np


def natural_key(s: str):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", s)]


def type_params(params):

    keys_float = [
        "l_freq",
        "h_freq",
        "tmin",
        "tmax",
        "baseline",
        "tmin_epochs",
        "tmax_epochs",
    ]
    keys_int = ["order", "resample"]

    keys = list(params.keys())

    for key in keys_float:
        if key in keys:
            if params[key] is not None:
                params[key] = float(params[key])

    for key in keys_int:
        if key in keys:
            if params[key] is not None:
                params[key] = int(params[key])

    return params


def encode_params(params):

    params = type_params(params)

    if isinstance(params, argparse.Namespace):
        params_dict = vars(params)
    else:
        params_dict = params

    sorted_params = dict(sorted(params_dict.items(), key=lambda x: x[0]))
    proc_id = []
    for k, v in sorted_params.items():
        proc_id.append(f"{k}-{v}")
    proc_id = "_".join(proc_id)

    return proc_id


def get_labels_from_epochs(epochs, label_keys={"event:left": 0, "event:right": 1}):
    y = list()

    _, markers = tm.markers_from_events(epochs.events, epochs.event_id)

    for marker in markers:
        for key, val in label_keys.items():
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
