import argparse
import hashlib
import json
import math

import pandas as pd


def type_params(params):
    keys_float = [
        "l_freq",
        "h_freq",
        "tmin",
        "tmax",
        "tmin_epochs",
        "tmax_epochs",
    ]
    keys_int = ["order", "resample"]
    keys_float_in_list = ["baseline"]

    keys_list = ["freqs", "n_cycles", "class_list"]

    keys = list(params.keys())

    for key in keys_float:
        if key in keys:
            if params[key] is not None:
                params[key] = float(params[key])

    for key in keys_float_in_list:
        if key in keys:
            if params[key] is not None:
                for idx in range(len(params[key])):
                    params[key][idx] = float(params[key][idx])

    for key in keys_int:
        if key in keys:
            if params[key] is not None:
                params[key] = int(params[key])

    for key in keys_list:
        if key in keys:
            if params[key] is not None:
                params[key] = list(params[key])

    if "iir_params" in keys:
        params["iir_params"] = type_params(params["iir_params"])

    return params


def sha256_short(s: str, n: int = 12):
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:n]


def deep_sort(obj):
    if isinstance(obj, dict):
        return {k: deep_sort(v) for k, v in sorted(obj.items(), key=lambda x: x[0])}
    elif isinstance(obj, list):
        return [deep_sort(v) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(deep_sort(v) for v in obj)
    else:
        return obj


def encode_params(params):
    params = type_params(params)

    if isinstance(params, argparse.Namespace):
        params_dict = vars(params)
    else:
        params_dict = params

    from ..__init__ import __version__

    sorted_params = deep_sort(params_dict)

    """
    sorted_params = dict(sorted(params_dict.items(), key=lambda x: x[0]))
    while True:
        for k, v in sorted_params.items():
            if isinstance(v, dict):
                sorted_params[k] = dict(sorted(v.items(), key=lambda x: x[0]))
    """

    hash = sha256_short(json.dumps(sorted_params))

    return hash, sorted_params


def _is_nan(x):
    try:
        return math.isnan(x)
    except Exception:
        return False


def nan_to_none(obj):
    if isinstance(obj, dict):
        return {k: nan_to_none(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [nan_to_none(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(nan_to_none(v) for v in obj)
    if _is_nan(obj):
        return None
    return obj


def decode_params(series):
    print(type(series))

    if not isinstance(series, pd.Series):
        raise TypeError("series must be pd.Series.")

    params = series.to_dict()
    params = nan_to_none(params)
    print(params)

    raise NotImplementedError()
    exit()

    return params
