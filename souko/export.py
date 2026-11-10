import mne
import json
from pathlib import Path
import numpy as np
from mne_bids import get_entity_vals
import warnings
from pprint import pprint
from copy import deepcopy

from .utils import expand_variables
from .load import read_data


def preprocess_raw(raw, picks, l_freq, h_freq, kwargs_filter):
    raw.pick(picks=picks)
    raw.filter(l_freq=l_freq, h_freq=h_freq, **kwargs_filter)
    return raw


def pick_channels(X, meta, picks):

    if isinstance(picks, list) and len(picks) == 1:
        picks = picks[0]

    if picks == "eeg":
        warnings.warn(
            f"When using cache (name_cache is not None), picks='eeg' is ignored",
        )

        return X

    ch_names = meta["ch_names"]

    if len(set(ch_names) & set(picks)) != len(set(picks)):
        raise ValueError(f"{picks} not in {ch_names}")

    idx = [ch_names.index(pick) for pick in picks]
    new_ch_names = [ch_names[i] for i in idx]

    new_X = X[:, idx, :]
    meta["ch_names"] = new_ch_names

    return new_X, meta


def crop(X, meta, tmin, tmax):

    times = np.asarray(meta["times"])

    idx_min = np.argmin(np.abs(times - tmin))
    idx_max = np.argmin(np.abs(times - tmax))

    sfreq = meta["sfreq"]

    if abs(times[idx_min] - tmin) > 1 / sfreq:
        raise ValueError(f"tmin={tmin} is too far from nearest time={times[idx_min]}")

    if abs(times[idx_max] - tmax) > 1 / sfreq:
        raise ValueError(f"tmax={tmax} is too far from nearest time={times[idx_max]}")

    X = X[:, :, idx_min : idx_max + 1]
    meta["times"] = times[idx_min : idx_max + 1].tolist()

    return X, meta


def _save_data(X, y, save_base, subject, session_num, run_num):
    np.save(
        save_base / f"sub-{subject}_ses-{session_num}_run-{run_num}_X.npy",
        X,
    )

    np.save(
        save_base / f"sub-{subject}_ses-{session_num}_run-{run_num}_y.npy",
        y,
    )


def export_data(
    dataset,
    cache_config=None,
    resample=None,
    picks="eeg",
    l_freq=7,
    h_freq=30,
    kwargs_filter={},
    tmin=None,
    tmax=None,
    event_id="auto",
    save_base=None,
    alignment=None,
    kwargs_alignment={},
    name="${NAME}",
    n_jobs=None,
    name_cache=None,
    cache_base="save",
    mmap_mode=False,
    event_not_found="warning",
):

    if cache_config is None:
        cache_config = {"use": True}

    dataset_name = dataset.__class__.__name__

    if name == "":
        raise ValueError("Name can not be empty.")

    name_dir = expand_variables(name, {"NAME": dataset_name})

    if name_cache is not None:
        name_cache = expand_variables(name_cache, {"NAME": dataset_name})

    if save_base is None:
        save_base = Path.home() / "datasets"
    else:
        save_base = Path(save_base)

    if cache_base == "save":
        cache_base = save_base

    if tmin is None:
        tmin = dataset.interval[0]
    if tmax is None:
        tmax = dataset.interval[1]

    for subject in dataset.subject_list:
        if name_cache is None:
            data = dataset.get_data(
                subjects=[subject],
                cache_config=cache_config,
            )[subject]
            sessions = list(range(1, len(data) + 1))
        else:
            sessions = get_entity_vals(
                cache_base / name_cache / f"sub-{subject}", "session"
            )

        for session_idx, (session_name) in enumerate(sessions):

            save_base_session = (
                save_base / name_dir / f"sub-{subject}" / f"ses-{session_idx + 1}"
            )

            save_base_session.mkdir(parents=True, exist_ok=True)

            if name_cache is None:
                session_data = list(data.values())[session_idx]
                if not session_data:
                    raise ValueError(f"Empty session: {session_name}")
                X_ses, y_ses = [], []
                run_manifest = []
                n_samples = []
                for run_idx, (run_name, run_raw) in enumerate(session_data.items()):

                    run_raw.load_data()

                    run_raw = preprocess_raw(
                        run_raw,
                        l_freq=l_freq,
                        h_freq=h_freq,
                        picks=picks,
                        kwargs_filter=kwargs_filter,
                    )

                    _events, _event_id = mne.events_from_annotations(run_raw)

                    event_id_raw = list(_event_id.keys())

                    if len(set(event_id_raw) & set(event_id)) != len(event_id):
                        if event_not_found == "warning":
                            warnings.warn(
                                f"Requested events {event_id} is not contained in this run {_event_id}",
                                RuntimeWarning,
                            )
                            continue
                        elif event_not_found == "raise":
                            raise RuntimeError(
                                f"Requested events {event_id} is not contained in this run {_event_id}",
                            )
                        else:
                            raise ValueError(
                                f"event_not_found: {event_not_found} is unknown"
                            )

                    run_manifest.append(run_name)
                    events, run_event_id = mne.events_from_annotations(
                        raw=run_raw,
                        event_id=event_id,
                    )

                    epochs = mne.Epochs(
                        run_raw,
                        events=events,
                        event_id=run_event_id,
                        tmin=tmin,
                        tmax=tmax,
                        baseline=None,
                    )

                    if resample is not None:
                        epochs.load_data()
                        epochs = epochs.resample(resample, n_jobs=n_jobs)

                    epochs = epochs.crop(tmin=tmin, tmax=tmax)
                    times = epochs.times

                    X = epochs.get_data()
                    y = epochs.events[:, 2]

                    X_ses.append(X)
                    y_ses.append(y)

                    n_samples.append(y.size)

                X = np.concatenate(X_ses, axis=0)
                y = np.concatenate(y_ses, axis=0)

                sfreq = epochs.info["sfreq"]
                highpass = epochs.info["highpass"]
                lowpass = epochs.info["lowpass"]
                if isinstance(times, np.ndarray):
                    times = times.tolist()
                runs = run_manifest
                ch_names = epochs.ch_names
            else:

                X, y, meta = read_data(
                    cache_base,
                    name_cache,
                    subject,
                    session_idx + 1,
                    mmap_mode=mmap_mode,
                )

                preprocessing_dict = {
                    "resample": resample,
                    "l_freq": l_freq,
                    "h_freq": h_freq,
                    "kwargs_filter": kwargs_filter,
                }

                meta_preprocessing_dict = deepcopy(meta["preprocessing"])
                meta_preprocessing_dict.pop("tmin")
                meta_preprocessing_dict.pop("tmax")

                if meta["dataset"] != dataset_name:
                    raise ValueError(
                        f"dataset name for cache {meta['dataset']} is not identical to {dataset_name}"
                    )

                if meta_preprocessing_dict != preprocessing_dict:
                    raise ValueError(
                        f"preprocessing parameters are not compatible: {meta_preprocessing_dict}"
                    )

                X, meta = pick_channels(X, meta, picks=picks)
                X, meta = crop(X, meta, tmin, tmax)

                sfreq = meta["sfreq"]
                highpass = meta["highpass"]
                lowpass = meta["lowpass"]
                times = meta["times"]
                runs = meta["runs"]
                n_samples = meta["n_samples_per_run"]
                ch_names = meta["ch_names"]

            print(f"Exporting data for subject {subject}, session {session_idx + 1}")

            if alignment is not None:
                X = alignment(X, **kwargs_alignment)

            np.save(
                save_base_session / f"sub-{subject}_ses-{session_idx + 1}_X.npy",
                X,
            )

            np.save(
                save_base_session / f"sub-{subject}_ses-{session_idx + 1}_y.npy",
                y,
            )

            if alignment is not None:
                alignment_name = f"{alignment.__module__}.{alignment.__qualname__}"
            else:
                alignment_name = None

            meta_data = {
                "sfreq": sfreq,
                "resample": resample,
                "highpass": highpass,
                "lowpass": lowpass,
                "event_id": event_id,
                "dataset": dataset_name,
                "subject": subject,
                "session": session_idx + 1,
                "session_name": str(session_name),
                "runs": runs,
                "n_samples_per_run": n_samples,
                "alignment": alignment_name,
                "kwargs_alignment": kwargs_alignment,
                "preprocessing": {
                    "resample": resample,
                    "tmin": tmin,
                    "tmax": tmax,
                    "l_freq": l_freq,
                    "h_freq": h_freq,
                    "kwargs_filter": kwargs_filter,
                },
                "name": name,
                "name_cache": name_cache,
                "mmap_mode": mmap_mode,
                "ch_names": ch_names,
                "times": times,
            }

            with open(
                save_base_session / f"sub-{subject}_ses-{session_idx+1}_meta.json", "w"
            ) as f:
                json.dump(meta_data, f, indent=4)


if __name__ == "__main__":
    mne.set_log_level("CRITICAL")

    from moabb.datasets import Dreyer2023
    import souko
    from pathlib import Path

    dataset = Dreyer2023()

    souko.xy.export_data(
        dataset,
        resample=128,
        tmin=0.5,
        tmax=5,
        event_id={"left_hand": 0, "right_hand": 1},
        name="${NAME}_test",
        name_cache="{NAME}",
        cache_base=Path.home() / "datasets" / "0.5-40",
        save_base=Path.home() / "debug" / "souko",
    )
