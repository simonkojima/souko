import mne
import json
from pathlib import Path
import numpy as np

from .utils import get_proc_name
from transfer_bci.euclidean import euclidean_alignment


def preprocess_raw(raw, l_freq, h_freq):
    raw.load_data()
    raw.pick(picks="eeg")
    raw.filter(l_freq=l_freq, h_freq=h_freq)
    return raw


def _save_data(X, y, save_base, subject, session_num, run_num):
    np.save(
        save_base / f"sub-{subject}_ses-{session_num}_run-{run_num}_X.npy",
        X,
    )

    np.save(
        save_base / f"sub-{subject}_ses-{session_num}_run-{run_num}_y.npy",
        y,
    )


def export_meta_data(
        save_base,
        subject,
        session_num,
        raw,
        resample,
        times,
        event_id,
):
    if isinstance(times, np.ndarray):
        times = times.tolist()

    meta_data = {
        "ch_names": raw.ch_names,
        "sfreq": raw.info["sfreq"],
        "resample": resample,
        "highpass": raw.info["highpass"],
        "lowpass": raw.info["lowpass"],
        "times": times,
        "event_id": event_id,
    }

    with open(save_base / f"sub-{subject}_ses-{session_num}_meta.json", "w") as f:
        json.dump(meta_data, f)


def export_data(
        dataset,
        cache_config={"use": True},
        resample=None,
        l_freq=7,
        h_freq=30,
        tmin=None,
        tmax=None,
        event_id="auto",
        save_base=None,
):
    dataset_name = dataset.__class__.__name__

    if save_base is None:
        save_base = Path.home() / "datasets" / dataset_name
    else:
        save_base = Path(save_base) / dataset_name

    if tmin is None:
        tmin = dataset.interval[0]
    if tmax is None:
        tmax = dataset.interval[1]

    for subject in dataset.subject_list:
        data = dataset.get_data(
            subjects=[subject],
            cache_config=cache_config,
        )[subject]
        for session_idx, (session_name, session_data) in enumerate(data.items()):
            X_ses, y_ses = [], []
            for run_idx, (run_name, run_raw) in enumerate(session_data.items()):

                save_base_session = (
                        save_base
                        / f"sub-{subject}"
                        / f"ses-{session_idx + 1}"
                        / get_proc_name(
                    resample=resample,
                    tmin=tmin,
                    tmax=tmax,
                    l_freq=l_freq,
                    h_freq=h_freq,
                )
                )

                save_base_session.mkdir(parents=True, exist_ok=True)

                print(
                    f"Exporting data for subject {subject}, session {session_idx + 1}, run {run_idx + 1}"
                )
                run_raw = preprocess_raw(run_raw, l_freq=l_freq, h_freq=h_freq)

                events, event_id = mne.events_from_annotations(
                    raw=run_raw,
                    event_id=event_id,
                )

                epochs = mne.Epochs(
                    run_raw,
                    events=events,
                    event_id=event_id,
                    tmin=tmin,
                    tmax=tmax,
                    baseline=None,
                )

                if resample is not None:
                    epochs.load_data()
                    epochs = epochs.resample(resample)

                epochs = epochs.crop(tmin=tmin, tmax=tmax)

                X = epochs.get_data()
                y = epochs.events[:, 2]

                X_ses.append(X)
                y_ses.append(y)

                times = epochs.times

                _save_data(
                    X=X,
                    y=y,
                    save_base=save_base_session,
                    subject=subject,
                    session_num=session_idx + 1,
                    run_num=run_idx + 1,
                )

                export_meta_data(
                    save_base=save_base_session,
                    subject=subject,
                    session_num=session_idx + 1,
                    raw=run_raw,
                    resample=resample,
                    times=times,
                    event_id=event_id,
                )

            X = np.concatenate(X_ses, axis=0)
            y = np.concatenate(y_ses, axis=0)

            np.save(
                save_base_session / f"sub-{subject}_ses-{session_idx + 1}_X.npy",
                X,
            )

            np.save(
                save_base_session / f"sub-{subject}_ses-{session_idx + 1}_y.npy",
                y,
            )

            X_ea = euclidean_alignment(X)
            np.save(
                save_base_session / f"sub-{subject}_ses-{session_idx + 1}_X_ea.npy",
                X_ea,
            )

            X_ea_online = euclidean_alignment(X, online=True)
            np.save(
                save_base_session / f"sub-{subject}_ses-{session_idx + 1}_X_ea_online.npy",
                X_ea_online,
            )


if __name__ == "__main__":
    mne.set_log_level("CRITICAL")

    from moabb.datasets import Dreyer2023

    dataset = Dreyer2023()

    export_data(
        dataset,
        resample=128,
        tmin=0.5,
        tmax=5,
        baseline=None,
        event_id={"left_hand": 0, "right_hand": 1},
    )
