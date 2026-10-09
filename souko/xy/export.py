import mne
import json
from pathlib import Path
import numpy as np

from .utils import get_proc_name, expand_variables


def preprocess_raw(raw, l_freq, h_freq, f_order, phase, picks, n_jobs=None):
    raw.load_data()
    raw.pick(picks=picks)
    raw.filter(
        l_freq=l_freq,
        h_freq=h_freq,
        method="iir",
        phase=phase,
        iir_params={"ftype": "butter", "btype": "bandpass", "order": 4},
        n_jobs=n_jobs,
    )
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
    cache_config=None,
    resample=None,
    picks="eeg",
    l_freq=7,
    h_freq=30,
    f_order=4,
    phase="zero",
    tmin=None,
    tmax=None,
    event_id="auto",
    save_base=None,
    ea=True,
    cn=True,
    online=True,
    name="${NAME}",
    n_jobs=None,
):
    """Export runs plus cached session EA representations.

    save_base is the dataset root (default ~/datasets). EA is fitted across
    the complete session; online EA is delegated to transfer_bci. Neither
    cached representation is refitted when loaders split the trials.
    """

    if cache_config is None:
        cache_config = {"use": True}
    if ea:
        from transfer_bci.euclidean import euclidean_alignment
    if cn:
        from transfer_bci.euclidean import channel_normalization
    dataset_name = dataset.__class__.__name__

    if name == "":
        raise ValueError("Name can not be empty.")

    name_dir = expand_variables(name, {"NAME": dataset_name})

    if save_base is None:
        save_base = Path.home() / "datasets" / name_dir
    else:
        save_base = Path(save_base) / name_dir

    if tmin is None:
        tmin = dataset.interval[0]
    if tmax is None:
        tmax = dataset.interval[1]

    requested_event_id = event_id
    for subject in dataset.subject_list:
        data = dataset.get_data(
            subjects=[subject],
            cache_config=cache_config,
        )[subject]
        for session_idx, (session_name, session_data) in enumerate(data.items()):
            if not session_data:
                raise ValueError(f"Empty session: {session_name}")
            X_ses, y_ses = [], []
            run_manifest = []
            session_event_id = {}
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
                run_raw = preprocess_raw(
                    run_raw.copy(),
                    l_freq=l_freq,
                    h_freq=h_freq,
                    f_order=f_order,
                    phase=phase,
                    picks=picks,
                    n_jobs=n_jobs,
                )

                events, run_event_id = mne.events_from_annotations(
                    raw=run_raw,
                    event_id=requested_event_id,
                )

                for label, code in run_event_id.items():
                    if label in session_event_id and session_event_id[label] != code:
                        raise ValueError(f"Inconsistent event code for {label}")
                    if any(
                        other != label and value == code
                        for other, value in session_event_id.items()
                    ):
                        raise ValueError(f"Event code {code} has inconsistent labels")
                    session_event_id[label] = code

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

                X = epochs.get_data()
                y = epochs.events[:, 2]

                X_ses.append(X)
                y_ses.append(y)

                times = epochs.times
                np.save(
                    save_base_session
                    / f"sub-{subject}_ses-{session_idx + 1}_run-{run_idx + 1}_samples.npy",
                    epochs.events[:, 0],
                )
                run_manifest.append(
                    {"id": run_idx + 1, "name": str(run_name), "n_trials": len(y)}
                )

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
                    event_id=run_event_id,
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

            for suffix, enabled, is_online in (
                ("ea", ea, False),
                ("ea_online", ea and online, True),
            ):
                filename = (
                    save_base_session
                    / f"sub-{subject}_ses-{session_idx + 1}_X_{suffix}.npy"
                )
                if enabled:
                    aligned = (
                        euclidean_alignment(X, online=True)
                        if is_online
                        else euclidean_alignment(X)
                    )
                    np.save(filename, aligned)
                else:
                    filename.unlink(missing_ok=True)

            for suffix, enabled, is_online in (
                ("cn", cn, False),
                ("cn_online", cn and online, True),
            ):
                filename = (
                    save_base_session
                    / f"sub-{subject}_ses-{session_idx + 1}_X_{suffix}.npy"
                )
                if enabled:
                    aligned = (
                        channel_normalization(X, online=True)
                        if is_online
                        else channel_normalization(X)
                    )
                    np.save(filename, aligned)
                else:
                    filename.unlink(missing_ok=True)

            metadata_path = (
                save_base_session / f"sub-{subject}_ses-{session_idx + 1}_meta.json"
            )
            with open(metadata_path) as stream:
                metadata = json.load(stream)
            metadata.update(
                {
                    "schema_version": 1,
                    "dataset": dataset_name,
                    "sfreq": float(epochs.info["sfreq"]),
                    "subject": subject,
                    "session": session_idx + 1,
                    "session_name": str(session_name),
                    "runs": run_manifest,
                    "event_id": session_event_id,
                    "preprocessing": {
                        "resample": resample,
                        "tmin": tmin,
                        "tmax": tmax,
                        "l_freq": l_freq,
                        "h_freq": h_freq,
                    },
                }
            )
            with open(metadata_path, "w") as stream:
                json.dump(metadata, stream)


if __name__ == "__main__":
    mne.set_log_level("CRITICAL")

    from moabb.datasets import Dreyer2023
    import souko

    dataset = Dreyer2023()

    souko.xy.export_data(
        dataset,
        resample=128,
        tmin=0.5,
        tmax=5,
        event_id={"left_hand": 0, "right_hand": 1},
    )
