import mne
import json
from pathlib import Path
import numpy as np
from mne_bids import get_entity_vals

from .utils import expand_variables
from .load import read_data


def preprocess_raw(raw, l_freq, h_freq, f_order, phase, picks, n_jobs=None):
    raw.pick(picks=picks)
    raw.filter(
        l_freq=l_freq,
        h_freq=h_freq,
        method="iir",
        phase=phase,
        iir_params={"ftype": "butter", "btype": "bandpass", "order": f_order},
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
    alignment=None,
    kwargs_alignment={},
    name="${NAME}",
    n_jobs=None,
    name_cache=None,
    mmap_mode=False,
):

    if cache_config is None:
        cache_config = {"use": True}

    dataset_name = dataset.__class__.__name__

    if name == "":
        raise ValueError("Name can not be empty.")

    name_dir = expand_variables(name, {"NAME": dataset_name})

    if save_base is None:
        save_base = Path.home() / "datasets"
    else:
        save_base = Path(save_base)

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
                save_base / name_cache / f"sub-{subject}", "session"
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
                    run_manifest.append(run_name)

                    run_raw.load_data()

                    run_raw = preprocess_raw(
                        run_raw,
                        l_freq=l_freq,
                        h_freq=h_freq,
                        f_order=f_order,
                        phase=phase,
                        picks=picks,
                        n_jobs=n_jobs,
                    )

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
                    save_base, name_cache, subject, session_idx + 1, mmap_mode=mmap_mode
                )

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
                },
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

    dataset = Dreyer2023()

    souko.xy.export_data(
        dataset,
        resample=128,
        tmin=0.5,
        tmax=5,
        event_id={"left_hand": 0, "right_hand": 1},
        name="${NAME}_test",
    )
