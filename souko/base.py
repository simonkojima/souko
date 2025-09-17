import os
import pandas as pd

from . import utils
import mne
import msgpack
import msgpack_numpy as m

import numpy as np


def check_files(dir, suffix):
    if os.path.exists(dir):
        files = sorted(os.listdir(dir), key=utils.natural_key)
        files_load = []

        for file in files:
            if file.endswith(suffix):
                files_load.append(file)

        if len(files_load) > 0:
            return files_load
        else:
            return None

    else:
        return None


class BaseDataset:
    def __init__(self, base_dir, subjects_list, sessions_list):
        self.base_dir = base_dir
        self.subjects_list = subjects_list
        self.sessions_list = sessions_list

    def _get_raw(self, subject):
        raise NotImplementedError()

    def _get_epochs(self, subject):
        raise NotImplementedError()

    def _check_subject(self, subject):
        if isinstance(subject, int) is False:
            raise ValueError("subject must be int.")
        if subject not in self.subjects_list:
            raise ValueError(f"Invalid subject: {subject}")

    def get_proc_list(self, data_type):
        return os.listdir(self.base_dir / "derivatives" / data_type)

    def get_raw(self, subject):
        self._check_subject(subject)
        return self._get_raw(subject)

    def get_epochs(
        self,
        subject,
        l_freq=1.0,
        h_freq=45.0,
        order=4,
        tmin=-1.0,
        tmax=2.0,
        baseline=None,
        resample=128,
        cache=True,
        force_update=False,
        concat_runs=False,
        concat_sessions=False,
    ):

        self._check_subject(subject)

        if cache is True:
            params = utils.encode_params(
                dict(
                    l_freq=l_freq,
                    h_freq=h_freq,
                    order=order,
                    tmin=tmin,
                    tmax=tmax,
                    baseline=baseline,
                    resample=resample,
                )
            )

            epochs_base = (
                self.base_dir / "derivatives" / "epochs" / params / f"sub-{subject}"
            )

            files = check_files(epochs_base, "-epo.fif")

            if (files is not None) and (force_update is False):

                files_meta = pd.read_csv(epochs_base / "files.tsv", sep="\t")
                sessions = files_meta["session"].tolist()
                epochs = {session: {} for session in list(set(sessions))}
                runs = files_meta["run"].tolist()
                fnames = files_meta["fname"].tolist()

                for session, run, fname in zip(sessions, runs, fnames):
                    epochs[session][run] = mne.read_epochs(epochs_base / fname)

            else:
                epochs = self._get_epochs(
                    subject=subject,
                    l_freq=l_freq,
                    h_freq=h_freq,
                    order=order,
                    tmin=tmin,
                    tmax=tmax,
                    baseline=baseline,
                    resample=resample,
                )

                epochs_base.mkdir(parents=True, exist_ok=True)

                files_meta = {"session": [], "run": [], "fname": []}
                for session, epochs_session in epochs.items():
                    for run, e in epochs_session.items():
                        e.save(epochs_base / f"{session}_{run}-epo.fif", overwrite=True)
                        files_meta["session"].append(session)
                        files_meta["run"].append(run)
                        files_meta["fname"].append(f"{session}_{run}-epo.fif")

                files_meta = pd.DataFrame(files_meta)
                files_meta.to_csv(epochs_base / "files.tsv", sep="\t", index=False)

        else:
            epochs = self._get_epochs(
                subject=subject,
                l_freq=l_freq,
                h_freq=h_freq,
                order=order,
                tmin=tmin,
                tmax=tmax,
                baseline=baseline,
                resample=resample,
            )

        if concat_runs:

            sessions = list(epochs.keys())

            for session in sessions:
                epochs[session] = mne.concatenate_epochs(list(epochs[session].values()))

            if concat_sessions:
                epochs = mne.concatenate_epochs(list(epochs.values()))

        return epochs

    def _get_data(
        self,
        subject,
        data_type,
        params,
        suffix=".msgpack",
        func_get_data=None,
        func_save_data=None,
        func_load_data=None,
        **kwargs,
    ):
        self._check_subject(subject)
        m.patch()

        cache = kwargs.get("cache")
        force_update = kwargs.get("force_update")
        concat_runs = kwargs.get("concat_runs")
        concat_sessions = kwargs.get("concat_sessions")

        if cache is True:
            if params is not None:
                proc_id = utils.encode_params(params)
            else:
                proc_id = "None"

            save_base = (
                self.base_dir / "derivatives" / data_type / proc_id / f"sub-{subject}"
            )

            files = check_files(save_base, suffix)

            if (files is not None) and (force_update is False):
                if func_load_data is None:
                    with open(save_base / f"sub-{subject}.msgpack", "rb") as f:
                        data = msgpack.load(f, strict_map_key=False)
                else:
                    data = func_load_data(save_base)
            else:
                data = func_get_data(subject, params)
                save_base.mkdir(parents=True, exist_ok=True)

                if func_save_data is not None:
                    func_save_data(data, save_base)
                else:
                    with open(save_base / f"sub-{subject}.msgpack", "wb") as f:
                        msgpack.dump(data, f)

        else:
            data = func_get_data(subject, params)

        if concat_runs:
            for session in self.sessions_list:
                values = list(data[session].values())
                try:
                    data[session] = np.concatenate(values, axis=0)
                except:
                    data[session] = values

            if concat_sessions:
                values = list(data.values())
                try:
                    data = np.concatenate(values, axis=0)
                except:
                    data = values

        return data
