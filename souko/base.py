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

    def _get_covs(self, subject):
        raise NotImplementedError()

    def _check_subject(self, subject):
        if isinstance(subject, int) is False:
            raise ValueError("subject must be int.")
        if subject not in self.subjects_list:
            raise ValueError(f"Invalid subject: {subject}")

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

    def _concat_covs(self, data, mode):
        match mode:
            case "runs":
                new_data = {}
                for session in self.sessions_list:
                    covs_concat = []
                    y_concat = []
                    for run in list(data[session].keys()):
                        covs_concat.append(data[session][run]["covs"])
                        y_concat.append(data[session][run]["y"])
                    covs_concat = np.concatenate(covs_concat, axis=0)
                    y_concat = np.concatenate(y_concat, axis=0)

                    new_data[session] = {}
                    new_data[session]["covs"] = covs_concat
                    new_data[session]["y"] = y_concat
            case "sessions":
                new_data = {}
                covs_concat = []
                y_concat = []

                for session in self.sessions_list:
                    covs_concat.append(data[session]["covs"])
                    y_concat.append(data[session]["y"])
                covs_concat = np.concatenate(covs_concat, axis=0)
                y_concat = np.concatenate(y_concat, axis=0)

                new_data["covs"] = covs_concat
                new_data["y"] = y_concat

        return new_data

    def ___get_covs(
        self,
        subject,
        tmin,
        tmax,
        picks="eeg",
        l_freq=8,
        h_freq=30,
        order=4,
        baseline=None,
        resample=128,
        tmin_epochs=-5.0,
        tmax_epochs=7.0,
        estimator="scm",
        cache=True,
        force_update=False,
        concat_runs=False,
        concat_sessions=False,
        **kwargs,
    ):

        self._check_subject(subject)

        if cache is True:
            params = utils.encode_params(
                dict(
                    tmin=tmin,
                    tmax=tmax,
                    tmin_epochs=tmin_epochs,
                    tmax_epochs=tmax_epochs,
                    l_freq=l_freq,
                    h_freq=h_freq,
                    order=order,
                    baseline=baseline,
                    resample=resample,
                    picks=picks,
                    estimator=estimator,
                )
            )

            covs_base = (
                self.base_dir / "derivatives" / "covs" / params / f"sub-{subject}"
            )

            files = check_files(covs_base, "-covs.npy")

            if (files is not None) and (force_update is False):
                data = {}
                for file in files:

                    files_meta = pd.read_csv(covs_base / "files.tsv", sep="\t")
                    sessions = files_meta["session"].tolist()
                    runs = files_meta["run"].tolist()

                    for session in sessions:
                        data = {session: {}}
                        for run in runs:
                            data[session][run] = {}

                    fnames_covs = files_meta["fname_covs"].tolist()
                    fnames_y = files_meta["fname_y"].tolist()

                    for session, run, fname_covs, fname_y in zip(
                        sessions, runs, fnames_covs, fnames_y
                    ):
                        data[session][run]["covs"] = np.load(covs_base / fname_covs)
                        data[session][run]["y"] = np.load(covs_base / fname_y)
            else:
                data = self._get_covs(
                    subject,
                    tmin=tmin,
                    tmax=tmax,
                    picks=picks,
                    l_freq=l_freq,
                    h_freq=h_freq,
                    order=order,
                    tmin_epochs=tmin_epochs,
                    tmax_epochs=tmax_epochs,
                    baseline=baseline,
                    resample=resample,
                    **kwargs,
                )

                covs_base.mkdir(parents=True, exist_ok=True)

                files_meta = {"session": [], "run": [], "fname_covs": [], "fname_y": []}
                for session in self.sessions_list:
                    for run, d in data[session].items():
                        np.save(covs_base / f"{session}_{run}-covs.npy", d["covs"])
                        np.save(covs_base / f"{session}_{run}-y.npy", d["y"])
                        files_meta["session"].append(session)
                        files_meta["run"].append(run)
                        files_meta["fname_covs"].append(f"{session}_{run}-covs.npy")
                        files_meta["fname_y"].append(f"{session}_{run}-y.npy")

                files_meta = pd.DataFrame(files_meta)
                files_meta.to_csv(covs_base / "files.tsv", sep="\t", index=False)

        else:

            data = self._get_covs(
                subject,
                tmin=tmin,
                tmax=tmax,
                picks=picks,
                l_freq=l_freq,
                h_freq=h_freq,
                order=order,
                tmin_epochs=tmin_epochs,
                tmax_epochs=tmax_epochs,
                baseline=baseline,
                resample=resample,
                **kwargs,
            )

        if concat_runs:
            new_data = self._concat_covs(data, "runs")
            data = new_data
            if concat_sessions:
                new_data = self._concat_covs(data, "sessions")
                data = new_data

        return data

    def _get_data(
        self,
        subject,
        data_type,
        params,
        suffix=".msgpack",
        func_get_data=None,
        func_save_data=None,
        func_load_data=None,
        cache=True,
        force_update=False,
    ):
        self._check_subject(subject)
        m.patch()

        if cache is True:
            proc_id = utils.encode_params(params)

            save_base = (
                self.base_dir / "derivatives" / data_type / proc_id / f"sub-{subject}"
            )

            files = check_files(save_base, suffix)

            if (files is not None) and (force_update is False):
                if func_load_data is not None:
                    with open(save_base / f"sub-{subject}.msgpack", "rb") as f:
                        data = msgpack.load(f, strict_map_key=False)
                else:
                    data = func_load_data(save_base)
            else:
                data = func_get_data(subject, params)

                if func_save_data is not None:
                    func_save_data(data, save_base)
                else:
                    with open(save_base / f"sub-{subject}.msgpack", "wb") as f:
                        msgpack.dump(data, f)

        else:
            data = func_get_data(subject, params)

        return data
