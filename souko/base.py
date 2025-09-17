import os
import pandas as pd
import functools

from . import utils
import mne
import msgpack
import msgpack_numpy as m

import numpy as np


def save_mne_objs(data, save_base, suffix):
    files = {"session": [], "run": [], "fname": []}
    for session, session_data in data.items():
        for run, obj_run in session_data.items():

            fname = f"{session}_{run}{suffix}"

            files["session"].append(session)
            files["run"].append(run)
            files["fname"].append(fname)

            obj_run.save(save_base / fname, overwrite=True)

    df = pd.DataFrame(files)
    df.to_csv(save_base / "files.tsv", sep="\t", index=False)


def load_mne_objs(base, suffix, func_load):
    df = pd.read_csv(base / "files.tsv", sep="\t")

    sessions = df["session"].tolist()
    runs = df["run"].tolist()
    fnames = df["fname"].tolist()

    data = {session: {} for session in sessions}

    for session, run, fname in zip(sessions, runs, fnames):
        data[session][run] = func_load(base / fname)

    return data


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

        cache = kwargs.get("cache", True)
        force_update = kwargs.get("force_update", False)
        concat_runs = kwargs.get("concat_runs", False)
        concat_sessions = kwargs.get("concat_sessions", False)

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
                    data = func_load_data(save_base, suffix)
            else:
                data = func_get_data(subject, params)
                save_base.mkdir(parents=True, exist_ok=True)

                if func_save_data is not None:
                    func_save_data(data, save_base, suffix)
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

    def _get_covs(self, subject, params):
        l_freq = params["l_freq"]
        h_freq = params["h_freq"]
        order = params["order"]
        tmin_epochs = params["tmin_epochs"]
        tmax_epochs = params["tmax_epochs"]
        baseline = params["baseline"]
        resample = params["resample"]
        picks = params["picks"]
        tmin = params["tmin"]
        tmax = params["tmax"]
        estimator = params["estimator"]

        epochs = self.get_epochs(
            subject=subject,
            l_freq=l_freq,
            h_freq=h_freq,
            order=order,
            tmin=tmin_epochs,
            tmax=tmax_epochs,
            baseline=baseline,
            resample=resample,
        )

        import pyriemann

        data = {}
        for k, e in epochs[1].items():
            e = e.pick(picks=picks).crop(tmin=tmin, tmax=tmax)
            covs = pyriemann.estimation.Covariances(estimator=estimator).fit_transform(
                e.get_data()
            )

            data[k] = covs

        data = {1: data}

        return data

    def _get_covs_rpa(self, subject, params, cache=True, force_update=False):

        tmin = params["tmin"]
        tmax = params["tmax"]
        picks = params["picks"]
        l_freq = params["l_freq"]
        h_freq = params["h_freq"]
        order = params["order"]
        baseline = params["baseline"]
        resample = params["resample"]
        tmin_epochs = params["tmin_epochs"]
        tmax_epochs = params["tmax_epochs"]
        estimator = params["estimator"]

        rescaling = params["rescaling"]
        online_rpa = params["online_rpa"]

        if online_rpa:
            pass
            # raise NotImplementedError("rpa_online is not implemented")

        covs = self.get_covs(
            subject=subject,
            tmin=tmin,
            tmax=tmax,
            picks=picks,
            l_freq=l_freq,
            h_freq=h_freq,
            order=order,
            baseline=baseline,
            resample=resample,
            tmin_epochs=tmin_epochs,
            tmax_epochs=tmax_epochs,
            estimator=estimator,
            cache=cache,
            force_update=force_update,
            concat_runs=True,
            concat_sessions=False,
        )

        from rosoku.tl import riemannian_alignment

        for session in self.sessions_list:
            covs[session] = riemannian_alignment(
                covs[session], scaling=rescaling, online=online_rpa
            )

        return covs

    def get_covs(
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
        recentering=False,
        rescaling=False,
        online_rpa=False,
    ):

        if recentering is False:
            data_type = "covs"
            params = dict(
                tmin=tmin,
                tmax=tmax,
                picks=picks,
                l_freq=l_freq,
                h_freq=h_freq,
                order=order,
                baseline=baseline,
                resample=resample,
                tmin_epochs=tmin_epochs,
                tmax_epochs=tmax_epochs,
                estimator=estimator,
            )

            data = self._get_data(
                subject,
                data_type,
                params,
                func_get_data=self._get_covs,
                cache=cache,
                force_update=force_update,
                concat_runs=concat_runs,
                concat_sessions=concat_sessions,
            )
        else:
            data_type = "covs-rpa"

            params = dict(
                tmin=tmin,
                tmax=tmax,
                picks=picks,
                l_freq=l_freq,
                h_freq=h_freq,
                order=order,
                baseline=baseline,
                resample=resample,
                tmin_epochs=tmin_epochs,
                tmax_epochs=tmax_epochs,
                estimator=estimator,
                rescaling=rescaling,
                online_rpa=online_rpa,
            )

            data = self._get_data(
                subject,
                data_type,
                params,
                func_get_data=functools.partial(
                    self._get_covs_rpa, cache=cache, force_update=force_update
                ),
                cache=cache,
                force_update=force_update,
            )

        return data

    def _get_labels(self, subject, params):

        params_epochs = utils.decode_params(self.get_proc_list("epochs")[-1])

        l_freq = params_epochs["l_freq"]
        h_freq = params_epochs["h_freq"]
        order = params_epochs["order"]
        tmin_epochs = params_epochs["tmin"]
        tmax_epochs = params_epochs["tmax"]
        baseline = params_epochs["baseline"]
        resample = params_epochs["resample"]

        label_keys = params["label_keys"]

        epochs = self.get_epochs(
            subject=subject,
            l_freq=l_freq,
            h_freq=h_freq,
            order=order,
            tmin=tmin_epochs,
            tmax=tmax_epochs,
            baseline=baseline,
            resample=resample,
        )

        labels = {1: {}}

        for k, e in epochs[1].items():
            labels[1][k] = utils.get_labels_from_epochs(e, label_keys)

        return labels

    def get_labels(
        self,
        subject,
        label_keys={"event:left": 0, "event:right": 1},
        cache=True,
        force_update=False,
        concat_runs=False,
        concat_sessions=False,
    ):
        data_type = "labels"
        params = dict(label_keys=label_keys)
        data = self._get_data(
            subject,
            data_type,
            params,
            func_get_data=self._get_labels,
            cache=cache,
            force_update=force_update,
            concat_runs=concat_runs,
            concat_sessions=concat_sessions,
        )

        return data

    def _get_tfrs(self, subject, params, n_jobs):

        l_freq = params["l_freq"]
        h_freq = params["h_freq"]
        tmin_epochs = params["tmin_epochs"]
        tmax_epochs = params["tmax_epochs"]
        order = params["order"]
        baseline = params["baseline"]
        resample = params["resample"]

        method = params["method"]
        freqs = params["freqs"]
        n_cycles = params["n_cycles"]
        use_fft = params["use_fft"]
        decim = params["decim"]

        epochs = self.get_epochs(
            subject=subject,
            l_freq=l_freq,
            h_freq=h_freq,
            order=order,
            tmin=tmin_epochs,
            tmax=tmax_epochs,
            baseline=baseline,
            resample=resample,
        )

        if isinstance(freqs, range):
            freqs = list(freqs)

        if isinstance(n_cycles, range):
            n_cycles = list(n_cycles)

        tfrs = {}
        for session in self.sessions_list:
            tfrs[session] = {}
            for run, e in epochs[session].items():
                t = e.compute_tfr(
                    method=method,
                    freqs=freqs,
                    n_cycles=n_cycles,
                    use_fft=use_fft,
                    return_itc=False,
                    average=False,
                    decim=decim,
                    n_jobs=n_jobs,
                )

                tfrs[session][run] = t

        return tfrs

    def get_tfrs(
        self,
        subject,
        tmin_epochs,
        tmax_epochs,
        l_freq,
        h_freq,
        order,
        baseline,
        resample,
        method,
        freqs,
        n_cycles,
        use_fft,
        decim,
        n_jobs=-1,
        cache=True,
        force_update=False,
    ):
        data_type = "tfrs"
        suffix = "-tfr.hdf5"

        params = dict(
            tmin_epochs=tmin_epochs,
            tmax_epochs=tmax_epochs,
            l_freq=l_freq,
            h_freq=h_freq,
            order=order,
            baseline=baseline,
            resample=resample,
            method=method,
            freqs=freqs,
            n_cycles=n_cycles,
            use_fft=use_fft,
            decim=decim,
        )

        data = self._get_data(
            subject,
            data_type=data_type,
            params=params,
            suffix=suffix,
            func_get_data=functools.partial(self._get_tfrs, n_jobs=n_jobs),
            func_save_data=save_mne_objs,
            func_load_data=functools.partial(
                load_mne_objs, func_load=mne.time_frequency.read_tfrs
            ),
            cache=cache,
            force_update=force_update,
        )

        return data
