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

    def get_participants(self):
        fname = self.base_dir / "participants.tsv"
        df = pd.read_tsv(fname, sep="\t")

        return df

    def get_manifest(self, data_type):
        fname = self.base_dir / "derivatives" / data_type / "manifest.parquet"
        manifest = pd.read_parquet(fname)

        return manifest

    def get_raw(self, subject):
        """
        Rawデータを取得する。

        Parameters
        ----------
        subject : int

        Returns
        -------
        dict of raw

        """
        self._check_subject(subject)
        return self._get_raw(subject)

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
        """
        インタフェース用関数

        Parameters
        ----------
        subject: int

        data_type: str
            ``DATASET/derivatives/{data_type}`` に該当する部分
        params: dict
            ``func_get_data`` に渡すパラメータ
        suffix: ファイルのsuffix, Default = ".msgpack"
            例えば，mne.Epochsなら ``-epo.fif`` みたいな
        func_get_data:
            データ取得用関数への参照

            .. code-block:: python

                def func_get_data(subject, params):
                    # ...
                    return data

        func_save_data: callable, default = None
            データ保存用関数

            Noneの場合はmsgpackでオブジェクト全体を保存

            .. code-block:: python

                def func_save_data(data, save_base, suffix):
                    pass

        func_load_data: callable, default = None
            データ読み出し用関数
            Noneの場合はmsgpack読み出し

            .. code-block:: python

                def func_load_data(save_base, suffix):
                    # ...
                    return data



        """
        self._check_subject(subject)
        m.patch()

        cache = kwargs.get("cache", True)
        force_update = kwargs.get("force_update", False)
        concat_runs = kwargs.get("concat_runs", False)
        concat_sessions = kwargs.get("concat_sessions", False)

        if cache is True:
            if params is not None:
                hash, canonical_params = utils.encode_params(params)
            else:
                params = {"None": None}
                hash, canonical_params = utils.encode_params(params)

            from .__init__ import __version__

            canonical_params["hash"] = f"{hash}"
            canonical_params["version"] = __version__

            manifest = pd.DataFrame([canonical_params])

            save_base = (
                self.base_dir / "derivatives" / data_type / hash / f"sub-{subject}"
            )

            fname_manifest = self.base_dir / "derivatives" / data_type / "manifest"

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

                if os.path.exists(f"{fname_manifest}.parquet"):
                    manifest_exist = pd.read_parquet(f"{fname_manifest}.parquet")
                    manifest = pd.concat(
                        [manifest_exist, manifest], axis=0, ignore_index=True
                    )

                manifest.to_csv(f"{fname_manifest}.tsv", sep="\t", index=False)
                manifest.to_parquet(f"{fname_manifest}.parquet")

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

        method = params["method"]
        iir_params = params["iir_params"]
        phase = params["phase"]
        fir_window = params["fir_window"]
        fir_design = params["fir_design"]

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
            method=method,
            iir_params=iir_params,
            phase=phase,
            fir_window=fir_window,
            fir_design=fir_design,
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

        method = params["method"]
        iir_params = params["iir_params"]
        phase = params["phase"]
        fir_window = params["fir_window"]
        fir_design = params["fir_design"]

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
            method=method,
            iir_params=iir_params,
            phase=phase,
            fir_window=fir_window,
            fir_design=fir_design,
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
        method="iir",
        iir_params={"ftype": "butter", "order": 4, "btype": "bandpass"},
        phase="zero",
        fir_window="hamming",
        fir_design="firwin",
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
                method=method,
                iir_params=iir_params,
                phase=phase,
                fir_window=fir_window,
                fir_design=fir_design,
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
                method=method,
                iir_params=iir_params,
                phase=phase,
                fir_window=fir_window,
                fir_design=fir_design,
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

        manifest = self.get_manifest("epochs")
        params_epochs = manifest.iloc[0].to_dict()

        l_freq = params_epochs["l_freq"]
        h_freq = params_epochs["h_freq"]
        method = params_epochs["method"]
        iir_params = params_epochs["iir_params"]
        phase = params_epochs["phase"]
        fir_window = params_epochs["fir_window"]
        fir_design = params_epochs["fir_design"]
        tmin = params_epochs["tmin"]
        tmax = params_epochs["tmax"]
        baseline = params_epochs["baseline"]
        resample = params_epochs["resample"]

        label_keys = params["label_keys"]

        epochs = self.get_epochs(
            subject=subject,
            l_freq=l_freq,
            h_freq=h_freq,
            tmin=tmin,
            tmax=tmax,
            baseline=baseline,
            resample=resample,
            method=method,
            iir_params=iir_params,
            phase=phase,
            fir_window=fir_window,
            fir_design=fir_design,
        )

        labels = {}

        for session in self.sessions_list:
            labels[session] = {}

            for k, e in epochs[session].items():
                labels[session][k] = utils.get_labels_from_epochs(e, label_keys)

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

        method_epochs = params["method_epochs"]
        iir_params = params["iir_params"]
        phase = params["phase"]
        fir_window = params["fir_window"]
        fir_design = params["fir_design"]

        method = params["method"]
        freqs = params["freqs"]
        n_cycles = params["n_cycles"]
        use_fft = params["use_fft"]
        decim = params["decim"]

        epochs = self.get_epochs(
            subject=subject,
            l_freq=l_freq,
            h_freq=h_freq,
            method=method_epochs,
            iir_params=iir_params,
            phase=phase,
            fir_window=fir_window,
            fir_design=fir_design,
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

    def get_epochs(
        self,
        subject,
        l_freq=1.0,
        h_freq=45.0,
        method="iir",
        iir_params={"ftype": "butter", "order": 4, "btype": "bandpass"},
        phase="zero",
        fir_window="hamming",
        fir_design="firwin",
        tmin=-0.2,
        tmax=0.5,
        baseline=None,
        resample=128,
        cache=True,
        force_update=False,
        concat_runs=False,
        concat_sessions=False,
    ):

        self._check_subject(subject)

        data_type = "epochs"
        suffix = "-epo.fif"

        params = dict(
            tmin=tmin,
            tmax=tmax,
            l_freq=l_freq,
            h_freq=h_freq,
            method=method,
            iir_params=iir_params,
            phase=phase,
            fir_window=fir_window,
            fir_design=fir_design,
            baseline=baseline,
            resample=resample,
        )

        data = self._get_data(
            subject,
            data_type=data_type,
            params=params,
            suffix=suffix,
            func_get_data=self._get_epochs,
            func_save_data=save_mne_objs,
            func_load_data=functools.partial(
                load_mne_objs,
                func_load=mne.read_epochs,
            ),
            cache=cache,
            force_update=force_update,
        )

        return data

    def get_tfrs(
        self,
        subject,
        tmin_epochs,
        tmax_epochs,
        method_epochs="iir",
        iir_params={"ftype": "butter", "order": 4, "btype": "bandpass"},
        phase="zero",
        fir_window="hamming",
        fir_design="firwin",
        l_freq=1.0,
        h_freq=45.0,
        baseline=None,
        resample=128,
        method="multitaper",
        freqs=range(1, 46),
        n_cycles=range(1, 46),
        use_fft=True,
        decim=2,
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
            method_epochs=method_epochs,
            iir_params=iir_params,
            phase=phase,
            fir_window=fir_window,
            fir_design=fir_design,
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
