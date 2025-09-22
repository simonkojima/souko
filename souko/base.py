import os
import warnings
import pandas as pd
import functools

from . import utils
import mne
import msgpack
import msgpack_numpy as m

import numpy as np


def load_dict(data, key, default):
    if key in list(data.keys()):
        return data[key]
    else:
        return default


def set_dict(params, key, value):
    if key not in list(params.keys()):
        params[key] = value
    return params


def proc_params_epochs(params):
    params = set_dict(params, "l_freq", 1.0)
    params = set_dict(params, "h_freq", 45.0)
    params = set_dict(params, "method", "iir")
    params = set_dict(
        params,
        "iir_params",
        {"ftype": "butter", "order": 4, "btype": "bandpass"},
    )
    params = set_dict(params, "phase", "zero")
    params = set_dict(params, "fir_window", "hamming")
    params = set_dict(params, "fir_design", "firwin")
    params = set_dict(params, "tmin", -0.2)
    params = set_dict(params, "tmax", 0.5)
    params = set_dict(params, "baseline", None)
    params = set_dict(params, "resample", 128)

    params = utils.type_params(params)

    return params


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
        """
        Souko Baseクラス

        Parameters
        ----------
        base_dir: path-like
            データのベースディレクトリ指定
        subjects_list: list of int
            subject number のリスト
        sessions_list: list of int
            セッションのリスト
            全被験者で同じセッション数の想定

        """
        self.base_dir = base_dir
        self.subjects_list = subjects_list
        self.sessions_list = sessions_list

    def update_manifest(self, data_type):
        """
        manifestファイルの更新
        手動でフォルダ消したときにこれ実行すれば，自動的に更新される

        Parameters
        ----------
        data_type: str
            "covs"とかそういうやつ

        """
        manifest = self.get_manifest(data_type)

        hashs_manifest = manifest["hash"].tolist()

        hashs = os.listdir(self.base_dir / "derivatives" / data_type)

        for h in hashs_manifest:
            if h not in hashs:
                manifest = manifest[manifest["hash"] != h]

                manifest.to_csv(
                    self.base_dir / "derivatives" / data_type / "manifest.tsv",
                    sep="\t",
                    index=False,
                )
                manifest.to_parquet(
                    self.base_dir / "derivatives" / data_type / "manifest.parquet"
                )

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
        """
        BIDSのparticipants.tsvを読み込み、pandas.DataFrameを返す
        """
        fname = self.base_dir / "participants.tsv"
        df = pd.read_csv(fname, sep="\t")

        return df

    def get_manifest(self, data_type):
        """
        各データタイプについて、hashとパラメータのリストを含むpandas.DataFrameを返す

        Parameters
        ----------
        data_type: str
            ``base_dir / "derivatives" / data_type / "manifest.parquet"`` を読み込む
        """
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

                    hash_list = manifest_exist["hash"].tolist()
                    if hash in hash_list:
                        manifest = None
                    else:
                        manifest = pd.concat(
                            [manifest_exist, manifest], axis=0, ignore_index=True
                        )

                if manifest is not None:
                    manifest.to_csv(f"{fname_manifest}.tsv", sep="\t", index=False)
                    manifest.to_parquet(f"{fname_manifest}.parquet")

        else:
            data = func_get_data(subject, params)

        if concat_runs:
            for session in self.sessions_list:
                values = list(data[session].values())

                if (isinstance(values[0], mne.Epochs)) or (
                    isinstance(values[0], mne.EpochsArray)
                    or (isinstance(values[0], mne.epochs.EpochsFIF))
                ):
                    data[session] = mne.concatenate_epochs(values)
                else:
                    try:
                        data[session] = np.concatenate(values, axis=0)
                    except:
                        data[session] = values

        if concat_sessions:
            values = list(data.values())

            if (isinstance(values[0], mne.Epochs)) or (
                isinstance(values[0], mne.EpochsArray)
                or (isinstance(values[0], mne.epochs.EpochsFIF))
            ):
                data = mne.concatenate_epochs(values)
            else:
                try:
                    data = np.concatenate(values, axis=0)
                except:
                    data = values

        return data

    def _get_covs(self, subject, params):
        params_epochs = params["params_epochs"]

        tmin = params["tmin"]
        tmax = params["tmax"]
        picks = params["picks"]
        estimator = params["estimator"]

        epochs = self.get_epochs(subject=subject, **params_epochs)

        import pyriemann

        data = {}
        for session in self.sessions_list:
            data[session] = {}
            for k, e in epochs[session].items():
                e = e.pick(picks=picks).crop(tmin=tmin, tmax=tmax)
                covs = pyriemann.estimation.Covariances(
                    estimator=estimator
                ).fit_transform(e.get_data())

                data[session][k] = covs

        return data

    def _get_covs_rpa(self, subject, params, cache=True, force_update=False):

        params_epochs = params["params_epochs"]

        tmin = params["tmin"]
        tmax = params["tmax"]
        picks = params["picks"]
        estimator = params["estimator"]

        rescaling = params["rescaling"]
        online_rpa = params["online_rpa"]

        covs = self.get_covs(
            subject=subject,
            tmin=tmin,
            tmax=tmax,
            picks=picks,
            params_epochs=params_epochs,
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
        **kwargs,
    ):
        """
        Covariance Matrixを計算
        ``pyriemann.estimation.Covariances()`` を利用

        Parameters
        ----------
        params_epochs: dict
            ``BaseDataset.get_epochs`` に渡される引数
        picks: str, default = "eeg"
        tmin: float, default = 0.0
        tmax: float, default = 1.0
        estimator: str, default = "scm"
            詳しくは， ``pyriemann.estimation.Covariances()`` 参照
        recentering: bool, default = False
            RPAのRecenteringを適用
        rescaling: bool, default = False
            RPAのRescalingを適用
        online_rpa: bool, default = False
            RPAをオンラインで適用．第Nエポック目のCOVをそこまでのデータでRPA適用．

        Returns
        -------
        dict of covariances (np.ndarray)
        """
        params_epochs = kwargs.get("params_epochs", {})
        params_epochs = proc_params_epochs(params_epochs)

        picks = kwargs.get("picks", "eeg")
        tmin = kwargs.get("tmin", 0.0)
        tmax = kwargs.get("tmax", 1.0)
        estimator = kwargs.get("estimator", "scm")

        recentering = kwargs.get("recentering", False)
        rescaling = kwargs.get("rescaling", False)
        online_rpa = kwargs.get("online_rpa", False)

        cache = kwargs.get("cache", True)
        force_update = kwargs.get("force_update", False)
        concat_runs = kwargs.get("concat_runs", False)
        concat_sessions = kwargs.get("concat_sessions", False)

        if recentering is False:
            data_type = "covs"
            params = dict(
                params_epochs=params_epochs,
                tmin=tmin,
                tmax=tmax,
                picks=picks,
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
                params_epochs=params_epochs,
                tmin=tmin,
                tmax=tmax,
                picks=picks,
                estimator=estimator,
                rescaling=rescaling,
                online_rpa=online_rpa,
            )

            if concat_runs is False:
                warnings.warn("All runs will be concatenated when RPA is enabled")

            data = self._get_data(
                subject,
                data_type,
                params,
                func_get_data=functools.partial(
                    self._get_covs_rpa, cache=cache, force_update=force_update
                ),
                cache=cache,
                force_update=force_update,
                concat_sessions=concat_sessions,
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
        """
        ラベルデータ取得

        Parameters
        ----------
        subject: int
        label_keys: dict, default = ``{"event:left":0, "event:right":1}``
        cache: bool, default = True
        force_update: bool, default = False
        concat_runs: bool, default = False
        concat_sessions: bool, default = False

        """
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

        params_epochs = params["params_epochs"]

        method = params["method"]
        freqs = params["freqs"]
        n_cycles = params["n_cycles"]
        use_fft = params["use_fft"]
        decim = params["decim"]

        epochs = self.get_epochs(subject=subject, **params_epochs)

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
        **kwargs,
    ):
        """
        epochs取得用

        Parameters
        ----------
        subject: int
        l_freq: float, default = 1.0
        h_freq: float, default = 45.0
        method: str, default = "iir"
        phase: str, default = "zero"
        fir_window: str, default = "hamming"
        fir_design: str, default = "firwin"
        tmin: float, default = -0.2
        tmax: float, default = 0.5
        baseline: tuple, list, or None, default = None
        resample: float, default = 128

        cache: bool, default = True
            データキャッシュを利用するか
        force_update: bool, default = False
            強制的に再計算
        concat_runs: bool, default = False
            セッションごとにrunデータを結合
        concat_sessions: bool, default = False
            全セッションデータを結合
            runが結合済みである必要がある

        Returns
        -------
        dict of epochs

        """

        l_freq = kwargs.get("l_freq", 1.0)
        h_freq = kwargs.get("h_freq", 45.0)
        method = kwargs.get("method", "iir")
        iir_params = kwargs.get(
            "iir_params",
            {"ftype": "butter", "order": 4, "btype": "bandpass"},
        )
        phase = kwargs.get("phase", "zero")
        fir_window = kwargs.get("fir_window", "hamming")
        fir_design = kwargs.get("fir_design", "firwin")
        tmin = kwargs.get("tmin", -0.2)
        tmax = kwargs.get("tmax", 0.5)
        baseline = kwargs.get("baseline", None)
        resample = kwargs.get("resample", 128)

        cache = kwargs.get("cache", True)
        force_update = kwargs.get("force_update", False)
        concat_runs = kwargs.get("concat_runs", False)
        concat_sessions = kwargs.get("concat_sessions", False)

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
            concat_runs=concat_runs,
            concat_sessions=concat_sessions,
        )

        return data

    def get_tfrs(
        self,
        subject,
        **kwargs,
    ):
        """
        Time-Frequency Representationの計算

        Parameters
        ----------
        subject: int
        params_epochs: dict
            ``BaseDataset.get_epochs`` に渡される引数
        method: str, default = "multitaper"
        freqs: list or range, default = range(1, 46)
        n_cycles: list or range, default = range(1, 46)
        use_fft: bool, default = True
        decim: int, default = 2
        n_jobs: int, default = -1
        cache: bool, default = True,
        force_update: bool, default = False
        concat_runs: bool, default = False
        concat_sessions: bool, default = False
        """
        data_type = "tfrs"
        suffix = "-tfr.hdf5"

        params_epochs = kwargs.get("params_epochs", {})
        params_epochs = proc_params_epochs(params_epochs)

        method = kwargs.get("method", "multitaper")
        freqs = kwargs.get("freqs", list(range(1, 46)))
        n_cycles = kwargs.get("n_cycles", list(range(1, 46)))
        use_fft = kwargs.get("use_fft", True)
        decim = kwargs.get("decim", 2)
        n_jobs = kwargs.get("n_jobs", -1)

        cache = kwargs.get("cache", True)
        force_update = kwargs.get("force_update", False)

        concat_runs = kwargs.get("concat_runs", False)
        concat_sessions = kwargs.get("concat_sessions", False)

        if isinstance(freqs, range):
            freqs = list(freqs)
        if isinstance(n_cycles, range):
            n_cycles = list(n_cycles)

        params = dict(
            params_epochs=params_epochs,
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

    def get_X_EA(
        self,
        subject,
        **kwargs,
    ):
        """
        Euclidean Alignmentを適用したエポックデータ取得

        Parameters
        ----------
        subject: int
        params_epochs: dict
            ``BaseDataset.get_epochs`` に渡される引数
        picks: str, default = "eeg"
        tmin: float, default = 0.0
        tmax: float, default = 1.0
        online_ea: bool, default = False
        cache: bool, default = True,
        force_update: bool, default = False
        concat_sessions: bool, default = False

        """
        data_type = "X-EA"

        params_epochs = kwargs.get("params_epochs", {})
        params_epochs = proc_params_epochs(params_epochs)

        picks = kwargs.get("picks", "eeg")
        tmin = kwargs.get("tmin", 0.0)
        tmax = kwargs.get("tmax", 1.0)
        online_ea = kwargs.get("online_ea", False)

        cache = kwargs.get("cache", True)
        force_update = kwargs.get("force_update", False)
        concat_runs = kwargs.get("concat_runs", False)
        concat_sessions = kwargs.get("concat_sessions", False)

        params = dict(
            params_epochs=params_epochs,
            picks=picks,
            tmin=tmin,
            tmax=tmax,
            online_ea=online_ea,
        )

        data = self._get_data(
            subject,
            data_type,
            params,
            func_get_data=functools.partial(
                self._get_X_EA, cache=cache, force_update=force_update
            ),
            cache=cache,
            force_update=force_update,
            concat_runs=False,
            concat_sessions=concat_sessions,
        )

        return data

    def _get_X_EA(self, subject, params, cache=True, force_update=False):

        params_epochs = params["params_epochs"]

        picks = params["picks"]
        tmin = params["tmin"]
        tmax = params["tmax"]

        online_ea = params["online_ea"]

        epochs = self.get_epochs(
            subject=subject,
            **params_epochs,
            cache=cache,
            force_update=force_update,
            concat_runs=True,
            concat_sessions=False,
        )
        from rosoku.tl import euclidean_alignment

        data = {}
        for session in self.sessions_list:
            data[session] = {}

            e = epochs[session]

            e = e.pick(picks=picks).crop(tmin=tmin, tmax=tmax)

            X_EA = euclidean_alignment(e.get_data(), online=online_ea)

            data[session] = X_EA

        return data
