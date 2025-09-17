from ..base import BaseDataset
from pathlib import Path

import mne
import pyriemann

import tag_mne as tm
import utils


class Dreyer2023(BaseDataset):
    def __init__(self, base_dir=Path.home() / "Documents" / "datasets" / "dreyer_2023"):
        subjects_list = list(range(1, 88))  # subjects 1-87
        sessions_list = [1]
        super().__init__(
            base_dir=base_dir, subjects_list=subjects_list, sessions_list=sessions_list
        )

    def _get_raw(self, subject):

        if subject <= 60:
            prefix = "A"
        elif subject <= 81:
            prefix = "B"
        elif subject <= 87:
            prefix = "C"

        subject_code = f"{prefix}{subject}"

        base = self.base_dir / "raw" / subject_code

        runs = {}
        for run in range(1, 7):

            if (subject_code == "A59") and (run == 5):
                break

            if run <= 2:
                run_name = "acquisition"
            elif run <= 6:
                run_name = "onlineT"

            fname = base / f"{subject_code}_R{run}_{run_name}.gdf"

            raw = mne.io.read_raw_gdf(fname)

            runs[f"R{run}"] = raw

        return {1: runs}

    def _get_epochs(
        self, subject, l_freq, h_freq, order, tmin, tmax, baseline, resample
    ):
        raws = self.get_raw(subject)

        epochs_dict = {1: {}}
        for key, raw in raws[1].items():

            raw.load_data()

            run = int(key[1])

            if run <= 2:
                rtype = "acquisition"
            elif run <= 6:
                rtype = "online"

            raw.filter(
                l_freq=l_freq,
                h_freq=h_freq,
                method="iir",
                iir_params={"ftype": "butter", "order": order, "btype": "bandpass"},
            )

            # eog and emg mapping
            mapping = dict()
            for ch in raw.ch_names:
                if "EOG" in ch:
                    mapping[ch] = "eog"
                elif "EMG" in ch:
                    mapping[ch] = "emg"

            raw.set_channel_types(mapping)
            raw.set_montage("standard_1020")

            events, event_id = mne.events_from_annotations(raw)

            samples, markers = tm.markers_from_events(events, event_id)
            markers = tm.add_tag(markers, f"subject:{subject}")
            markers = tm.add_event_names(markers, {"left": ["769"], "right": ["770"]})
            markers = tm.add_tag(markers, f"run:{run}")
            markers = tm.add_tag(markers, f"rtype:{rtype}")

            samples, markers = tm.remove(samples, markers, "event:misc")

            events, event_id = tm.events_from_markers(samples, markers)
            epochs = mne.Epochs(
                raw=raw,
                tmin=tmin,
                tmax=tmax,
                events=events,
                event_id=event_id,
                baseline=baseline,
            )

            epochs.load_data()

            if resample is not None:
                epochs.resample(resample)

            epochs_dict[1][key] = epochs

        return epochs_dict

    def get_epochs_phase(
        self,
        subject,
        l_freq=8,
        h_freq=30,
        order=4,
        tmin=-5.0,
        tmax=7.0,
        baseline=None,
        resample=128,
    ):
        epochs = self.get_epochs(
            subject,
            l_freq=l_freq,
            h_freq=h_freq,
            order=order,
            tmin=tmin,
            tmax=tmax,
            baseline=baseline,
            resample=resample,
        )

        epochs_acquisition = [epochs[1][f"R{run}"] for run in [1, 2]]
        if subject == 59:
            epochs_online = [epochs[1][f"R{run}"] for run in [3, 4]]
        else:
            epochs_online = [epochs[1][f"R{run}"] for run in [3, 4, 5, 6]]

        epochs_acquisition = mne.concatenate_epochs(epochs_acquisition)
        epochs_online = mne.concatenate_epochs(epochs_online)

        return {"acquisition": epochs_acquisition, "online": epochs_online}

    def _______get_covs(
        self,
        subject,
        tmin,
        tmax,
        picks,
        l_freq,
        h_freq,
        order,
        tmin_epochs,
        tmax_epochs,
        baseline,
        resample,
        **kwargs,
    ):
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

        label_keys = {"event:left": 0, "event:right": 1}
        data = {}
        for k, e in epochs[1].items():
            e = e.pick(picks=picks).crop(tmin=tmin, tmax=tmax)
            y = utils.get_labels_from_epochs(e, label_keys)
            covs = pyriemann.estimation.Covariances().fit_transform(e.get_data())

            data[k] = {}

            data[k]["covs"] = covs
            data[k]["y"] = y

        return {1: data}

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

        label_keys = {"event:left": 0, "event:right": 1}
        data = {}
        for k, e in epochs[1].items():
            e = e.pick(picks=picks).crop(tmin=tmin, tmax=tmax)
            y = utils.get_labels_from_epochs(e, label_keys)
            covs = pyriemann.estimation.Covariances(estimator=estimator).fit_transform(
                e.get_data()
            )

            data[k] = {}

            data[k]["covs"] = covs
            data[k]["y"] = y

        data = {1: data}

        return data

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
        **kwargs,
    ):

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
        )

        return data
