from ..base import BaseDataset
from pathlib import Path

import mne

import tag_mne as tm
from .. import utils


class Lee2019(BaseDataset):
    def __init__(self, base_dir="~/Documents/datasets/lee_2019"):
        subjects_list = list(range(1, 55))  # subjects 1-54
        sessions_list = [1, 2]
        super().__init__(
            base_dir=Path(base_dir).expanduser(),
            subjects_list=subjects_list,
            sessions_list=sessions_list,
        )

    def get_subject_code(self, subject):
        return f"sub-{subject}"

    def subject_code_to_subject(self, subject_code):
        return int(subject_code.split("-")[1])

    def _get_raw(self, subject):

        base = self.base_dir / "raw" / f"sub-{subject}"

        data = {}
        for session in self.sessions_list:
            data[session] = {}
            fname = (
                    base
                    / f"ses-{session}"
                    / "eeg"
                    / f"sub-{subject}_ses-{session}_task-MItrain_eeg.fif"
            )

            raw = mne.io.read_raw(fname)

            data[session][f"R1"] = raw

        return data

    def _get_epochs(self, subject, params):
        tmin = params["tmin"]
        tmax = params["tmax"]
        baseline = params["baseline"]
        resample = params["resample"]

        l_freq = params["l_freq"]
        h_freq = params["h_freq"]
        method = params["method"]
        iir_params = params["iir_params"]
        phase = params["phase"]
        fir_window = params["fir_window"]
        fir_design = params["fir_design"]

        raws = self.get_raw(subject)

        data = {}
        for session in self.sessions_list:
            data[session] = {}

            for run, raw in raws[session].items():
                raw.load_data()

                print(raw)

                raw.filter(
                    l_freq=l_freq,
                    h_freq=h_freq,
                    method=method,
                    iir_params=iir_params,
                    phase=phase,
                    fir_window=fir_window,
                    fir_design=fir_design,
                )

                # raw.set_montage("brainproducts-RNP-BA-128")

                epochs = mne.Epochs(
                    raw=raw,
                    tmin=tmin,
                    tmax=tmax,
                    baseline=baseline,
                )

                if resample is not None:
                    epochs.load_data()
                    epochs.resample(resample)

                data[session][run] = epochs

        return data

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
