from ..base import BaseDataset
from pathlib import Path

import mne

import tag_mne as tm
from .. import utils


class Dreyer2023(BaseDataset):
    def __init__(self, base_dir="~/Documents/datasets/dreyer_2023"):
        subjects_list = list(range(1, 88))  # subjects 1-87
        sessions_list = [1]
        super().__init__(
            base_dir=Path(base_dir).expanduser(),
            subjects_list=subjects_list,
            sessions_list=sessions_list,
        )
        self.trial_duration = 5
        self.subject_code_list = [self.get_subject_code(subject) for subject in self.subjects_list]

    def get_subject_code(self, subject):
        if subject <= 60:
            prefix = "A"
        elif subject <= 81:
            prefix = "B"
        elif subject <= 87:
            prefix = "C"

        subject_code = f"{prefix}{subject}"

        return subject_code

    def get_sex(self, subject):
        pat = self.get_participants()

        subject_code = self.get_subject_code(subject)

        return pat[pat["participant_id"] == subject_code]["sex"].tolist()[0]

    def subject_code_to_subject(self, subject_code):
        return int(subject_code[1:])

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
                method=method,
                iir_params=iir_params,
                phase=phase,
                fir_window=fir_window,
                fir_design=fir_design,
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
