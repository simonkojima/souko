import re
import tag_mne as tm
import numpy as np


def natural_key(s: str):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", s)]


def get_labels_from_epochs(epochs, label_keys={"event:left": 0, "event:right": 1}):
    y = list()

    _, markers = tm.markers_from_events(epochs.events, epochs.event_id)

    for marker in markers:
        for key, val in label_keys.items():
            if "/" in marker:
                if key in marker.split("/"):
                    y.append(val)
            else:
                if key in marker:
                    y.append(val)

    if len(epochs) != len(y):
        raise RuntimeError(
            f"lenth of epochs is not match with length of y.\n len(epochs): {len(epochs)}, len(y): {len(y)}"
        )

    return np.array(y)
