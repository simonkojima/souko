import numpy as np
import mne


def concatenate_events(events_list, add_offset_sample=True):
    offset = np.max(events_list[0][:, 0])

    for events in events_list[1: len(events_list)]:
        events[:, 0] += int(offset)
        offset = np.max(events[:, 0])

    events = np.concatenate(events_list, axis=0)

    return events


def concatenate_event_ids(event_id_list):
    seen = set()
    out = {}
    for d in event_id_list:
        dup = seen.intersection(d.keys())
        if dup:
            raise KeyError(f"event_key is duplicated: {sorted(dup)}")
        out.update(d)
        seen.update(d.keys())
    return out


def is_event_ids_identical(event_ids):
    event_id_0 = event_ids[0]
    for event_id in event_ids[1:]:
        if event_id_0 != event_id:
            return False
    return True


def concatenate_tfrs(tfrs_list, add_offset_event_id=True):
    if add_offset_event_id:
        events = tfrs_list[0].events
        ids = events[:, 2]
        offset = np.max(ids)

        for tfrs in tfrs_list[1: len(tfrs_list)]:
            tfrs.events[:, 2] += int(offset)
            tfrs.event_id = {k: int(v + offset) for k, v in tfrs.event_id.items()}
            offset = np.max(tfrs.events[:, 2])

    events_list = [tfrs.events for tfrs in tfrs_list]
    new_events = concatenate_events(events_list=events_list)

    if add_offset_event_id:
        event_ids = [tfrs.event_id for tfrs in tfrs_list]
        new_event_id = concatenate_event_ids(event_ids)
    else:
        event_ids = [tfrs.event_id for tfrs in tfrs_list]
        if is_event_ids_identical(event_ids) is False:
            raise RuntimeError("event_ids are not identical")
        new_event_id = tfrs_list[0].event_id

    new_data = [tfrs.get_data() for tfrs in tfrs_list]
    new_data = np.concatenate(new_data, axis=0)

    info = tfrs_list[0].info
    times = tfrs_list[0].times
    freqs = tfrs_list[0].freqs

    tfrs = mne.time_frequency.EpochsTFRArray(
        info=info,
        data=new_data,
        times=times,
        freqs=freqs,
        events=new_events,
        event_id=new_event_id,
    )

    return tfrs
