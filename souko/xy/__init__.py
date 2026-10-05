from . import load
from .load import get_data, get_data_cross, subject_list, sessions_list


def export_data(*args, **kwargs):
    """Import MNE and export dependencies only when exporting."""
    from .export import export_data as export
    return export(*args, **kwargs)
