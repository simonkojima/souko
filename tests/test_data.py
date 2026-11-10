import mne

if __name__ == "__main__":
    mne.set_log_level("CRITICAL")

    from moabb.datasets import Dreyer2023
    import souko
    from pathlib import Path

    dataset = Dreyer2023()

    souko.export_data(
        dataset,
        resample=128,
        tmin=0.5,
        tmax=5,
        event_id={"left_hand": 0, "right_hand": 1},
        name="${NAME}_test",
        name_cache="${NAME}",
        picks=["Cz", "Fz"],
        cache_base=Path.home() / "datasets" / "0.5-40",
        save_base=Path.home() / "debug" / "souko",
    )
