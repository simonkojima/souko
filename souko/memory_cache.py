class MemoryCache:
    def __init__(self, dataset, subjects):
        self.dataset = dataset
        self.subjects = subjects
        self.covs = {}

    def load_covs(
        self,
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
        recentering=False,
        rescaling=False,
        online_rpa=False,
        concat_runs=False,
        concat_sessions=False,
    ):

        for subject in self.subjects:
            data = self.dataset.get_covs(
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
                recentering=recentering,
                rescaling=rescaling,
                online_rpa=online_rpa,
                concat_runs=concat_runs,
                concat_sessions=concat_sessions,
            )

            self.covs[subject] = data

    def get_covs(self, subject):
        return self.covs[subject]
