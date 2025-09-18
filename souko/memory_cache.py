class MemoryCache:
    def __init__(self, dataset, subjects):
        self.dataset = dataset
        self.subjects = subjects
        self.data = {}

    def load(self, name, concat_runs=False, concat_sessions=False, **kwargs):
        self.data[name] = {}

        for subject in self.subjects:
            data = self.dataset.get_covs(
                subject=subject,
                **kwargs,
                concat_runs=concat_runs,
                concat_sessions=concat_sessions,
            )

            self.data[name][subject] = data

    def get(self, name, subject):
        return self.data[name][subject]
