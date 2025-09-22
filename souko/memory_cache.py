class MemoryCache:
    def __init__(self, subjects):
        """
        指定したsubjectsのデータを保持し，読み込ませる
        """
        # self.dataset = dataset
        self.subjects = subjects
        self.data = {}

    def load(
        self, name, func_get_data, concat_runs=False, concat_sessions=False, **kwargs
    ):
        self.data[name] = {}

        for subject in self.subjects:
            data = func_get_data(
                subject=subject,
                **kwargs,
                concat_runs=concat_runs,
                concat_sessions=concat_sessions,
            )

            self.data[name][subject] = data

    def get(self, name, subject):
        return self.data[name][subject]
