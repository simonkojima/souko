import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
from souko.split import split_indices, split_data
from souko.xy.load import get_data, get_data_cross, subject_list, sessions_list
from souko.xy.utils import get_proc_name


class SplitTests(unittest.TestCase):
    def test_multiclass_order_and_partition(self):
        y = np.tile([2, 7, 9], 10)
        train, test = split_indices(y)
        np.testing.assert_array_equal(train, np.arange(24))
        np.testing.assert_array_equal(test, np.arange(24, 30))
        for part in (train, test):
            self.assertTrue(np.all(np.diff(part) > 0))
        np.testing.assert_array_equal(np.sort(np.r_[train, test]), np.arange(30))

    def test_class_boundaries_differ(self):
        y = np.array([2, 2, 2, 7, 7, 2, 7, 7])
        train, test = split_indices(y, (0.5, 0.5))
        np.testing.assert_array_equal(train, [0, 1, 3, 4])
        np.testing.assert_array_equal(test, [2, 5, 6, 7])

    def test_validation(self):
        for ratios in ((0, 1), (-1, 2), (0.3, 0.3), (float('nan'), 1)):
            with self.assertRaises(ValueError):
                split_indices([0, 1], ratios)
        with self.assertRaises(ValueError):
            split_data(np.zeros((3, 2)), np.array([0, 1]))
        parts = split_indices([], (0.8, 0.2))
        self.assertEqual(sum(map(len, parts)), 0)


class LoaderTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.base = Path(self.temp.name)
        self.path = self.base / 'datasets' / 'Custom' / 'sub-3' / 'ses-1' / get_proc_name(128, .5, 5, 7, 30)
        self.path.mkdir(parents=True)
        self.prefix = 'sub-3_ses-1'
        self.X = np.arange(24).reshape(12, 2, 1)
        self.y = np.tile([2, 7, 9], 4)
        for run, sl in ((1, slice(0, 6)), (2, slice(6, 12))):
            np.save(self.path / f'{self.prefix}_run-{run}_X.npy', self.X[sl])
            np.save(self.path / f'{self.prefix}_run-{run}_y.npy', self.y[sl])
        np.save(self.path / f'{self.prefix}_X_ea.npy', self.X + 100)
        np.save(self.path / f'{self.prefix}_X_ea_online.npy', self.X + 200)
        (self.path / f'{self.prefix}_meta.json').write_text(json.dumps({'session_name': 'original'}))

    def tearDown(self):
        self.temp.cleanup()

    def load(self, **kwargs):
        return get_data('Custom', subject=3, base=self.base, **kwargs)

    def test_discovery_and_default_split(self):
        self.assertEqual(subject_list('Custom', self.base), [3])
        self.assertEqual(sessions_list('Custom', 3, self.base), [1])
        data = self.load(train_test_split=.5)
        np.testing.assert_array_equal(data['train']['eeg'], self.X[:6])
        np.testing.assert_array_equal(data['test']['eeg'], self.X[6:])

    def test_validation_never_uses_test(self):
        data = self.load(runs={'train': [1], 'test': [2]}, valid=True, train_valid_split=.5)
        np.testing.assert_array_equal(data['train']['eeg'], self.X[:3])
        np.testing.assert_array_equal(data['valid']['eeg'], self.X[3:6])
        np.testing.assert_array_equal(data['test']['eeg'], self.X[6:])

    def test_cached_ea_run_selection(self):
        for online, offset in ((False, 100), (True, 200)):
            data = self.load(runs={'train': [2], 'test': [1]}, ea=True, online=online)
            np.testing.assert_array_equal(data['train']['eeg'], self.X[6:] + offset)
            np.testing.assert_array_equal(data['test']['eeg'], self.X[:6] + offset)
        with self.assertRaises(ValueError):
            self.load(online=True)
        with self.assertRaises(ValueError):
            self.load(runs={'train': [1], 'test': [1]})

    def test_cross_metadata(self):
        data, info = get_data_cross('Custom', subject=3, base=self.base, return_info=True)
        np.testing.assert_array_equal(data['eeg'], self.X)
        self.assertEqual(info['sessions']['1']['session_name'], 'original')


if __name__ == '__main__':
    unittest.main()
