import os
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UTILS_DIR = os.path.join(ROOT_DIR, 'utils')

if UTILS_DIR not in sys.path:
    sys.path.insert(0, UTILS_DIR)

import data_processing


class DataProcessingTests(unittest.TestCase):
    def test_filter_zip_coordinate_rows_filters_and_reindexes(self):
        zip_codes_data = pd.DataFrame({
            'zip': [11111, 22222, 33333, 44444],
            'latitude': [1.0, 2.0, 3.0, 4.0],
            'longitude': [5.0, 6.0, 7.0, 8.0],
        })
        census_data = pd.DataFrame({
            'ZIPCODE': [22222, 44444],
            'N1': [10, 20],
            'A02650': [100, 200],
        })

        filtered, valid_zip_codes = data_processing.filter_zip_coordinate_rows(
            zip_codes_data,
            census_data,
        )

        self.assertEqual(list(filtered['zip']), [22222, 44444])
        self.assertEqual(list(filtered.index), [0, 1])
        self.assertEqual(valid_zip_codes, {22222, 44444})

    def test_build_groups_from_mapping_assigns_every_image(self):
        mapping = {90001: [0, 2], 90002: [1, 3]}

        groups = data_processing.build_groups_from_mapping(mapping, 4)

        np.testing.assert_array_equal(groups, np.array([90001, 90002, 90001, 90002]))

    def test_grouped_train_test_split_keeps_groups_disjoint(self):
        X = np.arange(24).reshape(12, 2)
        y = np.arange(12)
        groups = np.array([1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4])

        X_train, X_test, y_train, y_test, train_groups, test_groups = data_processing.grouped_train_test_split(
            X,
            y,
            groups,
            test_size=0.25,
            random_state=0,
        )

        self.assertTrue(set(train_groups).isdisjoint(set(test_groups)))
        self.assertEqual(len(X_train) + len(X_test), len(X))
        self.assertEqual(len(y_train) + len(y_test), len(y))

    def test_load_mapping_rejects_legacy_cache_format(self):
        mapping = data_processing.Mapping.__new__(data_processing.Mapping)

        with tempfile.TemporaryDirectory() as tmpdir:
            legacy_path = os.path.join(tmpdir, 'mapping.json')
            with open(legacy_path, 'w') as handle:
                handle.write('{"90001": [0, 1]}')

            with self.assertRaises(ValueError):
                mapping.loadMapping(legacy_path)

    def test_load_mapping_accepts_versioned_cache_format(self):
        mapping = data_processing.Mapping.__new__(data_processing.Mapping)

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = os.path.join(tmpdir, 'mapping.json')
            with open(cache_path, 'w') as handle:
                handle.write(
                    '{"version": 2, "bounds": {"min_x": 2794, "max_x": 2838, "min_y": 6528, "max_y": 6571}, "mapping": {"90001": [0, 1]}}'
                )

            mapping.loadMapping(cache_path)

        self.assertEqual(mapping.mapping[90001], [0, 1])


if __name__ == '__main__':
    unittest.main()
