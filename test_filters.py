import unittest

import pandas as pd

from cfstats import db


class TestApplyFiltersPandas(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame(
            {
                'a': [1, 2, 3, None],
                'b': ['foo', 'bar', 'foobar', None],
                'c': ['10', '20', 'x', '30'],
            },
            index=['s1', 's2', 's3', 's4']
        )

    def test_no_filters_returns_all_true(self):
        mask = db.apply_filters_pandas(self.df, [])
        self.assertEqual(mask.dtype, bool)
        self.assertListEqual(mask.tolist(), [True, True, True, True])

    def test_range_filter_inclusive(self):
        filters = [{'column': 'a', 'op': 'range', 'lo': 2, 'hi': 3, 'expr': None}]
        mask = db.apply_filters_pandas(self.df, filters)
        self.assertListEqual(mask.tolist(), [False, True, True, False])

    def test_range_filter_requires_numeric_and_notna(self):
        filters = [{'column': 'c', 'op': 'range', 'lo': 15, 'hi': 25, 'expr': None}]
        mask = db.apply_filters_pandas(self.df, filters)
        self.assertListEqual(mask.tolist(), [False, True, False, False])

    def test_regex_include(self):
        filters = [{'column': 'b', 'op': '==', 'lo': None, 'hi': None, 'expr': 'foo'}]
        mask = db.apply_filters_pandas(self.df, filters)
        self.assertListEqual(mask.tolist(), [True, False, True, False])

    def test_regex_exclude(self):
        filters = [{'column': 'b', 'op': '!=', 'lo': None, 'hi': None, 'expr': 'foo'}]
        mask = db.apply_filters_pandas(self.df, filters)
        self.assertListEqual(mask.tolist(), [False, True, False, True])

    def test_multiple_filters_and(self):
        filters = [
            {'column': 'a', 'op': 'range', 'lo': 2, 'hi': 3, 'expr': None},
            {'column': 'b', 'op': '==', 'lo': None, 'hi': None, 'expr': '^foo'},
        ]
        mask = db.apply_filters_pandas(self.df, filters)
        self.assertListEqual(mask.tolist(), [False, False, True, False])

    def test_missing_column_filters_to_false(self):
        filters = [{'column': 'missing', 'op': 'range', 'lo': 1, 'hi': 2, 'expr': None}]
        mask = db.apply_filters_pandas(self.df, filters)
        self.assertListEqual(mask.tolist(), [False, False, False, False])

    def test_invalid_regex_filters_to_false(self):
        filters = [{'column': 'b', 'op': '==', 'lo': None, 'hi': None, 'expr': '[unclosed'}]
        mask = db.apply_filters_pandas(self.df, filters)
        self.assertListEqual(mask.tolist(), [False, False, False, False])


if __name__ == '__main__':
    unittest.main()
