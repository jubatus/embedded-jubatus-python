import json
import unittest

from embedded_jubatus import Regression
from jubatus.common import Datum
from jubatus.regression.types import ScoredDatum
import numpy as np


CONFIG = {
    "method": "PA1",
    "parameter": {
        "sensitivity": 0.1,
        "regularization_weight": 3.402823e+38
    },
    "converter": {
        "num_filter_types": {},
        "num_filter_rules": [],
        "string_filter_types": {},
        "string_filter_rules": [],
        "num_types": {},
        "num_rules": [
            {"key": "*", "type": "num"}
        ],
        "string_types": {},
        "string_rules": [
            {"key": "*", "type": "space",
             "sample_weight": "bin", "global_weight": "bin"}
        ]
    }
}


class TestRegression(unittest.TestCase):
    def test_invalid_config(self):
        self.assertRaises(TypeError, Regression)
        self.assertRaises(ValueError, Regression, {})
        self.assertRaises(TypeError, Regression, {'method': 'hoge'})
        self.assertRaises(RuntimeError, Regression,
                          {'method': 'hoge', 'converter': {}})
        invalid_config = dict(CONFIG)
        invalid_config['method'] = 'hoge'
        self.assertRaises(RuntimeError, Regression, invalid_config)

    def test(self):
        x = Regression(CONFIG)
        self.assertEqual(5, x.train([
            ScoredDatum(0.0, Datum({'x': 1.0})),
            ScoredDatum(1.0, Datum({'x': 2.0})),
            ScoredDatum(2.0, Datum({'x': 4.0})),
            ScoredDatum(3.0, Datum({'x': 8.0})),
            ScoredDatum(4.0, Datum({'x': 16.0})),
        ]))
        ret = x.estimate([
            Datum({'x': 32.0}),
            Datum({'x': 1.5}),
        ])
        self.assertEqual(2, len(ret))
        self.assertTrue(ret[0] >= 8.0 and ret[0] < 9.0)
        self.assertTrue(ret[1] >= 0.0 and ret[1] < 1.0)
        self.assertEqual(CONFIG, json.loads(x.get_config()))

        model = x.save_bytes()
        x = Regression(CONFIG)
        x.load_bytes(model)
        self.assertEqual(ret, x.estimate([
            Datum({'x': 32.0}),
            Datum({'x': 1.5}),
        ]))

        st = x.get_status()
        self.assertTrue(isinstance(st, dict))
        self.assertEqual(len(st), 1)
        self.assertEqual(list(st.keys())[0], 'embedded')
        self.assertTrue(isinstance(st['embedded'], dict))

    def test_numpy(self):
        X = np.array([[1.0], [2.0], [4.0], [8.0], [16.0]])
        y = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        reg = Regression(CONFIG)
        reg.fit(X, y)
        ret = reg.predict(np.array([[32.0], [1.5]]))
        self.assertIsInstance(ret, np.ndarray)
        self.assertTrue(ret[0] >= 8.0 and ret[0] < 9.0)
        self.assertTrue(ret[1] >= 0.0 and ret[1] < 1.0)
