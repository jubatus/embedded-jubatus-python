import json
import unittest

from embedded_jubatus import Anomaly
from jubatus.anomaly.types import IdWithScore
from jubatus.common import Datum
import numpy as np


CONFIG = {
    'method': 'lof',
    'parameter': {
        'nearest_neighbor_num': 3,
        'reverse_nearest_neighbor_num': 5,
        'method': 'lsh',
        'parameter': {
            'hash_num': 512,
        }
    },
    'converter': {
        'num_filter_types': {},
        'num_filter_rules': [],
        'string_filter_types': {},
        'string_filter_rules': [],
        'num_types': {},
        'num_rules': [
            {'key': '*', 'type': 'num'}
        ],
        'string_types': {},
        'string_rules': [
            {'key': '*', 'type': 'space',
             'sample_weight': 'bin', 'global_weight': 'bin'}
        ]
    },
}


class TestAnomaly(unittest.TestCase):
    def test_invalid_configs(self):
        self.assertRaises(TypeError, Anomaly)
        self.assertRaises(ValueError, Anomaly, {})
        self.assertRaises(TypeError, Anomaly, {'method': 'hoge'})
        self.assertRaises(RuntimeError, Anomaly,
                          {'method': 'hoge', 'converter': {}})
        invalid_config = dict(CONFIG)
        invalid_config['method'] = 'hoge'
        self.assertRaises(RuntimeError, Anomaly, invalid_config)

    def test(self):
        x = Anomaly(CONFIG)
        self.assertEqual(float('inf'), x.add(Datum({'x': 0.1})).score)
        self.assertTrue(isinstance(x.add(Datum({'x': 0.101})), IdWithScore))
        x.add(Datum({'x': 0.1001}))
        x.calc_score(Datum({'x': 0.1001}))
        self.assertEqual(set(['0', '1', '2']), set(x.get_all_rows()))

        model = x.save_bytes()
        x.clear()
        self.assertEqual([], x.get_all_rows())

        x = Anomaly(CONFIG)
        x.load_bytes(model)
        self.assertEqual(set(['0', '1', '2']), set(x.get_all_rows()))
        self.assertEqual(CONFIG, json.loads(x.get_config()))

        p = x.add(Datum({'x': 0.2}))
        self.assertEqual(set(['0', '1', '2', '3']), set(x.get_all_rows()))
        self.assertEqual(p.score, x.update(p.id, Datum({'x': 0.2})))
        self.assertEqual(p.score, x.overwrite(p.id, Datum({'x': 0.2})))

        st = x.get_status()
        self.assertTrue(isinstance(st, dict))
        self.assertEqual(len(st), 1)
        self.assertEqual(list(st.keys())[0], 'embedded')
        self.assertTrue(isinstance(st['embedded'], dict))

    def test_add_bulk(self):
        x = Anomaly(CONFIG)
        data = [
            Datum({'x': 0.0999}),
            Datum({'x': 0.1}),
            Datum({'x': -0.1009}),
            Datum({'x': -0.101}),
            Datum({'x': 0.1011}),
        ]
        ret = x.add_bulk(data)
        self.assertEqual(['0', '1', '2', '3', '4'], ret)
        self.assertEqual(set(ret), set(x.get_all_rows()))

        x = Anomaly(CONFIG)
        x.fit(np.array([[d.num_values[0][1]] for d in data]))
        self.assertEqual(['0', '1', '2', '3', '4'], ret)
        self.assertEqual(set(ret), set(x.get_all_rows()))
