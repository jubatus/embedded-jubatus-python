import json
import unittest

from embedded_jubatus import NearestNeighbor
from jubatus.common import Datum
from jubatus.nearest_neighbor.types import IdWithScore


CONFIG = {
    "method": "euclid_lsh",
    "parameter": {
        "hash_num": 128
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


class TestNearestNeighbor(unittest.TestCase):
    def test_invalid_configs(self):
        self.assertRaises(TypeError, NearestNeighbor)
        self.assertRaises(ValueError, NearestNeighbor, {})
        self.assertRaises(TypeError, NearestNeighbor, {'method': 'hoge'})
        self.assertRaises(RuntimeError, NearestNeighbor,
                          {'method': 'hoge', 'converter': {}})
        invalid_config = dict(CONFIG)
        invalid_config['method'] = 'hoge'
        self.assertRaises(RuntimeError, NearestNeighbor, invalid_config)

    def test(self):
        x = NearestNeighbor(CONFIG)
        self.assertTrue(x.set_row("a0", Datum({'x': 0})))
        self.assertTrue(x.set_row("a1", Datum({'x': 0.25})))
        self.assertTrue(x.set_row("a2", Datum({'x': 0.5})))
        self.assertTrue(x.set_row("a3", Datum({'x': 1})))
        self.assertTrue(x.set_row("b0", Datum({'x': 10})))
        self.assertTrue(x.set_row("b1", Datum({'x': 10.25})))
        self.assertTrue(x.set_row("b2", Datum({'x': 10.5})))
        self.assertTrue(x.set_row("b3", Datum({'x': 11})))

        def _check_prefix(expected, lst):
            for x in lst:
                self.assertTrue(isinstance(x, IdWithScore))
                self.assertTrue(x.id.startswith(expected))

        ret = x.neighbor_row_from_id("a0", 3)
        self.assertEqual(3, len(ret))
        _check_prefix('a', ret)

        ret = x.neighbor_row_from_datum(Datum({'x': 0.25}), 3)
        self.assertEqual(3, len(ret))
        _check_prefix('a', ret)

        ret = x.similar_row_from_id("b3", 3)
        self.assertEqual(3, len(ret))
        _check_prefix('b', ret)

        ret = x.similar_row_from_datum(Datum({'x': 11}), 3)
        self.assertEqual(3, len(ret))
        _check_prefix('b', ret)

        self.assertEqual(
            set(['a0', 'a1', 'a2', 'a3', 'b0', 'b1', 'b2', 'b3']),
            set(x.get_all_rows()))
        self.assertEqual(CONFIG, json.loads(x.get_config()))
        model = x.save_bytes()

        x = NearestNeighbor(CONFIG)
        x.load_bytes(model)
        self.assertEqual(
            set(['a0', 'a1', 'a2', 'a3', 'b0', 'b1', 'b2', 'b3']),
            set(x.get_all_rows()))

        st = x.get_status()
        self.assertTrue(isinstance(st, dict))
        self.assertEqual(len(st), 1)
        self.assertEqual(list(st.keys())[0], 'embedded')
        self.assertTrue(isinstance(st['embedded'], dict))
