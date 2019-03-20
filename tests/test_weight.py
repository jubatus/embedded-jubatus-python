import json
import math
import unittest

from embedded_jubatus import Weight
from jubatus.common import Datum
from jubatus.weight.types import Feature


CONFIG = {
    "converter": {
        "num_filter_types": {},
        "num_filter_rules": [],
        "string_filter_types": {},
        "string_filter_rules": [],
        "num_types": {},
        "num_rules": [
            {"key": "n0", "type": "num"},
            {"key": "n1", "type": "log"},
            {"key": "n2", "type": "str"}
        ],
        "string_types": {},
        "string_rules": [
            {"key": "t0", "type": "space",
             "sample_weight": "bin", "global_weight": "bin"},
            {"key": "t1", "type": "space",
             "sample_weight": "tf", "global_weight": "idf1"}
        ]
    }
}


class TestWeight(unittest.TestCase):
    def test(self):
        w = Weight(CONFIG)
        d = Datum({'n0': 1, 'n1': 2, 'n2': 3, 't0': 'hello world'})
        for r in [w.update(d), w.calc_weight(d)]:
            self.assertEqual(5, len(r))
            for x in r:
                self.assertTrue(isinstance(x, Feature))
            m = dict([(x.key, x.value) for x in r])
            self.assertEqual(5, len(m))
            self.assertEqual(1.0, m['n0@num'])
            self.assertAlmostEqual(math.log(2), m['n1@log'])
            self.assertEqual(1.0, m['n2@str$3'])

        w.update(Datum({'t1': 'hello world'}))
        w.update(Datum({'t1': 'foo bar'}))
        w.update(Datum({'t1': 'hello'}))
        d = Datum({'t1': 'foo bar hello world hoge'})
        r0 = dict([(x.key, x.value) for x in w.calc_weight(d)])

        model = w.save_bytes()
        w = Weight(CONFIG)
        w.load_bytes(model)
        self.assertEqual(CONFIG, json.loads(w.get_config()))
        r1 = dict([(x.key, x.value) for x in w.calc_weight(d)])
        self.assertEqual(r0, r1)

        st = w.get_status()
        self.assertTrue(isinstance(st, dict))
        self.assertEqual(len(st), 1)
        self.assertEqual(list(st.keys())[0], 'embedded')
        self.assertTrue(isinstance(st['embedded'], dict))
