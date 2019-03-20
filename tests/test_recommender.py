import json
import unittest

from embedded_jubatus import Recommender
from jubatus.common import Datum
from jubatus.recommender.types import IdWithScore


CONFIG = {
    'method': 'lsh',
    'parameter': {
        'hash_num': 512,
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


class TestRecommender(unittest.TestCase):
    def test_invalid_config(self):
        self.assertRaises(TypeError, Recommender)
        self.assertRaises(ValueError, Recommender, {})
        self.assertRaises(TypeError, Recommender, {'method': 'hoge'})
        self.assertRaises(RuntimeError, Recommender,
                          {'method': 'hoge', 'converter': {}})
        invalid_config = dict(CONFIG)
        invalid_config['method'] = 'hoge'
        self.assertRaises(RuntimeError, Recommender, invalid_config)

    def test(self):
        def _valid_result(r):
            self.assertTrue(isinstance(r, Datum))
            d = dict(r.num_values)
            self.assertTrue(
                d.get('x', None) is not None and d.get('y', None) is not None)

        x = Recommender(CONFIG)
        x.update_row('0', Datum({'x': 0.9, 'y': 4.9}))
        x.update_row('1', Datum({'x': 1, 'y': 5}))
        x.update_row('2', Datum({'x': 1.15, 'y': 5.15}))
        x.update_row('3', Datum({'x': 1.2, 'y': 5.1}))
        x.update_row('4', Datum({'x': 1.05}))
        _valid_result(x.complete_row_from_datum(Datum({'x': 1.1})))
        _valid_result(x.complete_row_from_id('4'))
        r = x.similar_row_from_id('2', 3)
        self.assertTrue(isinstance(r, list))
        self.assertTrue(isinstance(r[0], IdWithScore))
        r = x.similar_row_from_id_and_score('2', 0.0)
        self.assertTrue(isinstance(r, list))
        self.assertTrue(isinstance(r[0], IdWithScore))
        r = x.similar_row_from_id_and_rate('2', 1.0)
        self.assertTrue(isinstance(r, list))
        self.assertTrue(isinstance(r[0], IdWithScore))
        r = x.similar_row_from_datum(Datum({'y': 5.05}), 3)
        self.assertTrue(isinstance(r, list))
        self.assertTrue(isinstance(r[0], IdWithScore))
        r = x.similar_row_from_datum_and_score(Datum({'y': 5.05}), 0.0)
        self.assertTrue(isinstance(r, list))
        self.assertTrue(isinstance(r[0], IdWithScore))
        r = x.similar_row_from_datum_and_rate(Datum({'y': 5.05}), 1.0)
        self.assertTrue(isinstance(r, list))
        self.assertTrue(isinstance(r[0], IdWithScore))
        _valid_result(x.decode_row('0'))
        assert set(x.get_all_rows()) == set([str(i) for i in range(5)])
        r = x.calc_similarity(Datum({'x': 1}), Datum({'y': 5}))
        self.assertTrue(isinstance(r, float))
        r = x.calc_l2norm(Datum({'x': 1, 'y': 5}))
        self.assertTrue(isinstance(r, float))

        model = x.save_bytes()
        x.clear()
        self.assertEqual([], x.get_all_rows())
        self.assertEqual(CONFIG, json.loads(x.get_config()))

        x = Recommender(CONFIG)
        x.load_bytes(model)
        self.assertTrue(x.get_all_rows())
        _valid_result(x.complete_row_from_id('4'))

        st = x.get_status()
        self.assertTrue(isinstance(st, dict))
        self.assertEqual(len(st), 1)
        self.assertEqual(list(st.keys())[0], 'embedded')
        self.assertTrue(isinstance(st['embedded'], dict))
