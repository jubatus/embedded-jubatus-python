import json
import unittest

from embedded_jubatus import Clustering
from jubatus.clustering.types import WeightedDatum
from jubatus.clustering.types import WeightedIndex
from jubatus.clustering.types import IndexedPoint
from jubatus.common import Datum


CONFIG = {
    'method': 'kmeans',
    'parameter': {
        'k': 2,
        'seed': 0,
    },
    'compressor_method': 'simple',
    'compressor_parameter': {
        'bucket_size': 6,
    },
    'distance': 'euclidean',
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
    }
}


class TestClustering(unittest.TestCase):
    def test_invalid_configs(self):
        self.assertRaises(TypeError, Clustering)
        self.assertRaises(Exception, Clustering, {})
        self.assertRaises(Exception, Clustering, {'method': 'hoge'})
        self.assertRaises(Exception, Clustering,
                          {'method': 'hoge', 'converter': {}})
        self.assertRaises(Exception, Clustering,
                          {'method': 'hoge', 'converter': {},
                           'compressor_method': 'hoge'})
        self.assertRaises(RuntimeError, Clustering,
                          {'method': 'hoge', 'converter': {},
                           'compressor_method': 'hoge',
                           'compressor_parameter': {}})
        self.assertRaises(RuntimeError, Clustering,
                          {'method': 'hoge', 'converter': {},
                           'compressor_method': 'hoge',
                           'compressor_parameter': {},
                           'distance': 'hoge'})
        invalid_config = dict(CONFIG)
        invalid_config['method'] = 'hoge'
        self.assertRaises(RuntimeError, Clustering, invalid_config)

    def test_default_configs(self):
        omitted_config = dict(CONFIG)
        del omitted_config['distance']
        try:
            x = Clustering(omitted_config)
        except:
            self.fail()
        self.assertTrue(True)

    def test(self):
        x = Clustering(CONFIG)
        self.assertEqual(0, x.get_revision())
        self.assertTrue(x.push([
            IndexedPoint('a', Datum({'x': 1.0})),
            IndexedPoint('b', Datum({'x': 0.9})),
            IndexedPoint('c', Datum({'x': 1.1})),
            IndexedPoint('d', Datum({'x': 5.0})),
            IndexedPoint('e', Datum({'x': 4.9})),
            IndexedPoint('f', Datum({'x': 5.1})),
        ]))
        self.assertEqual(1, x.get_revision())
        centers = x.get_k_center()
        self.assertTrue(isinstance(centers, list))
        self.assertEqual(2, len(centers))
        self.assertTrue(isinstance(centers[0], Datum))
        members = x.get_core_members()
        self.assertTrue(isinstance(members, list))
        self.assertEqual(2, len(members))
        self.assertTrue(isinstance(members[0], list))
        self.assertTrue(isinstance(members[0][0], WeightedDatum))
        c = x.get_nearest_center(Datum({'x': 1.05}))
        self.assertTrue(isinstance(c, Datum))
        self.assertTrue(c.num_values[0][1] >= 0.9 and
                        c.num_values[0][1] <= 1.1)
        c = x.get_nearest_members(Datum({'x': 1.05}))
        self.assertTrue(isinstance(c, list))
        self.assertTrue(isinstance(c[0], WeightedDatum))

        c = x.get_core_members_light()
        self.assertTrue(isinstance(c, list))
        self.assertTrue(isinstance(c[0], list))
        self.assertTrue(isinstance(c[0][0], WeightedIndex))

        c = x.get_nearest_members_light(Datum({'x': 1.05}))
        self.assertTrue(isinstance(c, list))
        self.assertTrue(isinstance(c[0], WeightedIndex))

        model = x.save_bytes()
        x = Clustering(CONFIG)
        x.load_bytes(model)

        self.assertEqual(CONFIG, json.loads(x.get_config()))
        self.assertEqual(1, x.get_revision())
        self.assertEqual(len(centers), len(x.get_k_center()))

        st = x.get_status()
        self.assertTrue(isinstance(st, dict))
        self.assertEqual(len(st), 1)
        self.assertEqual(list(st.keys())[0], 'embedded')
        self.assertTrue(isinstance(st['embedded'], dict))
