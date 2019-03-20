import json
import os
import pickle
import tempfile
import unittest
import sys

from embedded_jubatus import Classifier
from jubatus.classifier.types import EstimateResult
from jubatus.classifier.types import LabeledDatum
from jubatus.common import Datum
import numpy as np


CONFIG = {
    "method": "perceptron",
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
    },
    "parameter": {}
}


class TestClassifier(unittest.TestCase):
    def test_invalid_configs(self):
        self.assertRaises(TypeError, Classifier)
        self.assertRaises(ValueError, Classifier, {})
        self.assertRaises(TypeError, Classifier, {'method': 'hoge'})
        self.assertRaises(RuntimeError, Classifier,
                          {'method': 'hoge', 'converter': {}})
        invalid_config = dict(CONFIG)
        invalid_config['method'] = 'hoge'
        self.assertRaises(RuntimeError, Classifier, invalid_config)

    def test_num(self):
        x = Classifier(CONFIG)
        self.assertEqual(2, x.train([
            ('Y', Datum({'x': 1})),
            ('N', Datum({'x': -1})),
        ]))

        def _test_classify(x):
            y = x.classify([
                Datum({'x': 1}),
                Datum({'x': -1})
            ])
            self.assertEqual(['Y', 'N'], [list(sorted(
                z, key=lambda x:x.score, reverse=True))[0].label for z in y])
            self.assertEqual(x.get_labels(), {'N': 1, 'Y': 1})

        _test_classify(x)
        model = x.save_bytes()

        self.assertTrue(x.clear())
        self.assertEqual({}, x.get_labels())
        x.set_label('Y')
        x.set_label('N')
        self.assertEqual({'N': 0, 'Y': 0}, x.get_labels())
        x.delete_label(u'Y')
        self.assertEqual({'N': 0}, x.get_labels())

        x = Classifier(CONFIG)
        x.load_bytes(model)
        _test_classify(x)
        self.assertEqual(CONFIG, json.loads(x.get_config()))

        if sys.version_info[0] == 3:
            x = pickle.loads(pickle.dumps(x))
            _test_classify(x)
            self.assertEqual(CONFIG, json.loads(x.get_config()))

        st = x.get_status()
        self.assertTrue(isinstance(st, dict))
        self.assertEqual(len(st), 1)
        self.assertEqual(list(st.keys())[0], 'embedded')
        self.assertTrue(isinstance(st['embedded'], dict))

    def test_str(self):
        x = Classifier(CONFIG)
        self.assertEqual(2, x.train([
            ('Y', Datum({'x': 'y'})),
            ('N', Datum({'x': 'n'})),
        ]))
        y = x.classify([
            Datum({'x': 'y'}),
            Datum({'x': 'n'})
        ])
        self.assertEqual(['Y', 'N'], [list(sorted(
            z, key=lambda x:x.score, reverse=True))[0].label for z in y])

    def test_types(self):
        x = Classifier(CONFIG)
        x.train([
            LabeledDatum('Y', Datum({'x': 'y'})),
            LabeledDatum('N', Datum({'x': 'n'})),
        ])
        y = x.classify([
            Datum({'x': 'y'}),
            Datum({'x': 'n'})
        ])
        self.assertTrue(isinstance(y[0][0], EstimateResult))
        self.assertEqual(['Y', 'N'], [list(sorted(
            z, key=lambda x:x.score, reverse=True))[0].label for z in y])

    def test_loadsave(self):
        x = Classifier(CONFIG)
        x.train([
            LabeledDatum('Y', Datum({'x': 'y'})),
            LabeledDatum('N', Datum({'x': 'n'})),
        ])
        path = '/tmp/127.0.0.1_0_classifier_hoge.jubatus'

        def _remove_model():
            try:
                os.remove(path)
            except Exception:
                pass

        _remove_model()
        try:
            self.assertEqual({'127.0.0.1_0': '/tmp/127.0.0.1_0_classifier_hoge.jubatus'}, x.save('hoge'))
            self.assertTrue(os.path.isfile(path))
            x = Classifier(CONFIG)
            self.assertTrue(x.load('hoge'))
            y = x.classify([
                Datum({'x': 'y'}),
                Datum({'x': 'n'})
            ])
            self.assertEqual(['Y', 'N'], [list(sorted(
                z, key=lambda x:x.score, reverse=True))[0].label for z in y])
        finally:
            _remove_model()

    def test_numpy(self):
        x = Classifier(CONFIG)
        tdata = np.array([
            [1, 0, 1],
            [0, 1, 1],
        ], dtype='f8')
        ttargets = np.array([1, 0])
        x.partial_fit(tdata, ttargets)
        y = x.predict(np.array([
            [1, 0, 0],
            [0, 1, 0],
        ], dtype='f8'))
        self.assertEqual(1, y[0])
        self.assertEqual(0, y[1])
        y = x.decision_function(np.array([
            [1, 0, 0],
            [0, 1, 0],
        ], dtype='f8'))
        self.assertEqual(2, len(y))
        self.assertTrue(not isinstance(y[0], (list, tuple, np.ndarray)))
        self.assertTrue(y[0] > 0)
        self.assertTrue(y[1] < 0)
        self.assertEqual([0, 1], sorted(x.classes_))

        model = x.save_bytes()
        x = Classifier(CONFIG)
        x.load_bytes(model)
        self.assertEqual([0, 1], sorted(x.classes_))
        y = x.predict(np.array([
            [1, 0, 0],
            [0, 1, 0],
        ], dtype='f8'))
        self.assertEqual(1, y[0])
        self.assertEqual(0, y[1])

    def test_sparse(self):
        from scipy.sparse import csr_matrix

        x = Classifier(CONFIG)
        tdata = csr_matrix(np.array([
            [1, 0, 1],
            [0, 1, 1],
        ], dtype='f8'))
        ttargets = np.array([1, 0])
        x.partial_fit(tdata, ttargets)
        y = x.predict(csr_matrix(np.array([
            [1, 0, 0],
            [0, 1, 0],
        ], dtype='f8')))
        self.assertEqual(1, y[0])
        self.assertEqual(0, y[1])
        y = x.decision_function(csr_matrix(np.array([
            [1, 0, 0],
            [0, 1, 0],
        ], dtype='f8')))
        self.assertEqual(2, len(y))
        self.assertTrue(not isinstance(y[0], (list, tuple, np.ndarray)))
        self.assertTrue(y[0] > 0)
        self.assertTrue(y[1] < 0)

    def test_multilabel(self):
        x = Classifier(CONFIG)
        tdata = np.array([
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 0],
        ], dtype='f8')
        ttargets = np.array([1, 2, 0])
        x.partial_fit(tdata, ttargets)
        y = x.predict(np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
        ], dtype='f8'))
        self.assertTrue(isinstance(y, np.ndarray))
        self.assertEqual(y.dtype, np.int32)
        self.assertTrue(np.equal(y, np.array([0, 1, 2])).all())
        y = x.decision_function(np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
        ], dtype='f8'))
        self.assertTrue(isinstance(y, np.ndarray))
        self.assertEqual((3, 3), y.shape)
        self.assertTrue(y[0][0] > y[0][1] and y[0][0] > y[0][2])
        self.assertTrue(y[1][1] > y[1][0] and y[1][1] > y[1][2])
        self.assertTrue(y[2][2] > y[2][0] and y[2][2] > y[2][1])

    def test_load_config_from_file(self):
        with tempfile.NamedTemporaryFile() as fd:
            fd.write(json.dumps(CONFIG).encode('utf8'))
            fd.flush()

            x = Classifier(fd.name)
            self.assertEqual(CONFIG, json.loads(x.get_config()))

    def test_load_invalid_config_from_file(self):
        with tempfile.NamedTemporaryFile() as fd:
            fd.write(b'{"hoge')
            fd.flush()
            self.assertRaises(ValueError, Classifier, fd.name)

    def test_issue_30(self):
        path = '/tmp/127.0.0.1_0_classifier_foo.jubatus'
        open(path, 'wb').close()
        try:
            Classifier({'method': 'perceptron', 'converter': {}}).load('foo')
        except Exception:
            pass
        finally:
            os.remove(path)
