import json
import random
import unittest

from embedded_jubatus import Bandit
from jubatus.bandit.types import ArmInfo


CONFIG = {
    'method': 'epsilon_greedy',
    'parameter': {
        'assume_unrewarded': False,
        'epsilon': 0.1
    }
}


class TestBandit(unittest.TestCase):
    def test_invalid_configs(self):
        self.assertRaises(TypeError, Bandit)
        self.assertRaises(ValueError, Bandit, {})
        self.assertRaises(RuntimeError, Bandit, {'method': 'hoge'})
        invalid_config = dict(CONFIG)
        invalid_config['parameter'] = {'hoge': 0.1}
        self.assertRaises(RuntimeError, Bandit, invalid_config)

    def test(self):
        x = Bandit(CONFIG)

        player = 'player'
        slots = {
            'a': [
                lambda: random.random() < 0.1,
                lambda: random.normalvariate(50, 10),
            ],
            'b': [
                lambda: random.random() < 0.01,
                lambda: random.normalvariate(600, 100),
            ],
            'c': [
                lambda: random.random() < 0.001,
                lambda: random.normalvariate(8000, 1000),
            ],
        }
        keys = list(slots.keys())
        for k in keys:
            self.assertTrue(x.register_arm(k))
        self.assertFalse(x.register_arm(keys[0]))
        self.assertFalse(x.reset(player))

        for _ in range(10):
            arm = x.select_arm(player)
            f0, f1 = slots[arm]
            self.assertTrue(arm in keys)
            x.register_reward(player, arm, f1() if f0() else 0.0)
        info = x.get_arm_info(player)
        self.assertEqual(3, len(info))
        self.assertTrue(isinstance(info[keys[0]], ArmInfo))

        model = x.save_bytes()
        x = Bandit(CONFIG)
        x.load_bytes(model)
        self.assertEqual(CONFIG, json.loads(x.get_config()))
        info = x.get_arm_info(player)
        self.assertEqual(3, len(info))
        self.assertTrue(isinstance(info[keys[0]], ArmInfo))

        st = x.get_status()
        self.assertTrue(isinstance(st, dict))
        self.assertEqual(len(st), 1)
        self.assertEqual(list(st.keys())[0], 'embedded')
        self.assertTrue(isinstance(st['embedded'], dict))
