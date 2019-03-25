cdef class Bandit(_JubatusBase):
    cdef _Bandit *_handle

    def __cinit__(self):
        self._handle = NULL

    def __dealloc__(self):
        if self._handle != NULL:
            del self._handle

    def _init(self, config):
        self._handle = new _Bandit(config)
        self._type, self._model_ver = b'bandit', 1

    def get_config(self):
        return self._handle.get_config().decode('utf8')

    def save_bytes(self):
        return self._handle.dump(self._type, self._model_ver)

    def load_bytes(self, x):
        return self._handle.load(x, self._type, self._model_ver)

    def clear(self):
        self._handle.clear()
        return True

    def register_arm(self, arm_id):
        return self._handle.register_arm(arm_id.encode('utf8'))

    def delete_arm(self, arm_id):
        return self._handle.delete_arm(arm_id.encode('utf8'))

    def select_arm(self, player_id):
        return self._handle.select_arm(player_id.encode('utf8')).decode('utf8')

    def register_reward(self, player_id, arm_id, reward):
        return self._handle.register_reward(
            player_id.encode('utf8'), arm_id.encode('utf8'), reward)

    def get_arm_info(self, player_id):
        cdef map[string, arm_info] r
        r = self._handle.get_arm_info(player_id.encode('utf8'))
        return {
            it.first.decode('utf8'): ArmInfo(it.second.trial_count, it.second.weight)
            for it in r
        }

    def reset(self, player_id):
        return self._handle.reset(player_id.encode('utf8'))

    def get_status(self):
        cdef status_t status = self._handle.get_status()
        return {
            st.first.decode('utf8'): {
                it.first.decode('utf8'): it.second.decode('utf8')
                for it in st.second
            }
            for st in status
        }
