class ScalarMeanTracker(object):
    """用来算平均数的一个类，记录数据用"""
    def __init__(self) -> None:
        self._sums = {}
        self._counts = {}

    def add_scalars(self, scalars, count = True):
        for k in scalars:
            if k not in self._sums:
                self._sums[k] = scalars[k]
                self._counts[k] = 1
            else:
                self._sums[k] += scalars[k]
                self._counts[k] += count

    def pop_and_reset(self, no_div_list = []):
        for k in no_div_list:
            self._counts[k] = 1
        means = {k: self._sums[k] / self._counts[k] for k in self._sums}
        self._sums = {}
        self._counts = {}
        return means

class LabelScalarTracker(object):
    """带标签的用来算平均数的一个类，记录数据用"""
    def __init__(self):
        self.trackers = {}

    def __getitem__(self, key):
        if key in self.trackers:
            return self.trackers[key]
        else:
            self.trackers[key] = ScalarMeanTracker()
            return self.trackers[key]

    def items(self):
        return self.trackers.items()

    def pop_and_reset(self, no_div_list = []):
        out = {}
        for k in self.trackers:
            out[k] = self.trackers[k].pop_and_reset(no_div_list)
        self.trackers = {}
        return out
