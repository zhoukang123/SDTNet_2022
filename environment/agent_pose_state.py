class AgentPoseState:
    """ Representation of a simple state of a Thor Agent which includes
        the position, horizon and rotation. """

    def __init__(self, x, y, z, rotation=0, horizon=0):
        self.x = round(x, 2)
        self.y = round(y, 2)
        self.z = round(z, 2)
        self.rotation = round(rotation)
        self.horizon = round(horizon)
        #默认为z轴为旋转参考轴，y轴为高度轴。其他数据集坐标如果不一样写一个转换类转换。

    def __eq__(self, other):
        """ If we check for exact equality then we get issues.
            For now we consider this 'close enough'. """
        if isinstance(other, AgentPoseState):
            return (
                self.x == other.x
                and
                # self.y == other.y and
                self.z == other.z
                and self.rotation == other.rotation
                and self.horizon == other.horizon
            )
        return NotImplemented

    def __str__(self):
        """ Get the string representation of a state. """
        """
        return '{:0.2f}|{:0.2f}|{:d}|{:d}'.format(
            self.x,
            self.z,
            round(self.rotation),
            round(self.horizon)
        )
        """
        return "{:0.2f}|{:0.2f}|{:d}|{:d}".format(
            self.x, self.z, round(self.rotation), round(self.horizon)
        )

    def position(self):
        """ Returns just the position. """
        return dict(x=self.x, y=self.y, z=self.z)

def get_state_from_str(pose_str, y = 0.91):
    temp = [float(x) for x in pose_str.split("|")]
    temp.insert(1, y)
    return AgentPoseState(*temp)
        