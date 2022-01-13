class GaussianMLPInversePolicy(StochasticPolicy,LayersPowered,Serializable)
    def __init__(
            self,
            name,
            env_spec,
            hidden_sizes=(32,32)
            learn_std=1.0
            adaptive_std=False,
            std_share_network=False,
            std_hidden_sizes=(32,32),
            min_std=1e-6,
            std_hidden_nonlinearity=tf.nn.tanh,
                ):