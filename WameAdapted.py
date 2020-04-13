from keras.optimizers import Optimizer
from keras import backend as K


class WameAdapted(Optimizer):
    """
    WAME algorithm as described in Mosca Maglouas, with a changed update:
    update now follows the one used by Mosca (see WameMosca.py).
    """
    def __init__(self, lr=0.001, beta=0.9, eta_plus=1.2, eta_minus=0.1, zeta_min=0.01, zeta_max=100, **kwargs):
        super(WameAdapted, self).__init__(**kwargs)
        self.iterations = K.variable(0)

        self.lr = K.variable(lr)
        self.beta = K.variable(beta)
        self.eta_plus = K.variable(eta_plus)
        self.eta_minus = K.variable(eta_minus)
        self.zeta_min = zeta_min
        self.zeta_max = zeta_max

    def get_updates(self, params, loss):
        grads = self.get_gradients(loss, params)
        self.updates = [K.update_add(self.iterations, 1)]

        shapes = [K.get_variable_shape(p) for p in params]
        prev_grads = [K.zeros(shape) for shape in shapes]
        thetas = [K.zeros(shape) for shape in shapes]
        zetas = [K.ones(shape) for shape in shapes]
        zs = [K.zeros(shape) for shape in shapes]  # This is different to the papers

        for param, grad, theta, zeta, z, prev_grad in zip(params, grads, thetas, zetas, zs, prev_grads):
            # Line 4 to 8
            zeta_new = K.switch(
                K.greater(grad * prev_grad, 0),
                K.minimum(zeta * self.eta_plus, self.zeta_max),
                K.switch(
                    K.less(grad * prev_grad, 0),
                    K.maximum(zeta * self.eta_minus, self.zeta_min),
                    zeta
                )
            )  # note that I added a 'if gradient = 0 then zeta' condition
            # Line 9
            z_new = (self.beta * z) + (1. - self.beta) * zeta_new
            # Line 10
            theta_new = (self.beta * theta) + (1. - self.beta) * K.square(grad)
            # Line 11
            weight_delta = - self.lr / z_new * grad / (K.sqrt(theta_new) + 1e-11)
            # Line 12
            new_param = param + weight_delta

            self.updates.append(K.update(theta, theta_new))
            self.updates.append(K.update(param, new_param))
            self.updates.append(K.update(prev_grad, grad))
            self.updates.append(K.update(zeta, zeta_new))
            self.updates.append(K.update(z, z_new))
        return self.updates

    def get_config(self):
        config = {'lr': float(K.get_value(self.lr)),
                  'beta': float(K.get_value(self.beta)),
                  'eta_plus': float(K.get_value(self.eta_plus)),
                  'eta_minus': float(K.get_value(self.eta_minus)),
                  'zeta_min': float(K.get_value(self.zeta_min)),
                  'zeta_max': float(self.zeta_max)}
        base_config = super(WameAdapted, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

