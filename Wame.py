from keras.optimizers import Optimizer
from keras import backend as K
import numpy

__name__ = "wame"


class Wame(Optimizer):
    def __init__(self, alpha=0.9, eta_pos=1.2, eta_neg=0.1, zeta_min=0.01, zeta_max=100, **kwargs):
        super(Wame, self).__init__(**kwargs)
        self.alpha = K.variable(alpha, name='alpha')
        self.eta_pos = K.variable(eta_pos, name='eta_pos')
        self.eta_neg = K.variable(eta_neg, name='eta_neg')
        self.zeta_min = K.variable(zeta_min, name='zeta_min')
        self.zeta_max = K.variable(zeta_max, name='zeta_max')

    def get_updates(self, params, loss):
        grads = self.get_gradients(loss, params)
        shapes = [K.get_variable_shape(p) for p in params]
        old_grads = [K.zeros(shape) for shape in shapes]
        zetas = [K.ones(shape) for shape in shapes]
        zs = [K.zeros(shape) for shape in shapes]
        thetas = [K.zeros(shape) for shape in shapes]

        # prev_weight_deltas = [K.zeros(shape) for shape in shapes]
        # self.weights = delta_ws + old_grads # TODO: understand self.weights
        self.updates = []

        for param, grad, old_grad, zeta, z, theta in zip(params, grads, old_grads, zetas, zs, thetas):
            # Line 4 to 8
            new_zeta = K.switch(
                K.greater(grad * old_grad, 0),
                K.minimum(zeta * self.eta_pos, self.zeta_max),
                K.switch(
                    K.less(grad * old_grad, 0),
                    K.maximum(zeta * self.eta_neg, self.zeta_min),
                    zeta
                )
            )  # note that I added a 'if gradient = 0 then zeta' condition
            # Line 9
            new_z = self.alpha * z + (1 - self.alpha) * new_zeta
            # Line 10
            new_theta = self.alpha * theta + (1 - self.alpha) #* K.square(grad)
            # Line 11
            weight_delta = -0.1/new_z * grad * (1/new_theta)
            # TODO: Figure this out! It seems like the theta part in particular seems to be breaking the calculation
            #    Also, it seems like we should be taking the sign of grad rather than multiplying it directly.
            # weight_delta = -new_z * (grad/new_theta)
            # Line 12
            new_param = param + weight_delta

            # Apply constraints
            #if param in constraints:
            #    c = constraints[param]
            #    new_param = c(new_param)

            self.updates.append(K.update(param, new_param))
            self.updates.append(K.update(zeta, new_zeta))
            self.updates.append(K.update(old_grad, grad))
            self.updates.append(K.update(z, new_z))
            self.updates.append(K.update(theta, new_theta))

        return self.updates

    def get_config(self):
        config = {
            'alpha': float(K.get_value(self.alpha)),
            'eta_pos': float(K.get_value(self.eta_pos)),
            'eta_neg': float(K.get_value(self.eta_neg)),
            'zeta_min': float(K.get_value(self.zeta_min)),
            'zeta_max': float(K.get_value(self.zeta_max)),
        }
        base_config = super(Wame, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
