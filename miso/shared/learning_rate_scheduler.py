import time
import numpy as np
import scipy.stats as stats


class RollingBuffer:
    def __init__(self, buffer_len):
        self.__buffer = np.zeros(buffer_len)
        self.__counter = 0
        self.__buffer_len = buffer_len

    def append(self, data):
        self.__buffer = np.roll(self.__buffer, -1)
        self.__buffer[-1] = data
        self.__counter += 1
        if self.__counter > self.__buffer_len:
            self.__counter = self.__buffer_len

    def values(self):
        return self.__buffer[-self.__counter:]

    def mean(self):
        return np.sum(self.__buffer) / self.__counter

    def indices(self):
        return range(self.__counter)

    def clear(self):
        self.__counter = 0

    def length(self):
        return self.__buffer_len

    def full(self):
        return self.__counter == self.__buffer_len

    def slope_probability_less_than(self, prob):
        idxs = self.indices()
        n = len(idxs)
        if n < 3:
            return 1
        values = self.values()
        n = float(n)
        slope, intercept, r_value, p_value, std_err = stats.linregress(idxs, values)
        residuals = idxs * slope + intercept
        variance = np.sum(np.power(residuals - values, 2)) / (n - 2)
        slope_std_error = np.sqrt(variance * (12.0 / (np.power(n, 3) - n)))
        p_less_than_zero = stats.norm.cdf(prob, slope, slope_std_error)
        return p_less_than_zero


class AdaptiveLearningRateScheduler(object):
    def __init__(self,
                 optimizer,
                 factor=0.5,
                 nb_drops=4,
                 nb_epochs=10,
                 startup_delay_factor=2,
                 verbose=True):
        super(AdaptiveLearningRateScheduler, self).__init__()
        self.optimizer = optimizer
        self.factor = factor
        self.nb_drops = nb_drops
        self.nb_epochs = nb_epochs
        self.startup_delay_factor = startup_delay_factor
        self.verbose = verbose

        self.drop_count = 0
        self.buffer = RollingBuffer(self.nb_epochs)

        self.finished = False

    def step(self, epoch, loss):
        # Check learning rate
        if self.needs_update_lr(epoch, loss):
            self.reduce_lr(epoch)
            self.drop_count += 1
        if self.verbose:
            print("-" * 80)
        # Finished training?
        return self.drop_count >= self.nb_drops

    def needs_update_lr(self, epoch, loss):
        self.buffer.append(loss)

        if epoch < self.startup_delay_factor * self.nb_epochs:
            if self.verbose:
                print("-" * 80)
                print(f"Epoch: [{epoch}]  loss: {loss:04f}, warmup")
            return False

        prob = self.buffer.slope_probability_less_than(0)

        if self.verbose:
            print("-" * 80)
            print(f"Epoch: [{epoch}]  phase: {self.drop_count}, loss: {loss:04f}, prob: {prob:04f}, buffer full? {self.buffer.full()}")

        if self.buffer.full() and prob < 0.50:
            return True

        return False

    def reduce_lr(self, epoch):
        for i, param_group in enumerate(self.optimizer.param_groups):
            old_lr = float(param_group['lr'])
            new_lr = old_lr * self.factor
            param_group['lr'] = new_lr
            if self.verbose:
                epoch_str = ("%.2f" if isinstance(epoch, float) else
                             "%.5d") % epoch
                print('Epoch {}: reducing learning rate'
                      ' of group {} to {:.4e}.'.format(epoch_str, i, new_lr))