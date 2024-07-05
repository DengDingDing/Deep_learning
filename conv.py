import numpy as np


def conv2d(x, w):
    x_row, x_col = x.shape
    w_row, w_col = w.shape
    ret_row, ret_col = x_row - w_row + 1, x_col - w_col + 1
    ret = np.zeros((ret_row, ret_col))
    for i in range(ret_row):
        for j in range(ret_col):
            ret[i, j] = np.sum(x[i : i + w_row, j : j + w_col] * w)
    return ret


class ConvLayer:
    def __init__(self, in_channel, out_channel, kernel_size):
        self.w = np.random.randn(out_channel, in_channel, kernel_size, kernel_size)
        self.b = np.zeros((out_channel, 1))

    def _relu(self, in_data):
        return np.maximum(0, in_data)

    def forward(self, in_data):
        out_channel, in_channel, kernel_size, _ = self.w.shape
        in_channel, in_row, in_col = in_data.shape
        ret_row, ret_col = in_row - kernel_size + 1, in_col - kernel_size + 1
        ret = np.zeros((out_channel, ret_row, ret_col))
        for i in range(out_channel):
            for j in range(in_channel):
                ret[i] += conv2d(in_data[j], self.w[i, j])
            ret[i] += self.b[i]
            ret[i] = self._relu(ret[i])
        return ret
