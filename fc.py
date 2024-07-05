import numpy as np


class SquareLoss:
    def forward(self, y, t):
        self.loss = 0.5 * np.sum((y - t) ** 2)

    def backward(self):
        return self.loss


class FC:
    
    def __init__(self, in_num, out_num, lr=0.1):
        self._in_num = in_num
        self._out_num = out_num
        self.w = np.random.randn(in_num, out_num)
        self.b = np.zeros((out_num, 1))
        self.lr = lr

    def _sigmoid(self, in_data):
        return 1 / (1 + np.exp(-in_data))

    def forward(self, in_data):
        self.top_val = self._sigmoid(np.dot(self.w.T, in_data) + self.b)
        self.bottom_val = in_data
        return self.top_val

    def backward(self, loss):
        
        residual_z = loss * self.top_val * (1 - self.top_val)
        grad_w = np.dot(self.bottom_val, residual_z.T)
        grad_b = np.sum(residual_z)
        self.w -= self.lr * grad_w
        self.b -= self.lr * grad_b
        residual_x = np.dot(self.w, residual_z)
        return residual_x
