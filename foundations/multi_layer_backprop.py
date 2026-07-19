import numpy as np
from typing import List


class Solution:
    def forward_and_backward(self,
                              x: List[float],
                              W1: List[List[float]], b1: List[float],
                              W2: List[List[float]], b2: List[float],
                              y_true: List[float]) -> dict:
        # Architecture: x -> Linear(W1, b1) -> ReLU -> Linear(W2, b2) -> predictions
        # Loss: MSE = mean((predictions - y_true)^2)
        #
        # Return dict with keys:
        #   'loss':  float (MSE loss, rounded to 4 decimals)
        #   'dW1':   2D list (gradient w.r.t. W1, rounded to 4 decimals)
        #   'db1':   1D list (gradient w.r.t. b1, rounded to 4 decimals)
        #   'dW2':   2D list (gradient w.r.t. W2, rounded to 4 decimals)
        #   'db2':   1D list (gradient w.r.t. b2, rounded to 4 decimals)

        x = np.array(x, dtype=float)
        W1 = np.array(W1, dtype=float)
        b1 = np.array(b1, dtype=float)
        W2 = np.array(W2, dtype=float)
        b2 = np.array(b2, dtype=float)
        y_true = np.array(y_true, dtype=float)

        
        x_1 = W1 @ x + b1
        print(x_1.shape, W1.shape)
        x_2 = np.maximum(x_1, 0)
        print(x_2.shape, W2.shape)
        y_pred = W2 @ x_2 + b2

        L = np.mean((y_pred - y_true) ** 2)

        dy = (2 / len(y_pred)) * (y_pred - y_true)


        dW2 = np.outer(dy, x_2)
        db2 = dy

        dx2 = dy @ W2

        relu_mask = (x_1 > 0).astype(float)
        dx1 = dx2 * relu_mask

        dW1 = np.outer(dx1, x)
        db1 = dx1


        out = {
          'loss':  np.round(L, 4),
          'dW1':   np.round(dW1, 4),
          'db1':   np.round(db1, 4),
          'dW2':   np.round(dW2, 4),
          'db2':   np.round(db2, 4)
        }
        return out

        
