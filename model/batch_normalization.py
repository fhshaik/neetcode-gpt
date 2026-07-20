import numpy as np
from typing import Tuple, List


class Solution:
    def batch_norm(self, x: List[List[float]], gamma: List[float], beta: List[float],
                   running_mean: List[float], running_var: List[float],
                   momentum: float, eps: float, training: bool) -> Tuple[List[List[float]], List[float], List[float]]:
        # During training: normalize using batch statistics, then update running stats
        # During inference: normalize using running stats (no batch stats needed)
        # Apply affine transform: y = gamma * x_hat + beta
        # Return (y, running_mean, running_var), all rounded to 4 decimals as lists
        x = np.array(x)
        gamma = np.array(gamma)
        beta = np.array(beta)
        running_mean = np.array(running_mean)
        running_var = np.array(running_var)
        print(x)
        mu = np.expand_dims(x.mean(axis = 0), axis=0)
        print(mu)
        var = np.expand_dims(np.square(x - mu).mean(axis = 0), axis = 0)
        print(var)
        x_hat = (x - mu)/np.sqrt(var + eps)
        y = np.round(gamma * x_hat + beta, 4)
        if training == False:
            x_hat = (x - running_mean)/np.sqrt(running_var + eps)
            y = np.round(gamma * x_hat + beta, 4)
            return (y.tolist(), running_mean.squeeze().tolist(), running_var.squeeze().tolist())
        else:
            m = momentum
            running_mean=np.round((1 - m)*running_mean+  m * mu, 4).squeeze()
            running_var=np.round((1 - m)*running_var +  m * var, 4).squeeze()

            return (y.tolist(), running_mean.tolist(), running_var.tolist())



        
