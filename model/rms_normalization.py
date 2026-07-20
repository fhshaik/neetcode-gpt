import numpy as np
from typing import List


class Solution:
    def rms_norm(self, x: List[float], gamma: List[float], eps: float) -> List[float]:
        # Implement RMS Normalization (similar to LayerNorm but without mean centering or beta)
        # Normalize x, then scale by gamma
        # Return result rounded to 4 decimal places as a list
        rms = np.sqrt((np.square(np.array(x)) + eps).mean())
        x_hat = x/rms
        print(x_hat)
        print(gamma)
        out = np.dot(np.array(gamma), x_hat)
        print(out)
        return np.round(x_hat * np.array(gamma), 4)
        