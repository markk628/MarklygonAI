import numpy as np
from collections import deque

class RollingMinMaxScaler:
    """
    Min-Max Scaler that operates on a rolling window.
    Scales data to the range [0, 1].
    """
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.window = deque(maxlen=window_size)
        self._min_val = None
        self._max_val = None

    def partial_fit(self, X: np.ndarray):
        """
        Updates the scaler's internal window and min/max values with new data
        X should be a 1D numpy array-like (e.g., [value])
        """
        if X.shape[0] != 1:
            raise ValueError("Input X must be a 1D array with a single value, e.g., np.array([value]).")

        new_value = X[0]
        self.window.append(new_value)
        self._min_val = min(self.window)
        self._max_val = max(self.window)

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Scales the input data X using the current min/max of the rolling window
        X should be a 1D numpy array-like (e.g., [value])
        """
        if X.shape[0] != 1:
            raise ValueError("Input X must be a 1D array with a single value, e.g., np.array([value]).")

        value_to_scale = X[0]

        if self._min_val is None or len(self.window) == 0:
            return np.array([0.0])

        if self._max_val == self._min_val:
            return np.array([0.0])

        scaled_0_1 = (value_to_scale - self._min_val) / (self._max_val - self._min_val)
        scaled_neg1_1 = 2 * scaled_0_1 - 1

        return np.array([np.clip(scaled_neg1_1, -1.0, 1.0)])
    
    def partial_fit_transform(self, X: np.ndarray) -> np.ndarray:
        self.partial_fit(X)
        return self.transform(X)

    def get_window_stats(self):
        return self._min_val, self._max_val, len(self.window)
    
if __name__ == "__main__":
    prices = [100, 101, 102, 100, 99, 105, 110, 108, 112, 115, 116, 118, 117, 115, 120, 122, 125, 123, 128, 130] 
    volumes = [1000, 1200, 900, 1500, 1100, 800, 1300]

    price_scaler = RollingMinMaxScaler(window_size=5)
    volume_scaler = RollingMinMaxScaler(window_size=5)

    print("scaling price data...")
    for i, price in enumerate(prices):
        scaled_price = price_scaler.partial_fit_transform(np.array([price]))[0]
        min_win, max_win, len_win = price_scaler.get_window_stats()
        print(f"Step {i+1}: Price={price}, "
              f"Window Min={min_win:.2f}, Window Max={max_win:.2f}, "
              f"Window Len={len_win}, Scaled Price={scaled_price:.4f}")

    for i, vol in enumerate(volumes):
        scaled_vol = volume_scaler.partial_fit_transform(np.array([vol]))[0]
        min_win, max_win, len_win = volume_scaler.get_window_stats()
        print(f"Step {i+1}: Volume={vol}, "
              f"Window Min={min_win:.2f}, Window Max={max_win:.2f}, "
              f"Window Len={len_win}, Scaled Volume={scaled_vol:.4f}")