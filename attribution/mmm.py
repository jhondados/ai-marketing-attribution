"""Lightweight Media Mix Modeling."""
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from typing import Dict, List, Tuple

class MediaMixModel:
    """Simplified Robyn-style MMM with adstock and saturation."""
    def __init__(self):
        self.params = {}
        self.channels = []

    def adstock(self, spend: np.ndarray, decay: float) -> np.ndarray:
        """Apply adstock transformation (carryover effect)."""
        adstocked = np.zeros_like(spend)
        adstocked[0] = spend[0]
        for t in range(1, len(spend)):
            adstocked[t] = spend[t] + decay * adstocked[t-1]
        return adstocked

    def saturation(self, spend: np.ndarray, alpha: float, gamma: float) -> np.ndarray:
        """Hill saturation function (diminishing returns)."""
        return spend**alpha / (spend**alpha + gamma**alpha)

    def transform_channel(self, spend: np.ndarray, decay: float, alpha: float, gamma: float) -> np.ndarray:
        return self.saturation(self.adstock(spend, decay), alpha, gamma)

    def fit(self, media_df: pd.DataFrame, revenue: pd.Series) -> "MediaMixModel":
        self.channels = [c for c in media_df.columns]
        n_params = len(self.channels) * 3 + 2  # decay, alpha, gamma per channel + intercept + trend
        def objective(params):
            pred = params[0]  # intercept
            for i, ch in enumerate(self.channels):
                decay, alpha, gamma = params[i*3+1], params[i*3+2], params[i*3+3]
                pred = pred + self.transform_channel(media_df[ch].values, decay, alpha, gamma)
            return np.mean((pred - revenue.values)**2)
        x0 = [revenue.mean()] + [0.5, 0.5, np.mean(media_df.values)] * len(self.channels)
        bounds = [(0, None)] + [(0.1, 0.99), (0.1, 2.0), (1, None)] * len(self.channels)
        result = minimize(objective, x0, bounds=bounds, method="L-BFGS-B")
        self.params = {"intercept": result.x[0]}
        for i, ch in enumerate(self.channels):
            self.params[ch] = {"decay": result.x[i*3+1], "alpha": result.x[i*3+2], "gamma": result.x[i*3+3]}
        return self

    def budget_optimizer(self, total_budget: float, n_iter: int = 1000) -> Dict[str, float]:
        """Optimize budget allocation across channels."""
        if not self.params: raise ValueError("Fit model first")
        n = len(self.channels)
        def neg_revenue(alloc):
            total = 0
            for i, ch in enumerate(self.channels):
                p = self.params[ch]
                total += self.saturation(self.adstock(np.array([alloc[i]]), p["decay"]), p["alpha"], p["gamma"])[0]
            return -total
        x0 = [total_budget / n] * n
        bounds = [(total_budget * 0.05, total_budget * 0.6)] * n
        cons = [{"type": "eq", "fun": lambda x: sum(x) - total_budget}]
        result = minimize(neg_revenue, x0, bounds=bounds, constraints=cons)
        return {ch: round(float(alloc), 2) for ch, alloc in zip(self.channels, result.x)}
