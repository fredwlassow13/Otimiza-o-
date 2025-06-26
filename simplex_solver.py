import numpy as np
from scipy.optimize import linprog

def solve_lp(c, A, b):
    res = linprog(c=-np.array(c), A_ub=A, b_ub=b, method="highs")
    if res.success:
        shadow_prices = -res['ineqlin']['marginals']  # preços sombra
        return res.x, -res.fun, shadow_prices, res
    else:
        raise ValueError("Problema não tem solução ótima.")

def apply_variation(b, delta):
    return np.array(b) + np.array(delta)

def is_variation_valid(res, delta_b):
    return np.all(res['ineqlin']['marginals'] * delta_b >= -1e-5)
