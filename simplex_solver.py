import cvxpy as cp
import numpy as np

def solve_lp(c, A, b, signs=None):
    A = np.array(A)
    b = np.array(b)
    c = np.array(c)

    n_vars = len(c)
    n_rest = len(b)

    x = cp.Variable(n_vars, nonneg=True)

    # Se sinais não forem dados, assuma <=
    if signs is None:
        signs = ['<='] * n_rest

    constraints = []
    for i in range(n_rest):
        if signs[i] == '<=':
            constraints.append(A[i] @ x <= b[i])
        elif signs[i] == '>=':
            constraints.append(A[i] @ x >= b[i])
        elif signs[i] == '=':
            constraints.append(A[i] @ x == b[i])
        else:
            raise ValueError(f"Sinal inválido na restrição {i+1}: {signs[i]}")

    objective = cp.Maximize(c @ x)
    problem = cp.Problem(objective, constraints)
    problem.solve()

    if problem.status not in ["optimal", "optimal_inaccurate"]:
        raise ValueError("Problema não possui solução ótima.")

    shadow_prices = [constr.dual_value if constr.dual_value is not None else 0.0 for constr in constraints]

    return x.value, problem.value, shadow_prices, problem

def apply_variation(b, delta):
    return np.array(b) + np.array(delta)

def is_variation_valid(res, delta_b):
    """
    Verifica se a direção da variação ainda mantém viabilidade
    com base no sinal dos preços-sombra.
    """
    duals = [constr.dual_value if constr.dual_value is not None else 0.0 for constr in res.constraints]
    return all((d * db >= -1e-6) for d, db in zip(duals, delta_b))
