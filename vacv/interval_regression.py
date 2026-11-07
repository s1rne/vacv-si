"""Convex kernel regression with exact and one-sided interval observations."""
import numpy as np
from scipy.optimize import minimize


def interval_loss_gradient(prediction, lower, upper, weight):
    """Squared distance to each interval; feasible observations exert no force."""
    residual = np.where(prediction < lower, prediction-lower,
                        np.where(prediction > upper, prediction-upper, 0.))
    return .5 * np.dot(weight, residual**2), weight*residual


class IntervalKernelRegressor:
    """PSD-kernel RKHS ridge with an unpenalized intercept, fitted on latent SI."""
    def __init__(self, alpha=1., max_iter=1500):
        if not np.isfinite(alpha) or alpha <= 0:
            raise ValueError("alpha must be finite and positive")
        self.alpha, self.max_iter = float(alpha), max_iter

    def fit(self, kernel, lower, upper, sample_weight=None):
        kernel = np.asarray(kernel, dtype=float)
        lower, upper = np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)
        n = len(lower)
        weight = np.ones(n) if sample_weight is None else np.asarray(sample_weight, dtype=float)
        if (not n or kernel.shape != (n, n) or lower.shape != (n,) or upper.shape != (n,) or weight.shape != (n,)
                or not np.isfinite(kernel).all() or np.isnan(lower).any() or np.isnan(upper).any()
                or not np.isfinite(weight).all() or (weight <= 0).any()
                or (lower > upper).any() or np.isposinf(lower).any() or np.isneginf(upper).any()
                or (~np.isfinite(lower) & ~np.isfinite(upper)).any()):
            raise ValueError("Aligned finite kernel and valid informative intervals required")
        if not np.allclose(kernel, kernel.T, atol=1e-10):
            raise ValueError("Symmetric kernel required")
        values, vectors = np.linalg.eigh((kernel+kernel.T)/2)
        if values.min() < -1e-7:
            raise ValueError("Positive semidefinite kernel required")
        keep = values > max(1e-10, values.max()*1e-10)
        values, vectors = values[keep], vectors[:, keep]
        phi = vectors * np.sqrt(values)
        exact = np.isfinite(lower) & (lower == upper)
        if not exact.any():
            raise ValueError("At least one exact observation anchors the intercept")
        initial = np.r_[np.zeros(len(values)), np.average(lower[exact], weights=weight[exact])]

        def objective(parameters):
            coefficients, intercept = parameters[:-1], parameters[-1]
            loss, gradient = interval_loss_gradient(phi@coefficients+intercept, lower, upper, weight)
            return (loss + .5*self.alpha*np.dot(coefficients, coefficients),
                    np.r_[phi.T@gradient+self.alpha*coefficients, gradient.sum()])

        result = minimize(objective, initial, jac=True, method="L-BFGS-B",
                          options={"maxiter": self.max_iter, "ftol": 1e-12, "gtol": 1e-7})
        if not result.success or not np.isfinite(result.x).all():
            raise RuntimeError(f"Interval fit did not converge: {result.message}")
        self.dual_ = (vectors / np.sqrt(values)) @ result.x[:-1]
        self.intercept_ = result.x[-1]
        self.n_train_, self.iterations_ = n, result.nit
        self.objective_ = float(result.fun)
        return self

    def predict(self, cross_kernel):
        cross_kernel = np.asarray(cross_kernel, dtype=float)
        if cross_kernel.ndim != 2 or cross_kernel.shape[1] != self.n_train_ or not np.isfinite(cross_kernel).all():
            raise ValueError("Finite aligned cross-kernel required")
        result = cross_kernel@self.dual_+self.intercept_
        if not np.isfinite(result).all():
            raise ValueError("Nonfinite prediction")
        return result
