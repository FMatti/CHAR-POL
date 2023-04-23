import numpy as np
import matplotlib.pyplot as plt

from .methods import baseline


def compute_coefficient_errors(coeffs, coeffs_baseline, mode="absolute"):
    if mode == "absolute":
        errors = np.abs(coeffs - coeffs_baseline)
    elif mode == "relative":
        errors = np.abs(1 - coeffs / coeffs_baseline)
    return errors


def plot_coefficient_errors(A, method, mode="absolute", ax=None):
    coeffs_baseline = baseline(A)
    coeffs = method(A)
    errors = compute_coefficient_errors(coeffs, coeffs_baseline, mode)

    if ax is None:
        fig, ax = plt.subplots()

    ax.set_yscale('log')
    ax.plot(errors)
