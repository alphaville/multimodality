import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from scipy import ndimage


def estimate_kde_on_grid(x, y, gridsize=200, bw_method=None, padding=0.15):
    """
    Estimate 2D KDE on a rectangular grid.
    Returns X, Y, Z where Z is the estimated density on the grid.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    values = np.vstack([x, y])

    kde = gaussian_kde(values, bw_method=bw_method)

    xmin, xmax = x.min(), x.max()
    ymin, ymax = y.min(), y.max()

    dx = xmax - xmin
    dy = ymax - ymin

    xmin -= padding * dx
    xmax += padding * dx
    ymin -= padding * dy
    ymax += padding * dy

    xx = np.linspace(xmin, xmax, gridsize)
    yy = np.linspace(ymin, ymax, gridsize)
    X, Y = np.meshgrid(xx, yy)

    positions = np.vstack([X.ravel(), Y.ravel()])
    Z = kde(positions).reshape(X.shape)

    return X, Y, Z


def component_elongation(coords):
    """
    Compute elongation of a connected component from its grid coordinates.

    coords: array of shape (n_points, 2), columns are x and y coordinates.

    Returns:
        elongation_ratio: sqrt(lambda_max / lambda_min)
        eigenvalues: sorted descending
    """
    if len(coords) < 3:
        return np.nan, (np.nan, np.nan)

    centered = coords - coords.mean(axis=0, keepdims=True)
    cov = np.cov(centered.T)

    evals, _ = np.linalg.eigh(cov)
    evals = np.sort(evals)[::-1]

    eps = 1e-12
    elongation_ratio = np.sqrt((evals[0] + eps) / (evals[1] + eps))
    return elongation_ratio, evals


def analyze_superlevel_sets(
    X, Y, Z,
    frac_min=0.7,
    n_levels=15,
    connectivity=2,
    elongation_threshold=3.0,
    min_component_size=20
):
    """
    Analyze superlevel sets {Z >= t} for t in [frac_min * Zmax, Zmax].

    Parameters
    ----------
    X, Y, Z : 2D arrays
        Grid and density values.
    frac_min : float
        Minimum threshold as fraction of max density.
    n_levels : int
        Number of thresholds between frac_min * max(Z) and max(Z).
    connectivity : int
        1 for 4-connectivity, 2 for 8-connectivity in 2D.
    elongation_threshold : float
        Flag components with elongation ratio above this value.
    min_component_size : int
        Ignore very tiny connected components.

    Returns
    -------
    results : list of dicts
        One dict per threshold.
    """
    zmax = Z.max()
    levels = np.linspace(frac_min * zmax, zmax, n_levels)

    if connectivity == 1:
        structure = ndimage.generate_binary_structure(2, 1)
    else:
        structure = ndimage.generate_binary_structure(2, 2)

    results = []

    for t in levels:
        mask = Z >= t
        labeled, ncomp = ndimage.label(mask, structure=structure)

        components = []
        valid_count = 0

        for k in range(1, ncomp + 1):
            idx = np.where(labeled == k)
            size = len(idx[0])

            if size < min_component_size:
                continue

            valid_count += 1
            coords = np.column_stack([X[idx], Y[idx]])
            elongation, evals = component_elongation(coords)

            components.append({
                "label": k,
                "size": size,
                "elongation": elongation,
                "eigenvalues": evals
            })

        disconnected = valid_count > 1
        elongated = any(
            np.isfinite(c["elongation"]) and c["elongation"] > elongation_threshold
            for c in components
        )

        results.append({
            "level": t,
            "level_fraction_of_max": t / zmax,
            "n_components": valid_count,
            "disconnected": disconnected,
            "elongated": elongated,
            "components": components
        })

    return results


def print_summary(results, elongation_threshold=3.0):
    """
    Print a readable summary of the superlevel-set analysis.
    """
    any_disconnected = any(r["disconnected"] for r in results)
    any_elongated = any(r["elongated"] for r in results)

    print("Any disconnected superlevel set above threshold range?:", any_disconnected)
    print(f"Any elongated component (ratio > {elongation_threshold})?:", any_elongated)
    print()

    for r in results:
        print(
            f"Level = {r['level_fraction_of_max']:.3f} * max(pdf), "
            f"components = {r['n_components']}, "
            f"disconnected = {r['disconnected']}, "
            f"elongated = {r['elongated']}"
        )
        for c in r["components"]:
            print(
                f"   component {c['label']}: size={c['size']}, "
                f"elongation={c['elongation']:.3f}"
            )


def plot_results(X, Y, Z, results, frac_to_show=0.7):
    """
    Plot KDE contours and the superlevel set at one selected threshold.
    """
    zmax = Z.max()
    target_level = frac_to_show * zmax

    # find closest analyzed level
    best = min(results, key=lambda r: abs(r["level"] - target_level))
    level = best["level"]

    mask = Z >= level

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # KDE contours
    cs = axes[0].contour(X, Y, Z, levels=12)
    axes[0].contour(X, Y, Z, levels=[level], linewidths=2)
    axes[0].set_title(f"KDE contours\nhighlighted level = {best['level_fraction_of_max']:.2f} * max")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")

    # Binary superlevel set
    axes[1].imshow(
        mask.astype(int),
        origin="lower",
        extent=[X.min(), X.max(), Y.min(), Y.max()],
        aspect="auto"
    )
    axes[1].set_title(f"Superlevel set: pdf >= {best['level_fraction_of_max']:.2f} * max")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("y")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Example data: two clusters
    np.random.seed(1)
    n1, n2 = 700, 700
    d1 = np.random.multivariate_normal([-1.2, 0.0], [[0.25, 0.0], [0.0, 0.2]], size=n1)
    d2 = np.random.multivariate_normal([1.2, 0.1], [[0.25, 0.0], [0.0, 0.2]], size=n2)
    data2 = np.vstack([d1, d2])

    x2 = data2[:, 0]
    y2 = data2[:, 1]

    X2, Y2, Z2 = estimate_kde_on_grid(x2, y2, gridsize=220, bw_method='scott')

    results2 = analyze_superlevel_sets(
        X2, Y2, Z2,
        frac_min=0.7,
        n_levels=12,
        connectivity=2,
        elongation_threshold=2.5,
        min_component_size=15
    )

    print_summary(results2, elongation_threshold=2.5)
    plot_results(X2, Y2, Z2, results2, frac_to_show=0.7)