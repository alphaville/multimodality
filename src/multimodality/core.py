import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from scipy import ndimage


def _estimate_kde_on_grid(x, y, gridsize=200, bw_method=None, padding=0.15):
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


def _component_elongation(coords):
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


def _analyze_superlevel_sets(
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
            elongation, evals = _component_elongation(coords)

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


def multimodality_analysis(
        x_data, 
        y_data,
        frac_min=0.7,
        n_levels=15,
        connectivity=2,
        elongation_threshold=3.0,
        min_component_size=20,
        gridsize=220, 
        bw_method='scott', 
        padding=0.15):
    X, Y, Z = _estimate_kde_on_grid(x_data, y_data, gridsize=gridsize, bw_method=bw_method, padding=padding)
    results = _analyze_superlevel_sets(
        X, Y, Z,
        frac_min=frac_min,
        n_levels=n_levels,
        connectivity=connectivity,
        elongation_threshold=elongation_threshold,
        min_component_size=elongation_threshold,
    )
    return {
        "analysis": results,
        "x_": X, "y_": Y, "z_": Z
    }
    

def print_summary(results, elongation_threshold=3.0):
    """
    Print a readable summary of the superlevel-set analysis.
    """
    analysis = results["analysis"]
    any_disconnected = any(r["disconnected"] for r in analysis)
    any_elongated = any(r["elongated"] for r in analysis)

    print("Any disconnected superlevel set above threshold range?:", any_disconnected)
    print(f"Any elongated component (ratio > {elongation_threshold})?:", any_elongated)
    print()

    for r in analysis:
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


def plot_results(results, frac_to_show=0.7):
    """
    Plot KDE contours and the superlevel set at one selected threshold.
    """
    X = results["x_"]
    Y = results["y_"] 
    Z = results["z_"] 
    analysis = results["analysis"]
    zmax = Z.max()
    target_level = frac_to_show * zmax

    # find closest analyzed level
    best = min(analysis, key=lambda r: abs(r["level"] - target_level))
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

def unimodality_analysis(
    results,
    w_disc=0.9,
    w_elong=0.1,
    e_good=1.5,
    e_bad=3.0,
    agg="min",
):
    """
    Compute a unimodality index in [0, 1] from `results`.

    Parameters
    ----------
    results : list of dict
        Output of analyze_superlevel_sets(...)
    w_disc : float
        Weight for disconnectedness penalty.
    w_elong : float
        Weight for elongation penalty.
    e_good : float
        Elongation at or below this is considered fine.
    e_bad : float
        Elongation at or above this gets full elongation penalty.
    agg : str
        How to aggregate across levels: "mean", "min", or "weighted_mean".

    Returns
    -------
    summary : dict with:
        - unimodality_index
        - per_level_scores
        - per_level_penalties
    """
    analysis = results["analysis"]
    if not np.isclose(w_disc + w_elong, 1.0):
        raise ValueError("w_disc + w_elong must equal 1.")

    per_level_scores = []
    per_level_penalties = []

    level_fracs = np.array([r["level_fraction_of_max"] for r in analysis], dtype=float)

    for r in analysis:
        c = r["n_components"]

        # disconnectedness penalty
        if c <= 1:
            p_disc = 0.0
        else:
            p_disc = 1.0 - 1.0 / c

        # max elongation over components at this level
        if len(r["components"]) == 0:
            e_max = 1.0
        else:
            elongs = [comp["elongation"] for comp in r["components"] if np.isfinite(comp["elongation"])]
            e_max = max(elongs) if len(elongs) > 0 else 1.0

        # elongation penalty
        if e_max <= e_good:
            p_elong = 0.0
        elif e_max >= e_bad:
            p_elong = 1.0
        else:
            p_elong = (e_max - e_good) / (e_bad - e_good)

        total_penalty = w_disc * p_disc + w_elong * p_elong
        score = 1.0 - total_penalty

        per_level_scores.append(score)
        per_level_penalties.append({
            "level_fraction_of_max": r["level_fraction_of_max"],
            "n_components": c,
            "max_elongation": e_max,
            "p_disc": p_disc,
            "p_elong": p_elong,
            "score": score,
        })

    per_level_scores = np.array(per_level_scores)

    if agg == "mean":
        U = float(np.mean(per_level_scores))
    elif agg == "min":
        U = float(np.min(per_level_scores))
    elif agg == "weighted_mean":
        # weight more heavily the levels closer to the maximum
        weights = level_fracs / level_fracs.sum()
        U = float(np.sum(weights * per_level_scores))
    else:
        raise ValueError("agg must be one of: 'mean', 'min', 'weighted_mean'")

    return {
        "unimodality_index": U,
        "per_level_scores": per_level_scores,
        "per_level_penalties": per_level_penalties,
    }