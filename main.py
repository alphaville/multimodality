import multimodality as mm
import numpy as np

if __name__ == "__main__":
    # Example data: two clusters
    np.random.seed(1)
    n1, n2 = 700, 700
    d1 = np.random.multivariate_normal([-1.2, 0.0], [[0.25, 0.0], [0.0, 0.2]], size=n1)
    d2 = np.random.multivariate_normal([1.2, 0.1], [[0.25, 0.0], [0.0, 0.2]], size=n2)
    data2 = np.vstack([d1, d2])

    x2 = data2[:, 0]
    y2 = data2[:, 1]

    X2, Y2, Z2 = mm.estimate_kde_on_grid(x2, y2, gridsize=220, bw_method='scott')

    results2 = mm.analyze_superlevel_sets(
        X2, Y2, Z2,
        frac_min=0.7,
        n_levels=20,
        connectivity=2,
        elongation_threshold=2.5,
        min_component_size=15
    )

    mm.print_summary(results2, elongation_threshold=2.5)
    mm.plot_results(X2, Y2, Z2, results2, frac_to_show=0.7)