import multimodality as mm
import numpy as np

def _generate_example_data_elongated():
    # Example data: elongated Gaussian cloud
    np.random.seed(0)
    n = 1500
    mean = np.array([0.0, 0.0])
    cov = np.array([[4.0, 1.8],
                    [1.8, 1.0]])
    data = np.random.multivariate_normal(mean, cov, size=n)
    return data[:, 0], data[:, 1]


def _generate_example_data_clustered():
    # Example data: two clusters
    np.random.seed(1)
    n1, n2 = 700, 700
    d1 = np.random.multivariate_normal([-1.2, 0.0], [[0.25, 0.0], [0.0, 0.2]], size=n1)
    d2 = np.random.multivariate_normal([1.2, 0.1], [[0.25, 0.0], [0.0, 0.2]], size=n2)
    data2 = np.vstack([d1, d2])
    x2 = data2[:, 0]
    y2 = data2[:, 1]
    return x2, y2

if __name__ == "__main__":    
    x, y = _generate_example_data_elongated()
    results = mm.multimodality_analysis(x, y)
    
    # mm.print_summary(results, elongation_threshold=3)
    # mm.plot_results(results, frac_to_show=0.7)
    ua = mm.unimodality_analysis(results)
    print(ua["unimodality_index"])