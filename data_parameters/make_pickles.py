import numpy as np

twoD = {
    'transition_probabilities': np.array([
        [9/10, 1/10],
        [3/10, 7/10]
    ]),
    'means': np.array([
        [1/4, 1/2],
        [8/10, 1/10],
    ]),
    'start_probabilities': np.array([4/5, 1/5]),
    'covariances': np.array([np.eye(3) * 0.1 for i in range(2)])
}
    model.means_ = np.array([[3, 4],
                             [5, 7]])
    cov_base = np.array([[[2, 0.5],
                          [0.5, 2]],
                         [[1.5, -0.3],
                          [-0.3, 4]]])
A = {
    'transition_probabilities': np.array([
        [9/10, 1/10],
        [3/10, 7/10]
    ]),
    'means': np.array([
        [1/4, 1/2, 1/4],
        [8/10, 1/10, 1/10)],
    ]),
    'start_probabilities': np.array([4/5, 1/5]),
    'covariances': np.array([np.eye(3) * 0.1 for i in range(2)])
}

B = {
    'transition_probabilities': np.array([
        [9/10, 1/10],
        [3/20, 7/10]
    ]),
    'means': np.array([
        [1/4, 8/10, -3/4, 6/5, 6/7, -1/2],
        [-1, -2/3, 4/3, 5/6, 7/6, 1/2]
    'start_probabilities': np.array([4/5, 1/5]),
    'covariances': np.array([np.eye(6) * 0.1 for i in range(2)])
}

C = {
    'transition_probabilities': np.array([
        [4/10, 1/10, 5/10],
        [1/15, 13/15, 1/15],
        [3/8, 1/8, 1/2]
    ]),
    'means': np.array([
        [1/4, 8/10, -3/4, 6/5, 6/7, -1/2, 8/5, 3/2],
        [-1, -2/3, 4/3, 5/6, 7/6, 1/2, 5/4, 3/2]
    ]),
    'start_probabilities': np.array([1/3, 1/3, 1/3]),
    'covariances': np.array([np.eye(8) for i in range(2)])
}

D = {
    'transition_probabilities': np.array([
        [4/10, 1/10, 5/10],
        [1/15, 13/15, 1/15],
        [3/8, 1/8, 1/2]
    ]),
    'means': np.array([
        [0.32, -0.7,  1.44,  0.49,  0.51, -0.31,  1.16, -1.17,  0.16, 0.06],
        [0.84,  1.2, -0.95,  0.33,  0.59,  0.96,  0.06, -1.24, -0.18, 0.04]
    ])
    'start_probabilities': np.array([1/3, 1/3, 1/3]),
    'covariances': np.array([np.eye(10) for i in range(2)])
}
