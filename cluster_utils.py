import numpy as np
from scipy.spatial.distance import cdist

def calculate_cluster(new_point):
    centroids = [[-3.65019294e-01, -2.59134790e-01,  9.53280384e-01,
         9.81233244e-01,  0.00000000e+00,  3.57462020e-03,
         1.42984808e-02,  9.34763181e-01,  0.00000000e+00,
         9.72296693e-01,  3.12779267e-02,  3.72654155e-01,
         0.00000000e+00,  1.24218052e-01, -8.76542185e-01,
         8.08987747e-01],
       [ 6.39846278e-01,  6.80887724e-01, -3.62669421e-01,
         9.74903475e-01,  0.00000000e+00,  0.00000000e+00,
         2.12355212e-02,  5.26061776e-01,  4.63320463e-02,
         7.97297297e-01,  0.00000000e+00,  1.38996139e-01,
         7.72200772e-03,  1.44787645e-02,  1.03971971e+00,
         4.94459154e-01],
       [ 4.42245081e-01, -1.97942173e-01, -5.52818390e-01,
         0.00000000e+00,  0.00000000e+00,  1.11402359e-02,
         9.70511140e-01,  7.06422018e-01,  3.53866317e-02,
         9.09567497e-01,  3.93184797e-03,  1.63171691e-01,
         5.89777195e-03,  9.17431193e-03,  4.45422941e-01,
        -8.16762602e-01],
       [ 1.09173879e+00,  7.08029091e+00, -7.66967038e-01,
         8.97435897e-01,  0.00000000e+00,  0.00000000e+00,
         1.02564103e-01,  2.05128205e-01,  0.00000000e+00,
         0.00000000e+00,  0.00000000e+00,  5.12820513e-02,
         0.00000000e+00,  0.00000000e+00,  5.11356248e+00,
         3.82576248e+00],
       [-1.56752897e+00, -6.28224907e-01,  2.94382820e-01,
         1.61290323e-02,  1.61290323e-03,  3.22580645e-03,
         9.67741935e-01,  9.14516129e-01,  0.00000000e+00,
         9.51612903e-01,  6.45161290e-02,  4.74193548e-01,
         0.00000000e+00,  2.19354839e-01, -1.57329558e+00,
        -5.16680610e-01]]
    distances = cdist([new_point], centroids)
    assigned_cluster = np.argmin(distances)   
    return assigned_cluster
 