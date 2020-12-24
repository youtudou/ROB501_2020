import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from dcm_from_rpy import dcm_from_rpy
from triangulate import triangulate

# Camera intrinsic matrices.
Kl = np.array([[540.0, 0.0, 320], [0.0, 520.0, 240.0], [0, 0, 1]])
Kr = Kl

# Camera poses (left, right).
Twl = np.eye(4)
Twl[:3, :3] = dcm_from_rpy([-np.pi/2, 0.00, 0.00])
Twr = np.eye(4)
Twr[:3, :3] = dcm_from_rpy([-np.pi/2, 0.20, -0.20])
Twr[0, 3] = 1.0  # Baseline.

# Image plane points (left, right).
pl = np.array([[360], [237.0]])
pr = np.array([[240], [238.5]])

# Image plane uncertainties (covariances).
Sl = 2*np.eye(2)
Sr = 1*np.eye(2)

[Pl, Pr, P, S] = triangulate(Kl, Kr, Twl, Twr, pl, pr, Sl, Sr)

print(Pl)
print(Pr)
print(S)

# The ray Jacobians are:
#
# drayl = [[ 1.83668408e-03  8.15069356e-07  0.00000000e+00  0.00000000e+00]
#          [-1.36046144e-04  1.10034363e-05  0.00000000e+00  0.00000000e+00]
#          [-7.84881602e-07 -1.91772738e-03  0.00000000e+00  0.00000000e+00]]
#
# drayr = [[ 0.00000000e+00  0.00000000e+00  1.77466664e-03 -3.70090854e-04]
#          [ 0.00000000e+00  0.00000000e+00 -8.87869286e-05  8.04998687e-05]
#          [ 0.00000000e+00  0.00000000e+00 -3.55365882e-04 -1.86421391e-03]]
#
# Answers should be:
#
# [[ 1.30856538]
#  [17.66563268]
#  [ 0.10191711]]
#
# [[ 1.99392088]
#  [17.61220432]
#  [ 0.56320742]]
# 
# [[4.88182003e-02 7.24918258e-01 1.57003457e-02]
#  [7.24918258e-01 1.09396614e+01 2.37317656e-01]
#  [1.57003457e-02 2.37317656e-01 5.91946435e-03]]

# Visualize - plot rays and the estimate of P...
fig = plt.figure()
ax = fig.add_subplot(111, projection = '3d')
ax.plot(np.array([Twl[0, 3], Pl[0, 0]]), 
        np.array([Twl[1, 3], Pl[1, 0]]),
        np.array([Twl[2, 3], Pl[2, 0]]), 'b-')
ax.plot(np.array([Twr[0, 3], Pr[0, 0]]),
        np.array([Twr[1, 3], Pr[1, 0]]),
        np.array([Twr[2, 3], Pr[2, 0]]), 'r-')
ax.plot(np.array([Pl[0, 0], Pr[0, 0]]),
        np.array([Pl[1, 0], Pr[1, 0]]),
        np.array([Pl[2, 0], Pr[2, 0]]), 'g-')
ax.plot([P[0, 0]], [P[1, 0]], [P[2, 0]], 'bx', markersize = 8)
plt.show()