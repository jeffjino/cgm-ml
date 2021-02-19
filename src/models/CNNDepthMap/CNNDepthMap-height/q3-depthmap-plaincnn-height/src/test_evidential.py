import evidential_deep_learning as edl
import tensorflow as tf

# aaa = edl.losses.EvidentialRegression(90.7, [2., 3., 4., 5.])
# print(aaa)


# y = tf.constant([2.0, 3.0, 4.0])
# y_ = tf.constant([2.0, 3.0, 4.0])
# print(edl.losses.MSE(y, y_))

# Gaussian_NLL  # not used


y = 92.0
gamma = 1.
v = 1e-9  # 0. -> float division by zero
alpha = 1.
beta = 0. # 0 -> inf
aaa = edl.losses.NIG_NLL(y, gamma, v, alpha, beta)
print(aaa)