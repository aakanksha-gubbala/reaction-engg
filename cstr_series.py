import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline

n_tanks = 2


def get_tau(tau):
    c0 = 0.016
    for i in range(n_tanks):
        c1 = opt.fsolve(lambda c: c - c0 + tau * (0.033 * c ** 2 + 0.006 * c), c0)
        c0 = c1
        # print("%d\t%0.4f\n" % (i+1, c1))
    return c1 - 0.0008


tau = opt.fsolve(get_tau, 0.01)
print(tau)
# print(tau*0.278)

c0 = 0.08
x = np.linspace(0, 0.1, 100)
y = 9.92 * x ** 2 + 0.1984 * x

plt.title(r"$\tau$ = %0.3f, number of CSTRs = %d" % (tau, n_tanks))
plt.plot(x, y)
plt.xlim(0, 0.1)
plt.ylim(0, 0.1)
plt.xlabel(r"$C_B\, (kmol/m^3)$")
plt.ylabel(r"$-r_B\, (kmol/m^3.ks)$")
for i in range(n_tanks):
    plt.plot([c0, c0], [9.92 * c0 ** 2 + 0.1984 * c0, 0], color="green")
    c1 = opt.fsolve(lambda c: c - c0 + tau * (9.92 * c ** 2 + 0.1984 * c), c0)
    plt.plot([c1, c0], [9.92 * c1 ** 2 + 0.1984 * c1, 0], color="green")
    c0 = c1
    print("%d\t%0.4f\n" % (i + 1, c1))

plt.plot([c1, c0], [9.92 * c1 ** 2 + 0.1984 * c1, 0], color="green")

# plt.savefig("%dtanks.png" % (n_tanks), dpi=300)
t = [23.522, 4.989, 2.527, 0.964, 0.519]
n = [1, 2, 3, 6, 10]
n_x = np.linspace(1, 10, 30)
t_y = make_interp_spline(n, t, k=1)(n_x)

plt.plot(n_x, t_y)
plt.xlabel("Number of tanks")
plt.ylabel(r"$\tau,\, (ks)}$")
# plt.savefig("ntanks-tau.png", dpi=300)

plt.show()


