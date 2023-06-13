import numpy as np
import matplotlib.pyplot as plt

# increase the figure size
plt.rcParams['figure.figsize'] = [10, 10]

# remove the top and right spines from plot in the global plt setting
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False
# change the linewidth of the axes and spines
plt.rcParams['axes.linewidth'] = 2
plt.rcParams['lines.linewidth'] = 4
plt.rcParams['xtick.major.size'] = 10
plt.rcParams['xtick.major.width'] = 2
plt.rcParams['ytick.major.size'] = 10
plt.rcParams['ytick.major.width'] = 2
plt.rcParams['xtick.minor.size'] = 5
plt.rcParams['xtick.minor.width'] = 2
plt.rcParams['ytick.minor.size'] = 5
plt.rcParams['ytick.minor.width'] = 2
# change the fontsize of the ticks label
plt.rcParams['xtick.labelsize'] = 20
plt.rcParams['ytick.labelsize'] = 20
# change the fontsize of the axes label
plt.rcParams['axes.labelsize'] = 20
# change the fontsize of the legend
plt.rcParams['legend.fontsize'] = 20
# change the fontsize of the title
plt.rcParams['axes.titlesize'] = 20
# change the title font size
plt.rcParams['font.size'] = 20
# change the font family to Arial
plt.rcParams['font.family'] = 'Arial'

# simulation setup
dt = 0.0001
T = int(9 / dt)

# neuronal parameters
tau_e, tau_p, tau_s, tau_v = 0.020, 0.010, 0.010, 0.010
alpha_e, alpha_p, alpha_s, alpha_v = 1.0, 1.0, 1.0, 1.0

# network connectivity
Jee = 1.3
Jep = 1.6
Jes = 1.0
Jev = 0

Jpe = 1.0
Jpp = 1.3
Jps = 0.8
Jpv = 0

Jse = 0.8
Jsp = 0
Jss = 0
Jsv = 0.6

Jve = 1.1
Jvp = 0.4
Jvs = 0.4
Jvv = 0

# neuronal parameters
tau_e, tau_p, tau_s, tau_v = 0.020, 0.010, 0.010, 0.010
alpha_e, alpha_p, alpha_s, alpha_v = 1.0, 1.0, 1.0, 1.0


l_alpha = np.arange(0,20.1,1)
c = 0


l_eig_E, l_eig_E_VIP = [], []

for k, alpha in enumerate(l_alpha):
    J_E_VIP = np.zeros((2, 2))

    # depression variables
    x_ep, x_pp, x_vp = 1, 1, 1
    u_s = 1
    tau_x = 0.10

    # facilitation variables
    u_vs = 1
    U, U_max = 1, 3
    tau_u = 0.40

    r_e, r_p, r_s, r_v = 0, 0, 0, 0
    z_e, z_p, z_s, z_v = 0, 0, 0, 0

    l_r_e, l_r_p, l_r_s, l_r_v = [], [], [], []

    for i in range(T):
        if 50000 <= i < 70000:
            g_e, g_p, g_s, g_v = 4 + alpha, 4 + alpha, 3, 4 + c
        else:
            g_e, g_p, g_s, g_v = 4 + alpha, 4 + alpha, 3, 4

        z_e = Jee * r_e - x_ep * Jep * r_p - Jes * r_s - Jev * r_v + g_e
        z_p = Jpe * r_e - x_pp * Jpp * r_p - Jps * r_s - Jpv * r_v + g_p
        z_s = Jse * r_e - Jsp * r_p - Jss * r_s - Jsv * r_v + g_s
        z_v = Jve * r_e - x_vp * Jvp * r_p - u_vs * Jvs * r_s - Jvv * r_v + g_v

        z_e = z_e * (z_e > 0)
        z_p = z_p * (z_p > 0)
        z_s = z_s * (z_s > 0)
        z_v = z_v * (z_v > 0)

        r_e = r_e + (-r_e + np.power(z_e, alpha_e)) / tau_e * dt
        r_p = r_p + (-r_p + np.power(z_p, alpha_p)) / tau_p * dt
        r_s = r_s + (-r_s + np.power(z_s, alpha_s)) / tau_s * dt
        r_v = r_v + (-r_v + np.power(z_v, alpha_v)) / tau_v * dt

        r_e = r_e * (r_e > 0)
        r_p = r_p * (r_p > 0)
        r_s = r_s * (r_s > 0)
        r_v = r_v * (r_v > 0)

        # STD
        x_ep_pre = np.copy(x_ep)
        x_ep = x_ep + ((1 - x_ep) / tau_x - u_s * x_ep * r_p) * dt
        x_ep = np.clip(x_ep, 0, 1)

        x_pp_pre = np.copy(x_pp)
        x_pp = x_pp + ((1 - x_pp) / tau_x - u_s * x_pp * r_p) * dt
        x_pp = np.clip(x_pp, 0, 1)

        x_vp_pre = np.copy(x_vp)
        x_vp = x_vp + ((1 - x_vp) / tau_x - u_s * x_vp * r_p) * dt
        x_vp = np.clip(x_vp, 0, 1)

        # STF
        u_vs_pre = np.copy(u_vs)
        u_vs = u_vs + ((1 - u_vs) / tau_u + U * (U_max - u_vs) * r_s) * dt
        u_vs = np.clip(u_vs, 1, U_max)

        l_r_e.append(r_e)
        l_r_p.append(r_p)
        l_r_s.append(r_s)
        l_r_v.append(r_v)

        # Jacobians
        J_E_VIP[0, 0] = (Jee - 1) / tau_e
        J_E_VIP[0, 1] = 0
        J_E_VIP[1, 0] = Jve / tau_e
        J_E_VIP[1, 1] = (-Jvv - 1) / tau_v


        if i == 69999:
            eig_E_VIP, _ = np.linalg.eig(J_E_VIP)
            l_eig_E_VIP.append(max(np.real(eig_E_VIP)))

            l_eig_E.append((Jee - 1) /tau_e)

    l_r_e = np.asarray(l_r_e)
    l_r_p = np.asarray(l_r_p)
    l_r_s = np.asarray(l_r_s)
    l_r_v = np.asarray(l_r_v)


l_eig_E_VIP = np.asarray(l_eig_E_VIP)
l_eig_E = np.asarray(l_eig_E)

# plot the leading eigenvalues of the E and the E-VIP subnetwork
plt.figure()
plt.plot(l_eig_E)
plt.plot(l_eig_E_VIP)

plt.xticks([0, 5, 10, 15, 20])
plt.yticks([-30, -15, 0, 15, 30])
plt.xlabel(r'$\alpha$')
plt.ylabel('Leading eigenvalue')
plt.xlim([-1, 21])
plt.ylim([-30, 30])

plt.legend(['E-E', 'E-VIP'], loc='upper right')

plt.hlines(y=0, xmin=-1, xmax=21, colors='k', linestyles=[(0, (6, 6, 6, 6))])

plt.savefig('Eigenvalue_E_E_VIP.png')
plt.close()



   