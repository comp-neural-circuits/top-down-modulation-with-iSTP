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


l_alpha = np.arange(0, 20.1, 1)
l_c = [0.1]

l_det_E_PV_VIP = np.zeros((len(l_c), len(l_alpha)))

for n, c in enumerate(l_c):
    l_R_SV = []

    for k, alpha in enumerate(l_alpha):
        J_E_PV_VIP = np.zeros((6, 6))
    
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
        l_R_SV_temp = []

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

            # k-value
            p_pp = 1 / (1+u_s*tau_x *r_p)
            p_pp_prime = - (u_s * tau_x) / np.power(1 + u_s * tau_x * r_p, 2)
            p_ep = 1 / (1+u_s*tau_x *r_p)
            p_ep_prime = - (u_s * tau_x) / np.power(1 + u_s * tau_x * r_p, 2)
            p_vp = 1 / (1+u_s*tau_x *r_p)
            p_vp_prime = - (u_s * tau_x) / np.power(1 + u_s * tau_x * r_p, 2)
            p_vs = (1 + U * U_max * tau_u * r_s) / (1 + U * tau_u * r_s)
            p_vs_prime = (U * (U_max - 1) * tau_u) / np.power(1 + U * tau_u * r_s, 2)

            K_SV = (p_pp + p_pp_prime * r_p) * Jee * Jpp * Jsv - (p_ep + p_ep_prime * r_p) * Jsv * Jpe * Jep - (p_pp + p_pp_prime * r_p) * Jpp * Jsv + Jee * Jsv - Jsv

            # Jacobian of the E-PV-VIP subnetwork
            J_E_PV_VIP[0, 0] = (Jee - 1) / tau_e
            J_E_PV_VIP[0, 1] = - x_ep * Jep / tau_e
            J_E_PV_VIP[0, 2] = -Jev / tau_e
            J_E_PV_VIP[0, 3] = -Jep * r_p / tau_e
            J_E_PV_VIP[0, 4] = 0
            J_E_PV_VIP[0, 5] = 0
            J_E_PV_VIP[1, 0] = Jpe/ tau_p
            J_E_PV_VIP[1, 1] = (-1 - x_pp * Jpp)  / tau_p
            J_E_PV_VIP[1, 2] = -Jpv/ tau_p
            J_E_PV_VIP[1, 3] = 0
            J_E_PV_VIP[1, 4] = -Jpp * r_p/ tau_p
            J_E_PV_VIP[1, 5] = 0
            J_E_PV_VIP[2, 0] = Jve / tau_v
            J_E_PV_VIP[2, 1] = -x_vp * Jvp / tau_v
            J_E_PV_VIP[2, 2] = (-1 - Jvv)  / tau_v
            J_E_PV_VIP[2, 3] = 0
            J_E_PV_VIP[2, 4] = 0
            J_E_PV_VIP[2, 5] = -Jvp * r_p / tau_v
            J_E_PV_VIP[3, 0] = 0
            J_E_PV_VIP[3, 1] = -u_s * x_ep
            J_E_PV_VIP[3, 2] = 0
            J_E_PV_VIP[3, 3] = -1/tau_x - u_s * r_p
            J_E_PV_VIP[3, 4] = 0
            J_E_PV_VIP[3, 5] = 0
            J_E_PV_VIP[4, 0] = 0
            J_E_PV_VIP[4, 1] = -u_s * x_pp
            J_E_PV_VIP[4, 2] = 0
            J_E_PV_VIP[4, 3] = 0
            J_E_PV_VIP[4, 4] = -1/tau_x - u_s * r_p
            J_E_PV_VIP[4, 5] = 0
            J_E_PV_VIP[5, 0] = 0
            J_E_PV_VIP[5, 1] = -u_s * x_vp
            J_E_PV_VIP[5, 2] = 0
            J_E_PV_VIP[5, 3] = 0
            J_E_PV_VIP[5, 4] = 0
            J_E_PV_VIP[5, 5] = -1/tau_x - u_s * r_p

            if 40000 < i < 45000:
                mat_R = np.array(
                    [[1-Jee, (p_ep + p_ep_prime * r_p) * Jep, Jes, Jev],
                    [-Jpe, 1 + (p_pp + p_pp_prime * r_p) * Jpp, Jps, Jpv],
                    [-Jse, Jsp, 1+Jss, Jsv],
                    [-Jve, (p_vp + p_vp_prime * r_p) * Jvp, (p_vs + p_vs_prime * r_s) * Jvs, 1 + Jvv]
                    ]
                )

                det_mat_R = np.linalg.det(mat_R)
                
                R_SV = 1/det_mat_R * K_SV * c

                l_R_SV_temp.append(R_SV)

            if i == 69999:
                l_det_E_PV_VIP[n,k] = np.linalg.det(J_E_PV_VIP)

        l_r_e = np.asarray(l_r_e)
        l_r_p = np.asarray(l_r_p)
        l_r_s = np.asarray(l_r_s)
        l_r_v = np.asarray(l_r_v)

        l_R_SV_temp = np.array(l_R_SV_temp)

        l_R_SV.append(np.mean(l_R_SV_temp))

    # plotting
    fig, ax = plt.subplots()

    ax.hlines(y=0, xmin=-1, xmax=21, colors='k', linestyles=[(0, (6, 6, 6, 6))])

    p1, = ax.plot(l_det_E_PV_VIP[0])

    ax.set_yticks([-10e9, -5e9, 0, 5e9, 10e9])
    ax.set_yticklabels([-10, -5, 0, 5, 10])
    ax.set_ylim([-10e9, 10e9])
    ax.set_ylabel(r'Magnitude of det($\mathbf{M}_{\text{E-PV-VIP}}$)')

    ax2 = ax.twinx()
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(True)
    ax2.spines['bottom'].set_visible(True)
    ax2.spines['left'].set_visible(True)

    p2, = ax2.plot(l_R_SV)

    ax2.set_xticks([0, 5, 10, 15, 20])
    ax2.set_xticklabels([0, 5, 10, 15, 20])
    ax2.set_xlim([-1, 21])

    ax2.set_yticks(np.array([-0.2, -0.1, 0, 0.1, 0.2]))
    ax2.set_yticklabels(np.array([-0.2, -0.1, 0, 0.1, 0.2]))
    ax2.set_ylim([-0.2, 0.2])

    ax2.set_xlabel(r'$\alpha$')
    ax2.set_ylabel('Change in SST activity (Hz)')

    plt.legend([p1, p2], [r'det($\mathbf{M}_{E-PV-VIP}$)', ' analytics ($R_{SV}$)'], loc='top left')

    plt.savefig('Comparison_RSV_det_E_PV_VIP.png')
    plt.close()

