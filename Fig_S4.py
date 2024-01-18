import numpy as np
import matplotlib.pyplot as plt

# increase the figure size
plt.rcParams['figure.figsize'] = [12, 10]

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


l_alpha = np.arange(0,20.1,.5)
l_c = [0]


for n, c in enumerate(l_c):
    l_R_SV = []
    l_k_SV_PPD, l_k_SV_PED = [], []
    l_D, l_x_pp, l_x_ep = [], [], []

    for k, alpha in enumerate(l_alpha):
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

            x_ep_ss = 1 / (1 + u_s * tau_x * r_p)
            x_pp_ss = 1 / (1 + u_s * tau_x * r_p)
            x_vp_ss = 1 / (1 + u_s * tau_x * r_p)

            u_vs_ss = (1 + U * U_max * tau_u * r_s) / (1 + U * tau_u * r_s)

            k_SV_PPD = (p_pp + p_pp_prime * r_p - 1) * (Jee - 1) * Jpp * Jsv
            k_SV_PED = -(p_ep + p_ep_prime * r_p - 1) * Jsv * Jpe * Jep

            if i == 49999:

                mat_R = np.array(
                    [[1-Jee, (p_ep + p_ep_prime * r_p) * Jep, Jes, Jev],
                    [-Jpe, 1 + (p_pp + p_pp_prime * r_p) * Jpp, Jps, Jpv],
                    [-Jse, Jsp, 1+Jss, Jsv],
                    [-Jve, (p_vp + p_vp_prime * r_p) * Jvp, (p_vs + p_vs_prime * r_s) * Jvs, 1 + Jvv]
                    ]
                )

                det_mat_R = np.linalg.det(mat_R)

                l_D.append(1/det_mat_R)
                l_x_pp.append(p_pp + p_pp_prime * r_p - 1)
                l_x_ep.append(p_ep + p_ep_prime * r_p - 1)

                l_k_SV_PPD.append(k_SV_PPD)
                l_k_SV_PED.append(k_SV_PED)

        l_r_e = np.asarray(l_r_e)
        l_r_p = np.asarray(l_r_p)
        l_r_s = np.asarray(l_r_s)
        l_r_v = np.asarray(l_r_v)

    l_D = np.asarray(l_D)
    l_x_pp = np.asarray(l_x_pp)
    l_x_ep = np.asarray(l_x_ep)

    l_k_SV_PPD = np.asarray(l_k_SV_PPD)
    l_k_SV_PED = np.asarray(l_k_SV_PED)


    # plot magnitude of D (Fig. S4A)
    plt.figure()

    plt.plot(l_D)

    plt.yticks([0, 5, 10, 15])
    plt.ylim([0, 15])

    plt.xticks([0, 10, 20, 30, 40], [0, 5, 10, 15, 20])
    plt.xlim([-2, 42])

    plt.xlabel(r'$\alpha$')
    plt.ylabel('Magnitude of D')

    plt.savefig('Fig_S5A.png')
    plt.close()


    # plot PV-to-E STD and PV-to-PV STD dependent term (Fig. S5 B)
    plt.figure()

    plt.plot(l_x_pp, color='blue')
    plt.plot(l_x_ep, '--', color='orange')

    plt.yticks([-1, -0.5, 0, 0.5, 1])
    plt.ylim([-1, 1])

    plt.xticks([0, 10, 20, 30, 40], [0, 5, 10, 15, 20])
    plt.xlim([-2, 42])

    plt.axhline(y=0, color='k', linestyle='--')

    plt.xlabel(r'$\alpha$')
    plt.ylabel('Magnitude (a.u.)')

    plt.legend(["$x_{PP}^* + x_{PP}^{*'}r_P - 1$", "$x_{EP}^* + x_{EP}^{*'}r_P - 1$"], loc='upper left')

    plt.savefig('Fig_S4B.png')
    plt.close()

    # plot K_SV_PPD and K_SV_PED (Fig. S5 C)
    plt.figure()

    plt.plot(l_k_SV_PPD, color='green')
    plt.plot(l_k_SV_PED, color='orange')

    plt.yticks([-1, -0.5, 0, 0.5, 1])
    plt.ylim([-1.0, 1.0])

    plt.xticks([0, 10, 20, 30, 40], [0, 5, 10, 15, 20])
    plt.xlim([-2, 42])

    plt.axhline(y=0, color='k', linestyle='--')

    plt.xlabel(r'$\alpha$')
    plt.ylabel('Magnitude (a.u.)')

    plt.legend([r'$\mathbf{K}_{SV}^{\text{PPD}}$', r'$\mathbf{K}_{SV}^{\text{PED}}$'], loc='lower left')

    plt.savefig('Fig_S4C.png')
    plt.close()



