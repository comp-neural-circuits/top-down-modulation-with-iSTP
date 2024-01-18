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
Jep = 1.5
Jes = 0.9
Jev = 0

Jpe = 1.1
Jpp = 1.3
Jps = 0.8
Jpv = 0

Jse = 0.5
Jsp = 0
Jss = 0
Jsv = 0.6

Jve = 1.1
Jvp = 0.3
Jvs = 0.2
Jvv = 0

g_1 = 4
g_2 = 3

l_alpha = np.arange(0, 20.1, 0.5)
c = 3

l_R_SV = []
l_SST_change = []

for alpha in l_alpha:
    # depression variables
    x_ep, x_pp, x_vp = 1, 1, 1
    u_s = 1
    tau_x = 0.10

    # facilitation variables
    u_vs, u_se = 1, 1
    U, U_max = 1, 3
    U_max_se = 2
    tau_u = 0.40

    r_e, r_p, r_s, r_v = 0, 0, 0, 0
    z_e, z_p, z_s, z_v = 0, 0, 0, 0

    l_r_e, l_r_p, l_r_s, l_r_v = [], [], [], []
    l_R_SV_temp = []
    
    for i in range(T):
        if 50000 <= i < 70000:
            g_e, g_p, g_s, g_v = g_1 + alpha, g_1 + alpha, g_2, g_1 + c
        else:
            g_e, g_p, g_s, g_v = g_1 + alpha, g_1 + alpha, g_2, g_1

        z_e = Jee * r_e - x_ep * Jep * r_p - Jes * r_s - Jev * r_v + g_e
        z_p = Jpe * r_e - x_pp * Jpp * r_p - Jps * r_s - Jpv * r_v + g_p
        z_s = u_se * Jse * r_e - Jsp * r_p - Jss * r_s - Jsv * r_v + g_s
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
        x_ep = x_ep + ((1 - x_ep) / tau_x - u_s * x_ep * r_p) * dt
        x_ep = np.clip(x_ep, 0, 1)

        x_pp = x_pp + ((1 - x_pp) / tau_x - u_s * x_pp * r_p) * dt
        x_pp = np.clip(x_pp, 0, 1)

        x_vp = x_vp + ((1 - x_vp) / tau_x - u_s * x_vp * r_p) * dt
        x_vp = np.clip(x_vp, 0, 1)

        # STF
        u_vs = u_vs + ((1 - u_vs) / tau_u + U * (U_max - u_vs) * r_s) * dt
        u_vs = np.clip(u_vs, 1, U_max)

        u_se = u_se + ((1 - u_se) / tau_u + U * (U_max_se - u_se) * r_e) * dt
        u_se = np.clip(u_se, 1, U_max_se)


        if  i == 49999:
            # k-value
            p_pp = 1 / (1+u_s*tau_x *r_p)
            p_pp_prime = - (u_s * tau_x) / np.power(1 + u_s * tau_x * r_p, 2)

            p_ep = 1 / (1+u_s*tau_x *r_p)
            p_ep_prime = - (u_s * tau_x) / np.power(1 + u_s * tau_x * r_p, 2)

            p_vp = 1 / (1+u_s*tau_x *r_p)
            p_vp_prime = - (u_s * tau_x) / np.power(1 + u_s * tau_x * r_p, 2)

            p_vs = (1 + U * U_max * tau_u * r_s) / (1 + U * tau_u * r_s)
            p_vs_prime = (U * (U_max - 1) * tau_u) / np.power(1 + U * tau_u * r_s, 2)

            p_se = (1 + U * U_max_se * tau_u * r_e) / (1 + U * tau_u * r_e)
            p_se_prime = (U * (U_max_se - 1) * tau_u) / np.power(1 + U * tau_u * r_e, 2)

            k_SV = (p_pp + p_pp_prime * r_p) * Jee * Jpp * Jsv - (p_ep + p_ep_prime * r_p) * Jsv * Jpe * Jep - (p_pp + p_pp_prime * r_p) * Jpp * Jsv + Jee * Jsv - Jsv

            mat_R = np.array(
                [[1-Jee, (p_ep + p_ep_prime * r_p) * Jep, Jes, Jev],
                [-Jpe, 1 + (p_pp + p_pp_prime * r_p) * Jpp, Jps, Jpv],
                [-(p_se + p_se_prime * r_e) * Jse, Jsp, 1+Jss, Jsv],
                [-Jve, (p_vp + p_vp_prime * r_p) * Jvp, (p_vs + p_vs_prime * r_s) * Jvs, 1 + Jvv]
                ]
            )

            det_mat_R = np.linalg.det(mat_R)

            R_SV = 1/det_mat_R * k_SV * c

            l_R_SV.append(R_SV)
            
        l_r_e.append(r_e)
        l_r_p.append(r_p)
        l_r_s.append(r_s)
        l_r_v.append(r_v)
        
    l_r_e = np.asarray(l_r_e)
    l_r_p = np.asarray(l_r_p)
    l_r_s = np.asarray(l_r_s)
    l_r_v = np.asarray(l_r_v)

    l_SST_change.append(np.mean(l_r_s[60000:65000]) - np.mean(l_r_s[40000:45000]))

    if (alpha == 0 or alpha == 15):
        plt.figure()

        plt.plot(l_r_e)
        plt.plot(l_r_p)
        plt.plot(l_r_s)
        plt.plot(l_r_v)

        plt.xticks(np.arange(30000, 90000 + 5000, 20000), np.arange(0, 7, 2))

        if alpha == 0:
            plt.yticks([0, 5, 10, 15])
        else:
            plt.yticks([0, 20, 40, 60])

        plt.xlabel('Time (s)')
        plt.ylabel('Firing rate (a.u.)')
        plt.title('top down modulation')
        plt.axhline(y=np.mean(l_r_s[35000:45000]), color='k', linestyle='--')
        plt.xlim([30000, 90000])

        if alpha == 0:
            plt.hlines(y=14.9, xmin=50000, xmax=70000, color='gray')
            plt.ylim([0, 15])
        else:
            plt.hlines(y=59.9, xmin=50000, xmax=70000, color='gray')
            plt.ylim([0, 60])

        plt.legend(['E', 'PV', 'SST', 'VIP'], loc='upper right')

        if alpha == 0:
            plt.savefig('Fig_S9B.png')
        else:
            plt.savefig('Fig_S9C.png')

# plotting
plt.figure()

plt.plot(l_SST_change)
plt.plot(l_R_SV, '--')

plt.axhline(y=0, color='k', linestyle='--')

plt.xticks([0, 10, 20, 30, 40], [0, 5, 10, 15, 20])
plt.xlim([-2, 42])

plt.yticks([-2, 0, 2, 4])
plt.ylim([-2, 4])

plt.xlabel(r'$\alpha$')
plt.ylabel('Change in SST activity (a.u.)')

plt.title(r'Large perturbation (large $\delta g_V$)')

plt.legend(['simulation', 'analytics $\mathbf{R}_{SV}$'], loc='upper left')

plt.savefig('Fig_S9A.png')