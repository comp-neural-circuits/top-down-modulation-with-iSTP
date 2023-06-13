import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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

pal = sns.color_palette("deep")

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


c = 0.1

l_alpha = np.arange(0, 20.5, 0.5)
m_r_e_SST_freeze, m_r_p_SST_freeze, m_r_s_SST_freeze, m_r_v_SST_freeze = np.zeros((len(l_alpha), T)), np.zeros((len(l_alpha), T)), np.zeros((len(l_alpha), T)), np.zeros((len(l_alpha), T))
m_r_e_PV_freeze, m_r_p_PV_freeze, m_r_s_PV_freeze, m_r_v_PV_freeze = np.zeros((len(l_alpha), T)), np.zeros((len(l_alpha), T)), np.zeros((len(l_alpha), T)), np.zeros((len(l_alpha), T))
l_eig_E_PV_VIP_STP = []
l_eig_E_SST_VIP_STP = []


# standard experimental setup -- VIP perturbation
for k, alpha in enumerate(l_alpha):
    # E-PV no change, PV-SST no connections
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
    J_E_PV_VIP_STP = np.zeros((6, 6))
    J_E_SST_VIP = np.zeros((4, 4))

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
        u_vs = u_vs + ((U - u_vs) / tau_u + U * (U_max - u_vs) * r_s) * dt
        u_vs = np.clip(u_vs, 1, U_max)

        l_r_e.append(r_e)
        l_r_p.append(r_p)
        l_r_s.append(r_s)
        l_r_v.append(r_v)

        # derivatives of plasticity parameters
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

        # Jacobian of the weight matrix
        J_E_PV_VIP_STP[0, 0] = (Jee - 1) / tau_e
        J_E_PV_VIP_STP[0, 1] = - x_ep * Jep / tau_e
        J_E_PV_VIP_STP[0, 2] = -Jev / tau_e
        J_E_PV_VIP_STP[0, 3] = -Jep * r_p / tau_e
        J_E_PV_VIP_STP[0, 4] = 0
        J_E_PV_VIP_STP[0, 5] = 0
        J_E_PV_VIP_STP[1, 0] = Jpe/ tau_p
        J_E_PV_VIP_STP[1, 1] = (-1 - x_pp * Jpp)  / tau_p
        J_E_PV_VIP_STP[1, 2] = -Jpv/ tau_p
        J_E_PV_VIP_STP[1, 3] = 0
        J_E_PV_VIP_STP[1, 4] = -Jpp * r_p/ tau_p
        J_E_PV_VIP_STP[1, 5] = 0
        J_E_PV_VIP_STP[2, 0] = Jve / tau_v
        J_E_PV_VIP_STP[2, 1] = -x_vp * Jvp / tau_v
        J_E_PV_VIP_STP[2, 2] = (-1 - Jvv)  / tau_v
        J_E_PV_VIP_STP[2, 3] = 0
        J_E_PV_VIP_STP[2, 4] = 0
        J_E_PV_VIP_STP[2, 5] = -Jvp * r_p / tau_v
        J_E_PV_VIP_STP[3, 0] = 0
        J_E_PV_VIP_STP[3, 1] = -u_s * x_ep
        J_E_PV_VIP_STP[3, 2] = 0
        J_E_PV_VIP_STP[3, 3] = -1/tau_x - u_s * r_p
        J_E_PV_VIP_STP[3, 4] = 0
        J_E_PV_VIP_STP[3, 5] = 0
        J_E_PV_VIP_STP[4, 0] = 0
        J_E_PV_VIP_STP[4, 1] = -u_s * x_pp
        J_E_PV_VIP_STP[4, 2] = 0
        J_E_PV_VIP_STP[4, 3] = 0
        J_E_PV_VIP_STP[4, 4] = -1/tau_x - u_s * r_p
        J_E_PV_VIP_STP[4, 5] = 0
        J_E_PV_VIP_STP[5, 0] = 0
        J_E_PV_VIP_STP[5, 1] = -u_s * x_vp
        J_E_PV_VIP_STP[5, 2] = 0
        J_E_PV_VIP_STP[5, 3] = 0
        J_E_PV_VIP_STP[5, 4] = 0
        J_E_PV_VIP_STP[5, 5] = -1/tau_x - u_s * r_p

        J_E_SST_VIP[0, 0] = (Jee - 1) / tau_e
        J_E_SST_VIP[0, 1] = -Jes / tau_e
        J_E_SST_VIP[0, 2] = -Jev / tau_e
        J_E_SST_VIP[0, 3] = 0
        J_E_SST_VIP[1, 0] = Jse/ tau_s
        J_E_SST_VIP[1, 1] = -1/tau_s
        J_E_SST_VIP[1, 2] = -Jsv/ tau_s
        J_E_SST_VIP[1, 3] = 0
        J_E_SST_VIP[2, 0] = Jve / tau_v
        J_E_SST_VIP[2, 1] = -u_vs *Jvs / tau_v
        J_E_SST_VIP[2, 2] = -1/tau_v
        J_E_SST_VIP[2, 3] = - Jvs * r_s / tau_v
        J_E_SST_VIP[3, 0] = 0
        J_E_SST_VIP[3, 1] = U * (U_max - u_vs)
        J_E_SST_VIP[3, 2] = 0
        J_E_SST_VIP[3, 3] = -1/tau_u - U * r_s

        if i == 49999:
            l_eig_E_PV_VIP_STP.append(max(np.real(np.linalg.eig(J_E_PV_VIP_STP)[0][0]), np.real(np.linalg.eig(J_E_PV_VIP_STP)[0][1]),  np.real(np.linalg.eig(J_E_PV_VIP_STP)[0][2]), np.real(np.linalg.eig(J_E_PV_VIP_STP)[0][3]), np.real(np.linalg.eig(J_E_PV_VIP_STP)[0][4]), np.real(np.linalg.eig(J_E_PV_VIP_STP)[0][5])))
            l_eig_E_SST_VIP_STP.append(max(np.real(np.linalg.eig(J_E_SST_VIP)[0][0]), np.real(np.linalg.eig(J_E_SST_VIP)[0][1]),  np.real(np.linalg.eig(J_E_SST_VIP)[0][2]), np.real(np.linalg.eig(J_E_SST_VIP)[0][3])))

    l_eig_E_PV_VIP_STP = np.asarray(l_eig_E_PV_VIP_STP)
    l_eig_E_SST_VIP_STP = np.asarray(l_eig_E_SST_VIP_STP)


# Inhibition stabilization 
l_freeze = ['SST', 'PV']
for population in l_freeze:
    for k, alpha in enumerate(l_alpha):
        # E-PV no change, PV-SST no connections
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
            g_e, g_p, g_s, g_v = 4 + alpha, 4 + alpha, 3, 4

            z_e = Jee * r_e - x_ep * Jep * r_p - Jes * r_s - Jev * r_v + g_e
            z_p = Jpe * r_e - x_pp * Jpp * r_p - Jps * r_s - Jpv * r_v + g_p
            z_s = Jse * r_e - Jsp * r_p - Jss * r_s - Jsv * r_v + g_s
            z_v = Jve * r_e - x_vp * Jvp * r_p - u_vs * Jvs * r_s - Jvv * r_v + g_v

            z_e = z_e * (z_e > 0)
            z_p = z_p * (z_p > 0)
            z_s = z_s * (z_s > 0)
            z_v = z_v * (z_v > 0)

            # perturb E population
            if i == 40000:
                r_e = r_e + (-r_e + np.power(z_e, alpha_e)) / tau_e * dt + 0.1
            else:
                r_e = r_e + (-r_e + np.power(z_e, alpha_e)) / tau_e * dt

            # freeze PV population
            if i > 39999 and population == 'PV':
                r_p =  r_p
            else:
                r_p = r_p + (-r_p + np.power(z_p, alpha_p)) / tau_p * dt

            # freeze SST population
            if i > 39999 and population == 'SST':
                r_s = r_s
            else:
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
            u_vs = u_vs + ((U - u_vs) / tau_u + U * (U_max - u_vs) * r_s) * dt
            u_vs = np.clip(u_vs, 1, U_max)

            l_r_e.append(r_e)
            l_r_p.append(r_p)
            l_r_s.append(r_s)
            l_r_v.append(r_v)

        l_r_e = np.asarray(l_r_e)
        l_r_p = np.asarray(l_r_p)
        l_r_s = np.asarray(l_r_s)
        l_r_v = np.asarray(l_r_v)

        if population == 'SST':
            m_r_e_SST_freeze[k] = l_r_e
            m_r_p_SST_freeze[k] = l_r_p
            m_r_s_SST_freeze[k] = l_r_s
            m_r_v_SST_freeze[k] = l_r_v
        elif population == 'PV':
            m_r_e_PV_freeze[k] = l_r_e
            m_r_p_PV_freeze[k] = l_r_p
            m_r_s_PV_freeze[k] = l_r_s
            m_r_v_PV_freeze[k] = l_r_v


# leading eigenvalues of the Jacobian matrix -- inhibition stabilization (Fig. 6 A)
plt.figure()

markers_on = [0, 30]

plt.plot(l_eig_E_PV_VIP_STP, '--D', markevery=markers_on, color=pal[0])
plt.plot(l_eig_E_SST_VIP_STP, '--D', markevery=markers_on, color=pal[3])

plt.xticks([0, 10, 20, 30, 40], [0, 5, 10, 15, 20])
plt.yticks([-10, -5, 0, 5, 10])

plt.xlabel(r'$\alpha$')
plt.ylabel('Leading eigenvalue')

plt.hlines(y=0, xmin=-2, xmax=42, colors='k', linestyles=[(0, (6, 6, 6, 6))])
plt.vlines(x=9, ymin=-10, ymax=10, colors='k', linestyles=[(0, (6, 6, 6, 6))])

plt.xlim([-2, 42])
plt.ylim([-10, 10])

plt.legend(['E-PV-VIP subnetwork', 'E-SST-VIP subnetwork'], loc='best')
plt.savefig('Leading_eigenvalue.png')
plt.close()

# example activity plots PV freeze -- inhibition stabilization  (Fig. 6 B)
for i in [0, 30]:
    plt.figure()

    plt.plot(m_r_e_PV_freeze[i]/m_r_e_PV_freeze[i][30000], color=pal[0])
    plt.plot(m_r_p_PV_freeze[i]/m_r_p_PV_freeze[i][30000], color=pal[1])
    plt.plot(m_r_s_PV_freeze[i]/m_r_s_PV_freeze[i][30000], color=pal[2])
    plt.plot(m_r_v_PV_freeze[i]/m_r_v_PV_freeze[i][30000], color=pal[3])

    plt.xticks(np.arange(30000, 90000 + 5000, 20000), np.arange(0, 6 + 0.5, 2))
    plt.xlabel('Time (s)')
    plt.ylabel('Normalized activity')
    plt.xlim([3000, 9000])

    if i==0:
        plt.yticks([0.9, 0.95, 1, 1.05, 1.1])
        plt.ylim([0.9, 1.1])
    if i==30:
        plt.yticks([0.98, 0.99, 1, 1.01, 1.02])
        plt.ylim([0.98, 1.02])

    plt.legend(['E', 'PV', 'SST', 'VIP'], loc='bottom left')

    plt.savefig('Network_activity_PV_freeze_alpha_' + str(l_alpha[i]) + '.png')
    plt.close()

# example activity plots SST freeze -- inhibition stabilization (Fig. 6 C)
for i in [0, 30]:
    plt.figure()

    plt.plot(m_r_e_SST_freeze[i]/m_r_e_SST_freeze[i][30000], color=pal[0])
    plt.plot(m_r_p_SST_freeze[i]/m_r_p_SST_freeze[i][30000], color=pal[1])
    plt.plot(m_r_s_SST_freeze[i]/m_r_s_SST_freeze[i][30000], color=pal[2])
    plt.plot(m_r_v_SST_freeze[i]/m_r_v_SST_freeze[i][30000], color=pal[3])

    plt.xticks(np.arange(30000, 90000 + 5000, 20000), np.arange(0, 6 + 0.5, 2))

    if i==0:
        plt.yticks([0.95, 1, 1.05, 1.1])
        plt.ylim([0.95, 1.1])
    if i==30:
        plt.yscale('log')
        plt.yticks([0.1, 1, 10, 100])
        plt.ylim([0.1, 100])

    plt.xlabel('Time (s)')
    plt.ylabel('Normalized activity')
    plt.xlim([30000, 90000])
    

    plt.legend(['E', 'PV', 'SST', 'VIP'], loc='upper right')

    plt.savefig('Network_activity_SST_freeze_alpha_' + str(l_alpha[i]) + '.png')
    plt.close()
