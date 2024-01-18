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

# neuronal parameters
tau_e, tau_p, tau_s, tau_v = 0.020, 0.010, 0.010, 0.010
alpha_e, alpha_p, alpha_s, alpha_v = 1.0, 1.0, 1.0, 1.0


c = 3
l_alpha = np.arange(0,20.1,.5)

ratio_inh_SST_PV = []
inh_total = np.zeros_like(l_alpha)
e_total = np.zeros_like(l_alpha)
input_sum = np.zeros_like(l_alpha)

for index, alpha in enumerate(l_alpha):
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
    l_inh_e, l_inh_p, l_inh_s, l_inh_v = [], [], [], []
    l_inh_e_p, l_inh_p_p, l_inh_v_p = [], [], []
    l_inh_e_s, l_inh_p_s, l_inh_v_s = [], [], []
    l_inh_s_v = []
    l_ex_e_e = []
    l_sum = []

    for i in range(T):
        if 50000 <= i < 70000:
            g_e, g_p, g_s, g_v = 4 + alpha, 4 + alpha, 3, 4 + c
        else:
            g_e, g_p, g_s, g_v = 4 + alpha, 4 + alpha, 3, 4

        z_e = Jee * r_e - x_ep * Jep * r_p - Jes * r_s - Jev * r_v + g_e
        z_p = Jpe * r_e - x_pp * Jpp * r_p - Jps * r_s - Jpv * r_v + g_p
        z_s = Jse * r_e - Jsp * r_p - Jss * r_s - Jsv * r_v + g_s
        z_v = Jve * r_e - x_vp * Jvp * r_p - u_vs * Jvs * r_s - Jvv * r_v + g_v

        r_e = r_e + (-r_e + z_e) / tau_e * dt
        r_p = r_p + (-r_p + z_p) / tau_p * dt
        r_s = r_s + (-r_s + z_s) / tau_s * dt
        r_v = r_v + (-r_v + z_v) / tau_v * dt

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

        inh_e_p = -x_ep * Jep * r_p
        inh_p_p = -x_pp * Jpp * r_p
        inh_v_p = -x_vp * Jvp * r_p

        inh_e_s = -Jes * r_s
        inh_p_s = -Jps * r_s
        inh_v_s = -u_vs * Jvs * r_s

        inh_s_v = -Jsv * r_v

        inh_e = inh_e_p + inh_e_s
        inh_p = inh_p_p + inh_p_s
        inh_s = inh_s_v
        inh_v = inh_v_p + inh_v_s

        ex_e_e = Jee * r_e

        if i == 49999:
            ratio_inh_SST_PV.append(inh_e_s/inh_e_p)
            inh_total[index] = inh_e_s + inh_e_p
            e_total[index] = ex_e_e
            input_sum[index] = inh_e + ex_e_e

        l_r_e.append(r_e)
        l_r_p.append(r_p)
        l_r_s.append(r_s)
        l_r_v.append(r_v)

        l_inh_e.append(inh_e)
        l_inh_p.append(inh_p)
        l_inh_s.append(inh_s)
        l_inh_v.append(inh_v)

        l_inh_e_p.append(inh_e_p)
        l_inh_p_p.append(inh_p_p)
        l_inh_v_p.append(inh_v_p)

        l_inh_e_s.append(inh_e_s)
        l_inh_p_s.append(inh_p_s)
        l_inh_v_s.append(inh_v_s)

        l_inh_s_v.append(inh_s_v)

        l_ex_e_e.append(ex_e_e)

        l_sum.append(inh_e + ex_e_e)

    l_r_e = np.asarray(l_r_e)
    l_r_p = np.asarray(l_r_p)
    l_r_s = np.asarray(l_r_s)
    l_r_v = np.asarray(l_r_v)

    l_inh_e = np.asarray(l_inh_e)
    l_inh_p = np.asarray(l_inh_p)
    l_inh_s = np.asarray(l_inh_s)
    l_inh_v = np.asarray(l_inh_v)

    l_inh_e_p = np.asarray(l_inh_e_p)
    l_inh_p_p = np.asarray(l_inh_p_p)
    l_inh_v_p = np.asarray(l_inh_v_p)

    l_inh_e_s = np.asarray(l_inh_e_s)
    l_inh_p_s = np.asarray(l_inh_p_s)
    l_inh_v_s = np.asarray(l_inh_v_s)

    l_inh_s_v = np.asarray(l_inh_s_v)

    l_ex_e_e = np.asarray(l_ex_e_e)

    l_sum = np.asarray(l_sum)


    # plot the input to E for low and high baseline activity(Fig. S7A, B)
    if alpha == 0 or alpha == 15:
        plt.figure()

        plt.plot(l_ex_e_e)
        plt.plot(l_inh_e_p)
        plt.plot(l_inh_e_s)
        plt.plot(l_inh_e)
        plt.plot(l_sum)
        
        plt.axhline(y=0, color='k', linestyle='--')

        plt.xticks(np.arange(30000, 90000 + 5000, 20000), np.arange(0, 7, 2))
        plt.xlim([30000, 90000])
        plt.xlabel('Time (s)')

        plt.ylabel('Input to E (a.u.)')

        plt.legend(['E', 'PV', 'SST', 'I', 'E + I'], loc='upper left')

        if alpha == 0:
            plt.hlines(y=7.9, xmin=50000, xmax=70000, color='gray')
            plt.yticks([-8, -4, 0, 4, 8])
            plt.ylim([-8, 8])
            plt.title('Low baseline state')
            plt.savefig('Fig_S7A.png')
        else:
            plt.hlines(y=59.9, xmin=50000, xmax=70000, color='gray')
            plt.yticks([-40, -20, 0, 20, 40, 60])
            plt.ylim([-40, 60])
            plt.title('High baseline state')
            plt.savefig('Fig_S7B.png')
            plt.close()

# plot the ratio of SST over PV to E inhibition for different alpha (Fig. S7C)
plt.figure()

plt.plot(ratio_inh_SST_PV)
plt.axhline(y=1, color='k', linestyle='--')

plt.yticks([0, 0.5, 1.0, 1.5, 2.0, 2.5])
plt.ylim([0, 2.5])

plt.xticks([0, 10, 20, 30, 40], [0, 5, 10, 15, 20])
plt.xlim([-2, 42])

plt.xlabel(r'$\alpha$')
plt.ylabel('Ratio of SST-to-E \n to PV-to-E inhibition')

plt.savefig('Fig_S7C.png')


