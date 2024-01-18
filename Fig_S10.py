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
Jee = 1.7
Jep = 2.1
Jes = 1.5
Jev = 0

Jpe = 1.0
Jpp = 1.2
Jps = 1.3
Jpv = 0

Jse = 0.7
Jsp = 0
Jss = 0
Jsv = 0.4

Jve = 0.9
Jvp = 0.5
Jvs = 0.4
Jvv = 0

g_1 = 5.0
g_2 = 3.0

l_factor = np.arange(0, 20.1, 0.5)
c = 3

l_SST_change = []

for factor in l_factor:
    # depression variables
    x_ee, x_pe = 1, 1
    x_ep, x_pp, x_vp = 1, 1, 1
    x_es, x_ps = 1, 1

    u_d_ee, u_d_pe = 0.19, 0.04
    u_d_ep, u_d_pp, u_d_vp = 0.49, 0.50, 0.37
    u_d_es, u_d_ps = 0.12, 0.11

    tau_x = 0.10
    tau_x_ee = 0.01

    # facilitation variables
    u_se = 1
    u_ve = 1
    u_vs = 1
    u_sv = 1

    u_f_se, u_f_ve, u_f_vs, u_f_sv = 0.18, 0.03, 0.28, 0.05

    U_max = 3
    U_max_se = 2
    tau_u = 0.40

    r_e, r_p, r_s, r_v = 0, 0, 0, 0
    z_e, z_p, z_s, z_v = 0, 0, 0, 0

    l_r_e, l_r_p, l_r_s, l_r_v = [], [], [], []
    l_R_SV_temp = []
    
    for i in range(T):
        if 50000 <= i < 70000:
            g_e, g_p, g_s, g_v = g_1 + factor, g_1 + factor, g_2, g_1 + c + 1
        else:
            g_e, g_p, g_s, g_v = g_1 + factor, g_1 + factor, g_2, g_1 + 1

        z_e = x_ee * Jee * r_e - x_ep * Jep * r_p - x_es * Jes * r_s - Jev * r_v + g_e
        z_p = x_pe * Jpe * r_e - x_pp * Jpp * r_p - x_ps * Jps * r_s - Jpv * r_v + g_p
        z_s = u_se * Jse * r_e - Jsp * r_p - Jss * r_s - u_sv * Jsv * r_v + g_s
        z_v = u_ve * Jve * r_e - x_vp * Jvp * r_p - u_vs * Jvs * r_s - Jvv * r_v + g_v

        z_e = z_e * (z_e > 0)
        z_p = z_p * (z_p > 0)
        z_s = z_s * (z_s > 0)
        z_v = z_v * (z_v > 0)

        r_e = r_e + (-r_e + z_e) / tau_e * dt
        r_p = r_p + (-r_p + z_p) / tau_p * dt
        r_s = r_s + (-r_s + z_s) / tau_s * dt
        r_v = r_v + (-r_v + z_v) / tau_v * dt

        r_e = r_e * (r_e > 0)
        r_p = r_p * (r_p > 0)
        r_s = r_s * (r_s > 0)
        r_v = r_v * (r_v > 0)

        # STD
        x_ee = x_ee + ((1 - x_ee) / tau_x_ee - u_d_ee * x_ee * r_e) * dt
        x_ee = np.clip(x_ee, 0, 1)

        x_pe = x_pe + ((1 - x_pe) / tau_x - u_d_pe * x_pe * r_e) * dt
        x_pe = np.clip(x_pe, 0, 1)

        x_ep = x_ep + ((1 - x_ep) / tau_x - u_d_ep * x_ep * r_p) * dt
        x_ep = np.clip(x_ep, 0, 1)

        x_pp = x_pp + ((1 - x_pp) / tau_x - u_d_pp * x_pp * r_p) * dt
        x_pp = np.clip(x_pp, 0, 1)

        x_vp = x_vp + ((1 - x_vp) / tau_x - u_d_vp * x_vp * r_p) * dt
        x_vp = np.clip(x_vp, 0, 1)

        x_es = x_es + ((1 - x_es) / tau_x - u_d_es * x_es * r_s) * dt
        x_es = np.clip(x_es, 0, 1)

        x_ps = x_ps + ((1 - x_ps) / tau_x - u_d_ps * x_ps * r_s) * dt
        x_ps = np.clip(x_ps, 0, 1)

        # STF
        u_se = u_se + ((1 - u_se) / tau_u + u_f_se * (U_max_se - u_se) * r_e) * dt
        u_se = np.clip(u_se, 1, U_max_se)

        u_ve = u_ve + ((1 - u_ve) / tau_u + u_f_ve * (U_max - u_ve) * r_e) * dt
        u_ve = np.clip(u_ve, 1, U_max)

        u_vs = u_vs + ((1 - u_vs) / tau_u + u_f_vs * (U_max - u_vs) * r_s) * dt
        u_vs = np.clip(u_vs, 1, U_max)

        u_sv = u_sv + ((1 - u_sv) / tau_u + u_f_sv * (U_max - u_sv) * r_v) * dt
        u_sv = np.clip(u_sv, 1, U_max)
            
        l_r_e.append(r_e)
        l_r_p.append(r_p)
        l_r_s.append(r_s)
        l_r_v.append(r_v)
        
    l_r_e = np.asarray(l_r_e)
    l_r_p = np.asarray(l_r_p)
    l_r_s = np.asarray(l_r_s)
    l_r_v = np.asarray(l_r_v)

    l_SST_change.append(np.mean(l_r_s[65000:70000]) - np.mean(l_r_s[40000:45000]))
    
    # Network responses to the perturbation (Fig. S10A, B)
    if factor in [0, 20]:
        plt.figure()

        plt.plot(l_r_e)
        plt.plot(l_r_p)
        plt.plot(l_r_s)
        plt.plot(l_r_v)

        plt.xticks(np.arange(30000, 90000 + 5000, 20000), np.arange(0, 7, 2))

        if factor == 0:
            plt.yticks([0, 5, 10, 15])
            s_title = 'Low baseline state'
        else:
            plt.yticks([0, 5, 10, 15, 20])
            s_title = 'High baseline state'

        plt.xlabel('Time (s)')
        plt.ylabel('Firing rate (a.u.)')
        plt.title(str(s_title))
        plt.axhline(y=np.mean(l_r_s[35000:45000]), color='k', linestyle='--')
        plt.xlim([30000, 90000])

        if factor == 0:
            plt.ylim([0, 15])
        else:
            plt.ylim([0, 20])

        plt.legend(['E', 'PV', 'SST', 'VIP'], loc='upper right')

        if factor == 0:
            plt.hlines(y=14.9, xmin=50000, xmax=70000, color='gray')
            plt.savefig('Fig_S10A.png')
        else:
            plt.hlines(y=19.9, xmin=50000, xmax=70000, color='gray')
            plt.savefig('Fig_S10B.png')
        plt.close()

# Change in SST activity (Fig. S10C)
plt.figure()

plt.plot(l_SST_change, color='gray')
plt.axhline(y=0, color='k', linestyle='--')

plt.xticks([0, 10, 20, 30, 40], [0, 5, 10, 15, 20])
plt.xlim([-2, 42])

plt.yticks([-1, -0.5, 0, .5, 1.0])
plt.ylim([-1.0, 1.0])

plt.xlabel(r'$\alpha$')
plt.ylabel('Change in SST activity (a.u.)')

plt.title(r'Large perturbation (large $\delta g_V$)')
plt.savefig('Fig_S10C.png')
