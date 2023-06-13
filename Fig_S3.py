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


l_alpha = np.arange(0,20.1,1)

c = 3

l_r_e_bs, l_r_p_bs, l_r_s_bs, l_r_v_bs = [], [], [], []

for alpha in l_alpha:

    r_e, r_p, r_s, r_v = 0, 0, 0, 0
    z_e, z_p, z_s, z_v = 0, 0, 0, 0

    l_r_e, l_r_p, l_r_s, l_r_v = [], [], [], []

    for i in range(T):
        if 50000 <= i < 70000:
            g_e, g_p, g_s, g_v = 4 + alpha, 4 + alpha, 3, 4 + c
        else:
            g_e, g_p, g_s, g_v = 4 + alpha, 4 + alpha, 3, 4

        z_e = Jee * r_e - Jep * r_p - Jes * r_s - Jev * r_v + g_e
        z_p = Jpe * r_e - Jpp * r_p - Jps * r_s - Jpv * r_v + g_p
        z_s = Jse * r_e - Jsp * r_p - Jss * r_s - Jsv * r_v + g_s
        z_v = Jve * r_e - Jvp * r_p - Jvs * r_s - Jvv * r_v + g_v

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

        l_r_e.append(r_e)
        l_r_p.append(r_p)
        l_r_s.append(r_s)
        l_r_v.append(r_v)

    l_r_e = np.asarray(l_r_e)
    l_r_p = np.asarray(l_r_p)
    l_r_s = np.asarray(l_r_s)
    l_r_v = np.asarray(l_r_v)

    l_r_e_bs.append(np.mean(l_r_e[35000:45000]))
    l_r_p_bs.append(np.mean(l_r_p[35000:45000]))
    l_r_s_bs.append(np.mean(l_r_s[35000:45000]))
    l_r_v_bs.append(np.mean(l_r_v[35000:45000]))


    # plotting
    s_title_1 = 'top_down_modulation'
    s_title_t1 = 'top down modulation'

    s_title_2 = 'without_STP'
    s_title_t2 = 'without STP'

    if (alpha == 0 or alpha == 15):
        plt.figure()

        plt.plot(l_r_e)
        plt.plot(l_r_p)
        plt.plot(l_r_s)
        plt.plot(l_r_v)

        plt.xticks(np.arange(30000, 90000 + 5000, 20000), np.arange(0, 6 + 0.5, 2))

        if alpha == 0:
            plt.yticks([0, 5, 10, 15])
        else:
            plt.yticks([0, 10, 20, 30])

        plt.xlabel('Time (s)')
        plt.ylabel('Firing rate (Hz)')
        plt.title(str(s_title_t1) + ' ' + str(s_title_t2))
        plt.hlines(y=np.mean(l_r_s[35000:45000]), xmin=30000, xmax=90000, colors='k', linestyles=[(0, (6, 6, 6, 6))])
        plt.xlim([30000, 90000])

        if alpha == 0:
            plt.ylim([0, 15])
        else:
            plt.ylim([0, 30])

        plt.legend(['E', 'PV', 'SST', 'VIP'], loc='upper left')

        if alpha == 0:
            plt.savefig('Low_baseline_' + str(s_title_1) + '_' + str(s_title_2) + '.png')
        else:
            plt.savefig('High_baseline_' + str(s_title_1) + '_' + str(s_title_2) + '.png')