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

c = 3

l_alpha = np.arange(0,20.1,.5)
inh_total_low, inh_total_high = np.zeros_like(l_alpha), np.zeros_like(l_alpha)
e_low, e_high = np.zeros_like(l_alpha), np.zeros_like(l_alpha)
sum_low, sum_high = np.zeros_like(l_alpha), np.zeros_like(l_alpha)

for index, alpha in enumerate(l_alpha):
    l_inh_e, l_e_e, l_sum = [], [], []

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
        x_ep = x_ep + ((1 - x_ep) / tau_x - u_s * x_ep * r_p) * dt
        x_ep = np.clip(x_ep, 0, 1)

        x_pp = x_pp + ((1 - x_pp) / tau_x - u_s * x_pp * r_p) * dt
        x_pp = np.clip(x_pp, 0, 1)

        x_vp = x_vp + ((1 - x_vp) / tau_x - u_s * x_vp * r_p) * dt
        x_vp = np.clip(x_vp, 0, 1)

        # STF
        u_vs = u_vs + ((1 - u_vs) / tau_u + U * (U_max - u_vs) * r_s) * dt
        u_vs = np.clip(u_vs, 1, U_max)

        # input to E
        inh_e_p = -x_ep * Jep * r_p
        inh_e_s = -Jes * r_s
        inh_e = inh_e_p + inh_e_s

        ex_e_e = Jee * r_e

        sum_input = inh_e + ex_e_e

        l_inh_e.append(inh_e)
        l_e_e.append(ex_e_e)
        l_sum.append(sum_input)

        if i == 49998:
            inh_total_low[index] = inh_e
            e_low[index] = ex_e_e
            sum_low[index] = sum_input
        if i == 69998:
            inh_total_high[index] = inh_e
            e_high[index] = ex_e_e
            sum_high[index] = sum_input

    l_inh_e = np.array(l_inh_e)
    l_e_e = np.array(l_e_e)
    l_sum = np.array(l_sum)

    # input to E plot at low and high baseline states (Fig. 2H, I)
    if (alpha == 0 or alpha == 15):
        plt.figure()

        plt.plot(l_e_e)
        plt.plot(l_inh_e, color='red')
        plt.plot(l_sum, color='purple')
        
        plt.axhline(y=0, color='k', linestyle='--')

        plt.xticks(np.arange(30000, 90000 + 5000, 20000), np.arange(0, 6 + 0.5, 2))
        plt.xlim([30000, 90000])
        plt.xlabel('Time (s)')

        plt.ylabel('Input to E (a.u.)')        

        plt.legend(['E', 'I', 'E+I'], loc='upper left')

        if alpha == 0:
            plt.hlines(y=7.9, xmin=50000, xmax=70000, color='gray')
            
            plt.title('Low baseline state')

            plt.yticks([-8, -4, 0, 4, 8])
            plt.ylim([-8, 8])

            plt.savefig('Fig_2H.png')
            plt.close()
        else:
            plt.hlines(y=59.9, xmin=50000, xmax=70000, color='gray')
            plt.title('High baseline state')
            plt.yticks([-40, -20, 0, 20, 40, 60])
            plt.ylim([-40, 60])

            plt.savefig('Fig_2I.png')
            plt.close()

# difference in input to E between baseline and top-down stimulation (Fig. 2J)
plt.figure()

s_title_t1 = 'Input to E stim-baseline'

plt.plot(e_high - e_low)
plt.plot(inh_total_high - inh_total_low, color='red')
plt.plot(sum_high - sum_low, color='purple')

plt.yticks([-10, 0, 10, 20, 30, 40])
plt.ylim([-10, 40])

plt.xticks([0, 10, 20, 30, 40], [0, 5, 10, 15, 20])
plt.xlim([-2, 42])

plt.axhline(y=0, color='k', linestyle='--')

plt.xlabel(r'$\alpha$')
plt.ylabel('Change in input to E (a.u.)')
plt.title(str(s_title_t1))

plt.legend(['E', 'I', 'E+I'], loc='upper left')

plt.savefig('Fig_2J.png')
plt.close()

