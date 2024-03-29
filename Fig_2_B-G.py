import numpy as np
import matplotlib.pyplot as plt

# increase the figure size
plt.rcParams["figure.figsize"] = [12, 10]

# remove the top and right spines from plot in the global plt setting
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False
# change the linewidth of the axes and spines
plt.rcParams["axes.linewidth"] = 2
plt.rcParams["lines.linewidth"] = 4
plt.rcParams["xtick.major.size"] = 10
plt.rcParams["xtick.major.width"] = 2
plt.rcParams["ytick.major.size"] = 10
plt.rcParams["ytick.major.width"] = 2
plt.rcParams["xtick.minor.size"] = 5
plt.rcParams["xtick.minor.width"] = 2
plt.rcParams["ytick.minor.size"] = 5
plt.rcParams["ytick.minor.width"] = 2
# change the fontsize of the ticks label
plt.rcParams["xtick.labelsize"] = 20
plt.rcParams["ytick.labelsize"] = 20
# change the fontsize of the axes label
plt.rcParams["axes.labelsize"] = 20
# change the fontsize of the legend
plt.rcParams["legend.fontsize"] = 20
# change the fontsize of the title
plt.rcParams["axes.titlesize"] = 20
# change the title font size
plt.rcParams["font.size"] = 20

# change the font family to Arial
plt.rcParams["font.family"] = "Arial"

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

l_b_STP = [True, False]
l_alpha = np.arange(0, 20.1, 1)
l_change_in_SST, l_change_in_SST_STP = [], []

for b_STP in l_b_STP:
    for alpha in l_alpha:
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

            if b_STP:
                z_e = Jee * r_e - x_ep * Jep * r_p - Jes * r_s - Jev * r_v + g_e
                z_p = Jpe * r_e - x_pp * Jpp * r_p - Jps * r_s - Jpv * r_v + g_p
                z_s = Jse * r_e - Jsp * r_p - Jss * r_s - Jsv * r_v + g_s
                z_v = Jve * r_e - x_vp * Jvp * r_p - u_vs * Jvs * r_s - Jvv * r_v + g_v
            else:
                z_e = Jee * r_e - Jep * r_p - Jes * r_s - Jev * r_v + g_e
                z_p = Jpe * r_e - Jpp * r_p - Jps * r_s - Jpv * r_v + g_p
                z_s = Jse * r_e - Jsp * r_p - Jss * r_s - Jsv * r_v + g_s
                z_v = Jve * r_e - Jvp * r_p - Jvs * r_s - Jvv * r_v + g_v

            r_e = r_e + (-r_e + z_e) / tau_e * dt
            r_p = r_p + (-r_p + z_p) / tau_p * dt
            r_s = r_s + (-r_s + z_s) / tau_s * dt
            r_v = r_v + (-r_v + z_v) / tau_v * dt

            r_e = r_e * (r_e > 0)
            r_p = r_p * (r_p > 0)
            r_s = r_s * (r_s > 0)
            r_v = r_v * (r_v > 0)

            if b_STP:
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

            else:
                pass

            l_r_e.append(r_e)
            l_r_p.append(r_p)
            l_r_s.append(r_s)
            l_r_v.append(r_v)

        l_r_e = np.asarray(l_r_e)
        l_r_p = np.asarray(l_r_p)
        l_r_s = np.asarray(l_r_s)
        l_r_v = np.asarray(l_r_v)

        if b_STP:
            l_change_in_SST_STP.append(
                np.mean(l_r_s[60000:65000]) - np.mean(l_r_s[40000:45000])
            )
        else:
            l_change_in_SST.append(
                np.mean(l_r_s[60000:65000]) - np.mean(l_r_s[40000:45000])
            )

        # plotting
        s_title_1 = "top_down_modulation"
        s_title_t1 = "top down modulation"

        # network activity with and iSTP mechanisms for low and high baseline states (Fig. 2B,C,E,F)
        if alpha == 0 or alpha == 15:
            plt.figure()

            plt.plot(l_r_e)
            plt.plot(l_r_p)
            plt.plot(l_r_s)
            plt.plot(l_r_v)

            plt.axhline(y=np.mean(l_r_s[35000:45000]), color="k", linestyle="--")

            plt.xticks(np.arange(30000, 90000 + 5000, 20000), np.arange(0, 6 + 0.5, 2))
            plt.xlim([30000, 90000])
            plt.xlabel("Time (s)")
            plt.ylabel("Firing rate (a.u.)")

            plt.legend(["E", "PV", "SST", "VIP"], loc="upper left")

            if alpha == 0:
                plt.hlines(y=14.9, xmin=50000, xmax=70000, color="gray")
                plt.yticks([0, 5, 10, 15])
                plt.ylim([0, 15])
                if b_STP:
                    s_title_2 = "with_STP"
                    s_title_t2 = "with STP"
                    plt.title("Low baseline " + str(s_title_t2))
                    plt.savefig("Fig_2B.png")
                    plt.close()
                else:
                    s_title_2 = "without_STP"
                    s_title_t2 = "without STP"
                    plt.title("Low baseline " + str(s_title_t2))
                    plt.savefig("Fig_2E.png")
                    plt.close()
            else:
                if b_STP:
                    plt.hlines(y=59.9, xmin=50000, xmax=70000, color="gray")
                    plt.yticks([0, 20, 40, 60])
                    plt.ylim([0, 60])
                    plt.title("High baseline " + str(s_title_t2))
                    plt.savefig("Fig_2C.png")
                    plt.close()
                else:
                    plt.hlines(y=14.9, xmin=50000, xmax=70000, color="gray")
                    plt.yticks([0, 5, 10, 15])
                    plt.ylim([0, 15])
                    plt.title("High baseline " + str(s_title_t2))
                    plt.savefig("Fig_2F.png")
                    plt.close()


# Change in SST activity with STP mechanisms for different baseline activity levels (Fig. 2D)
plt.figure()
plt.plot(l_change_in_SST_STP, color="gray")

plt.xticks([0, 5, 10, 15, 20], [0, 5, 10, 15, 20])
plt.yticks([-10, -5, 0, 5, 10])

plt.xlabel(r"$\alpha$")
plt.ylabel("Change in SST activity (a.u.)")
plt.xlim([-1, 21])
plt.ylim([-10, 10])
plt.axhline(y=0, color="k", linestyle="--")
plt.title("Network with iSTP")

plt.savefig("Fig_2D.png")
plt.close()

# Change in SST activity with STP mechanisms for different baseline activity levels (Fig. 2G)
plt.figure()
plt.plot(l_change_in_SST, color="gray")

plt.xticks([0, 5, 10, 15, 20], [0, 5, 10, 15, 20])
plt.yticks([-10, -5, 0, 5, 10])

plt.xlabel(r"$\alpha$")
plt.ylabel("Change in SST activity (a.u.)")
plt.xlim([-1, 21])
plt.ylim([-10, 10])
plt.axhline(y=0, color="k", linestyle="--")
plt.title("Network without iSTP")

plt.savefig("Fig_2G.png")
plt.close()
