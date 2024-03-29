import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

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

l_alpha = np.arange(20, -0.1, -1)

x_ep_ini, x_pp_ini, x_vp_ini = 1, 1, 1
u_vs_ini = 1

l_intact, l_inactive_PV_E, l_inactive_PV_PV, l_inactive_PV_VIP, l_inactive_SST_VIP = (
    [],
    [],
    [],
    [],
    [],
)
l_PV_E, l_PV_E_base = [], []

l_r_baseline = np.zeros((len(l_alpha), 4, 5))
titles = []

for i, alpha in enumerate(l_alpha):
    for m in range(5):
        # depression variables
        x_ep, x_pp, x_vp = x_ep_ini, x_pp_ini, x_vp_ini
        u_s = 1
        tau_x = 0.10

        # facilitation variables
        u_vs = u_vs_ini
        U, U_max = 1, 3
        tau_u = 0.40

        r_e, r_p, r_s, r_v = 0, 0, 0, 0
        z_e, z_p, z_s, z_v = 0, 0, 0, 0

        l_r_e, l_r_p, l_r_s, l_r_v = [], [], [], []

        for t in range(T):
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
            if m != 1:
                x_ep = x_ep + ((1 - x_ep) / tau_x - u_s * x_ep * r_p) * dt
                x_ep = np.clip(x_ep, 0, 1)
            else:
                if alpha == np.max(l_alpha):
                    if t <= 50000:
                        x_ep = x_ep + ((1 - x_ep) / tau_x - u_s * x_ep * r_p) * dt
                        x_ep = np.clip(x_ep, 0, 1)
                        x_ep_ini = x_ep
                    else:
                        pass
                else:
                    pass

            if m != 2:
                x_pp = x_pp + ((1 - x_pp) / tau_x - u_s * x_pp * r_p) * dt
                x_pp = np.clip(x_pp, 0, 1)
            else:
                if alpha == np.max(l_alpha):
                    if t <= 50000:
                        x_pp = x_pp + ((1 - x_pp) / tau_x - u_s * x_pp * r_p) * dt
                        x_pp = np.clip(x_pp, 0, 1)
                        x_pp_ini = x_pp
                    else:
                        pass
                else:
                    pass

            if m != 3:
                x_vp = x_vp + ((1 - x_vp) / tau_x - u_s * x_vp * r_p) * dt
                x_vp = np.clip(x_vp, 0, 1)
            else:
                if alpha == np.max(l_alpha):
                    if t <= 50000:
                        x_vp = x_vp + ((1 - x_vp) / tau_x - u_s * x_vp * r_p) * dt
                        x_vp = np.clip(x_vp, 0, 1)
                        x_vp_ini = x_vp
                    else:
                        pass
                else:
                    pass

            # STF
            if m != 4:
                u_vs = u_vs + ((1 - u_vs) / tau_u + U * (U_max - u_vs) * r_s) * dt
                u_vs = np.clip(u_vs, 1, U_max)
            else:
                if alpha == np.max(l_alpha):
                    if t <= 50000:
                        u_vs = (
                            u_vs + ((1 - u_vs) / tau_u + U * (U_max - u_vs) * r_s) * dt
                        )
                        u_vs = np.clip(u_vs, 1, U_max)
                        u_vs_ini = u_vs
                    else:
                        pass
                else:
                    pass

            if m == 0 and t == 40000:
                l_PV_E.append(-x_ep * Jep * r_p)
                l_PV_E_base.append(-Jep * r_p)

            l_r_e.append(r_e)
            l_r_p.append(r_p)
            l_r_s.append(r_s)
            l_r_v.append(r_v)

        l_r_e = np.asarray(l_r_e)
        l_r_p = np.asarray(l_r_p)
        l_r_s = np.asarray(l_r_s)
        l_r_v = np.asarray(l_r_v)

        if m == 0:
            l_intact.append(np.mean(l_r_e[35000:45000]))
        elif m == 1:
            l_inactive_PV_E.append(np.mean(l_r_e[35000:45000]))
        elif m == 2:
            l_inactive_PV_PV.append(np.mean(l_r_e[35000:45000]))
        elif m == 3:
            l_inactive_PV_VIP.append(np.mean(l_r_e[35000:45000]))
        else:
            l_inactive_SST_VIP.append(np.mean(l_r_e[35000:45000]))

# excitatory activity as function of alpha (Fig. S5A)
plt.figure()

norm = mpl.colors.Normalize(vmin=1, vmax=10)
cmap_blue = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.Blues)
cmap_purple = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.Purples)

l_intact.reverse()
l_inactive_PV_E.reverse()
l_inactive_PV_PV.reverse()
l_inactive_PV_VIP.reverse()
l_inactive_SST_VIP.reverse()

plt.plot(l_intact, color=cmap_blue.to_rgba(4 + 3))
plt.plot(l_inactive_PV_PV, color=cmap_blue.to_rgba(3 + 3))
plt.plot(l_inactive_PV_VIP, color=cmap_blue.to_rgba(2 + 3))
plt.plot(l_inactive_SST_VIP, color=cmap_blue.to_rgba(1 + 3))
plt.plot(l_inactive_PV_E, color=cmap_purple.to_rgba(6 + 3))

plt.xticks([0, 5, 10, 15, 20], [0, 5, 10, 15, 20])
plt.yticks([0, 15, 30, 45])

plt.xlabel(r"$\alpha$")
plt.ylabel("Firing rate (a.u.)")
plt.xlim([-1, 21])
plt.ylim([0, 45])

plt.legend(
    [
        "all mechanisms",
        "Inactive PV-to-PV STD",
        "Inactive PV-to-VIP STD",
        "Inactive SST-to-VIP STF",
        "Inactive PV-to-E STD",
    ],
    loc="best",
)

plt.savefig("Fig_S5A.png")

# PV-E synaptic strength as function of alpha (Fig. S5B)
plt.figure()

l_PV_E.reverse()
l_PV_E_base.reverse()

plt.plot(l_PV_E, color="pink")
plt.plot(l_PV_E_base, color="pink", linestyle="--")

plt.xticks([0, 5, 10, 15, 20], [0, 5, 10, 15, 20])
plt.xlabel(r"$\alpha$")
plt.xlim([-1, 21])

plt.yticks([-25, -20, -15, -10, -5, 0])
plt.ylabel("Inh. input to E")
plt.ylim([-25, 0])

plt.legend(["PV", "PV non-plastic"], loc="best")
plt.savefig("Fig_S5B.png")
