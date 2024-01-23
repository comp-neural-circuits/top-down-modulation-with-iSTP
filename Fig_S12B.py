import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D

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


def compute_response_reversal_index(y):
    """Compute the slope between two points and detect a zero crossing."""

    slope = y[1] - y[0]
    has_zero_crossing = y[0] * y[1] < 0

    if has_zero_crossing:
        if slope > 0:
            return slope
        return -5
    else:
        return 0


def get_limits(low: float, high: float):
    """Returns offset limits for axes"""
    offset = (high - low) / 40
    return [low - offset, high + offset]


def get_hist_limits(low: float, high: float):
    """Returns offset limits for axes"""
    offset = (high - low) / 40
    return [low, high + offset]


# simulation setup
dt = 0.0001
T = int(9 / dt)

# neuronal parameters
tau_e, tau_p, tau_s, tau_v = 0.020, 0.010, 0.010, 0.010
alpha_e, alpha_p, alpha_s, alpha_v = 1.0, 1.0, 1.0, 1.0

# network connectivity
Jee_range = [1.2, 2.2]
Jep = 1.7
Jes = 1.4
Jev = 0

Jpe = 2.2
Jpp = 1.6
Jps = 1.1
Jpv = 0

Jse = 1.0
Jsp = 0
Jss = 0
Jsv = 0.6

Jve = 1.3
Jvp = 0.4
Jvs = 0.4
Jvv = 0

l_alpha = [0, 20]
c = 3

g_1 = 4
g_2 = 3

step_size = 0.01

l_Jee = np.round(np.arange(Jee_range[0], Jee_range[1] + step_size, step_size), 2)

m_unstable = np.zeros((len(l_Jee)))
m_response_reversal_index = np.zeros((len(l_Jee)))

for k, Jee in enumerate(l_Jee):
    l_change_SST = []
    unstable = False

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
            # clip firing rate to keep the code stable
            if r_e > 400:
                unstable = True
                m_unstable[k] = 1
                r_e = 400
            if r_p > 400:
                unstable = True
                m_unstable[k] = 1
                r_p = 400
            if r_s > 400:
                unstable = True
                m_unstable[k] = 1
                r_s = 400
            if r_v > 400:
                unstable = True
                m_unstable[k] = 1
                r_v = 400

            if unstable:
                break

            if 50000 <= i < 70000:
                g_e, g_p, g_s, g_v = g_1 + alpha, g_1 + alpha, g_2, g_1 + c
            else:
                g_e, g_p, g_s, g_v = g_1 + alpha, g_1 + alpha, g_2, g_1

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
            x_ep = x_ep + ((1 - x_ep) / tau_x - u_s * x_ep * r_p) * dt
            x_ep = np.clip(x_ep, 0, 1)

            x_pp = x_pp + ((1 - x_pp) / tau_x - u_s * x_pp * r_p) * dt
            x_pp = np.clip(x_pp, 0, 1)

            x_vp = x_vp + ((1 - x_vp) / tau_x - u_s * x_vp * r_p) * dt
            x_vp = np.clip(x_vp, 0, 1)

            # STF
            u_vs = u_vs + ((1 - u_vs) / tau_u + U * (U_max - u_vs) * r_s) * dt
            u_vs = np.clip(u_vs, 1, U_max)

            l_r_e.append(r_e)
            l_r_p.append(r_p)
            l_r_s.append(r_s)
            l_r_v.append(r_v)

        l_r_e = np.asarray(l_r_e)
        l_r_p = np.asarray(l_r_p)
        l_r_s = np.asarray(l_r_s)
        l_r_v = np.asarray(l_r_v)

        # compute the change in SST
        if len(l_r_s) > 65000:
            l_change_SST.append(
                np.mean(l_r_s[60000:65000]) - np.mean(l_r_s[40000:45000])
            )
        else:
            l_change_SST.append(0)

        if len(l_r_e) > 45000:
            if any(
                [
                    np.mean(l_r_e[40000:45000]) <= 0.1,
                    np.mean(l_r_p[40000:45000]) <= 0.1,
                    np.mean(l_r_v[40000:45000]) <= 0.1,
                    np.mean(l_r_s[40000:45000]) <= 0.1,
                ]
            ):
                unstable = True
                m_unstable[k] = 2

    # compute response reversal index
    if not unstable:
        m_response_reversal_index[k] = compute_response_reversal_index(l_change_SST)
    else:
        m_response_reversal_index[k] = -10

regime = m_unstable

indices_unstable = regime == 1
indices_one_dead = regime == 2

indices_stable_reversal = np.logical_and(regime == 0, m_response_reversal_index != 0)
indices_stable_no_reversal = np.logical_and(regime == 0, m_response_reversal_index == 0)
indices_unstable_or_one_dead = np.logical_or(indices_unstable, indices_one_dead)
first_index_unstable = np.argmax(indices_unstable_or_one_dead)

regime[indices_stable_no_reversal] = 3

m_response_reversal_index[indices_unstable] = -4
m_response_reversal_index[indices_one_dead] = -2

################################

colors = {0: "red", 1: "blue", 2: "grey", 3: "white"}
color_map = mcolors.ListedColormap(list(colors.values()))

################################

# RRI vs ratio
plt.figure()

y_values = m_response_reversal_index
color = regime
x = l_Jee

plt.scatter(
    x,
    y_values,
    c=color,
    cmap=color_map,
    vmin=0,
    vmax=3,
    edgecolors="black",
    linewidths=3,
    s=300,
    alpha=0.5,
)

plt.axvline(x=l_Jee[first_index_unstable] - (step_size / 2), color="k", linestyle="--")
plt.axvline(x=1.0, color="k", linestyle="--")
plt.axhline(y=0, color="k", linestyle="--")

plt.xticks(np.round(np.arange(Jee_range[0], Jee_range[1] + 0.2, 0.2), 1))
plt.yticks([-5, 0, 20, 40, 60])

plt.xlabel(r"$J_{EE}$")
plt.ylabel("RRI")

plt.ylim(get_limits(-5, 60))
plt.xlim(get_limits(Jee_range[0], Jee_range[1]))

legend_elements = [
    Line2D(
        [0],
        [0],
        marker="o",
        color="w",
        label="Response reversal",
        markerfacecolor="red",
        markersize=20,
        alpha=0.5,
    ),
    Line2D(
        [0],
        [0],
        marker="o",
        color="w",
        label="Unstable networks",
        markerfacecolor="blue",
        markersize=20,
        alpha=0.5,
    ),
    Line2D(
        [0],
        [0],
        marker="o",
        color="w",
        label="Partially silent",
        markerfacecolor="grey",
        markersize=20,
        alpha=0.5,
    ),
    Line2D(
        [0],
        [0],
        marker="o",
        color="w",
        label="No response reversal",
        markerfacecolor="white",
        markersize=20,
        alpha=0.5,
    ),
]
plt.legend(handles=legend_elements, loc="upper right")

plt.savefig("Fig_S12B_left.png")
plt.close()

################################
hist_all, bin_edges_all = np.histogram(
    m_response_reversal_index, bins=65, range=(-5, 60)
)

hist_stable, bin_edges_stable = np.histogram(
    m_response_reversal_index[indices_stable_reversal], bins=65, range=(-5, 60)
)
hist_unstable, bin_edges_unstable = np.histogram(
    m_response_reversal_index[indices_unstable], bins=65, range=(-5, 60)
)
hist_one_dead, bin_edges_one_dead = np.histogram(
    m_response_reversal_index[indices_one_dead], bins=65, range=(-5, 60)
)
hist_no_reversal, bin_edges_no_reversal = np.histogram(
    m_response_reversal_index[indices_stable_no_reversal], bins=65, range=(-5, 60)
)
max_hist = np.ceil(max(hist_all) / 25) * 25

# hist of RRI
plt.figure()

left = np.zeros(len(hist_stable))
plt.barh(
    bin_edges_stable[:-1],
    hist_stable,
    height=1,
    color="red",
    alpha=0.5,
    left=left,
    edgecolor="black",
    label="Response reversal",
)
left += hist_stable
plt.barh(
    bin_edges_unstable[:-1],
    hist_unstable,
    height=1,
    color="blue",
    alpha=0.5,
    left=left,
    edgecolor="black",
    label="Unstable networks",
)
left += hist_unstable
plt.barh(
    bin_edges_one_dead[:-1],
    hist_one_dead,
    height=1,
    color="grey",
    alpha=0.5,
    left=left,
    edgecolor="black",
    label="Partially silent",
)
left += hist_one_dead
plt.barh(
    bin_edges_no_reversal[:-1],
    hist_no_reversal,
    height=1,
    color="white",
    alpha=0.5,
    left=left,
    edgecolor="black",
    label="No response reversal",
)

plt.xticks([0, max_hist])
plt.yticks([-5, 0, 20, 40, 60])

plt.xlabel("Count")
plt.ylabel("RRI")

plt.xlim(get_hist_limits(0, max_hist))
plt.ylim(get_limits(-5, 60))

plt.legend(loc="upper right")

plt.savefig("Fig_S12B_middle.png")
plt.close()

################################
num_stable = np.sum(regime == 0)
num_unstable = np.sum(regime == 1)
num_one_dead = np.sum(regime == 2)
num_no_reversal = np.sum(regime == 3)

max_hist = (
    np.ceil(np.max([num_stable, num_unstable, num_one_dead, num_no_reversal]) / 25) * 25
)

# hist of regime
plt.figure()

plt.bar(
    0.5,
    num_stable,
    width=0.5,
    color="red",
    alpha=0.5,
    edgecolor="black",
    label="Response reversal",
)
plt.bar(
    1.5,
    num_unstable,
    width=0.5,
    color="blue",
    alpha=0.5,
    edgecolor="black",
    label="Unstable networks",
)
plt.bar(
    2.5,
    num_one_dead,
    width=0.5,
    color="grey",
    alpha=0.5,
    edgecolor="black",
    label="Partially silent",
)
plt.bar(
    3.5,
    num_no_reversal,
    width=0.5,
    color="white",
    alpha=0.5,
    edgecolor="black",
    label="No response reversal",
)

plt.xticks([0.5, 1.5, 2.5, 3.5], ["", "", "", ""])
plt.yticks(np.linspace(0, max_hist, 6))

plt.xlabel("Regime")
plt.ylabel("Count")

plt.ylim(get_hist_limits(0, max_hist))
plt.xlim(get_limits(0, 4))

plt.legend(loc="upper right")

plt.savefig("Fig_S12B_right.png")
plt.close()
