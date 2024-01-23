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


# F-I curve
plt.figure()
ax = plt.gca()

l_x = np.arange(-100, 100, 1)
l_y = np.maximum(0, l_x)

plt.plot(l_x, l_y)

plt.xticks([-100, 0, 100])
plt.yticks([0, 50, 100])
plt.xlabel("Input")
plt.ylabel("Firing rate")
plt.xlim([-100, 100])
plt.ylim([0, 100])
plt.axvline(x=0, color="k", linestyle="--")
plt.title("F-I curve")
plt.savefig("network_input_output_curve.png")

# STP mechanisms
u_s = 1
U, U_max = 1, 3
tau_u = 40
tau_x = 10

x_pp = 1
u_vs = 1

l_x_pp = []
l_u_vs = []

r_p = 1
r_s = 2

dt = 0.001
T = int(3 / dt)

for t in range(T):
    x_pp = x_pp + ((1 - x_pp) / tau_x - u_s * x_pp * r_p) * dt

    u_vs = u_vs + ((U - u_vs) / tau_u + U * (U_max - u_vs) * r_s) * dt

    l_x_pp.append(x_pp)
    l_u_vs.append(u_vs)

# plotting
plt.figure()
plt.plot(l_x_pp, color="gray")
plt.xticks([])
plt.yticks([])
plt.xlabel("Presynaptic stimulation")
plt.ylabel("Connection strength")
plt.title("STD", color="gray")
plt.savefig("STD.png")

plt.figure()
plt.plot(l_u_vs, color="purple")
plt.xticks([])
plt.yticks([])
plt.xlabel("Presynaptic stimulation")
plt.ylabel("Connection strength")
plt.title("STF", color="purple")
plt.savefig("STF.png")
