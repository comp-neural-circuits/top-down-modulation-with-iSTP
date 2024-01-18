import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.io as sio

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

def get_limits(low:float, high:float):
    """ Returns offset limits for axes"""
    offset = (high - low) / 40
    return [low - offset, high + offset]

def get_hist_limits(low:float, high:float):
    """ Returns offset limits for axes"""
    offset = (high - low) / 40
    return [low, high + offset]


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

l_alpha = [0, 20]
c = 3

g_1 = 4
g_2 = 3

l_Jep = np.arange(0.9, 1.701, 0.05)
l_Jes = np.arange(0.9, 1.701, 0.05)

m_unstable = np.zeros((len(l_Jep), len(l_Jes)))
m_response_reversal_index = np.zeros((len(l_Jep), len(l_Jes)))
m_eff_ratio = np.zeros((len(l_Jep), len(l_Jes)))

for j, Jep in enumerate(l_Jep):
    for k, Jes in enumerate(l_Jes):
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
                    m_unstable[j,k] = 1
                    r_e = 400
                if r_p > 400:
                    unstable = True
                    m_unstable[j,k] = 1
                    r_p = 400
                if r_s > 400:
                    unstable = True
                    m_unstable[j,k] = 1
                    r_s = 400
                if r_v > 400:
                    unstable = True
                    m_unstable[j,k] = 1
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
                l_change_SST.append(np.mean(l_r_s[60000:65000]) - np.mean(l_r_s[40000:45000]))
            else:
                l_change_SST.append(0)

            if len(l_r_e) > 45000:
                if any([np.mean(l_r_e[40000:45000]) <= 0.1, np.mean(l_r_p[40000:45000]) <= 0.1, np.mean(l_r_v[40000:45000]) <= 0.1, np.mean(l_r_s[40000:45000]) <= 0.1]):
                    unstable = True
                    m_unstable[j,k] = 2


        # compute response reversal index
        if not unstable:
            m_response_reversal_index[j,k] = compute_response_reversal_index(l_change_SST)
        else:
            m_response_reversal_index[j,k] = -10

sio.savemat('m_response_reversal_index_Jep_Jes.mat', mdict={'m_response_reversal_index': m_response_reversal_index})
sio.savemat('m_regime_Jep_Jes.mat', mdict={'m_regime': m_unstable})


###################### Jep / Jpe ######################
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

l_alpha = [0, 20]
c = 3

g_1 = 4
g_2 = 3

l_Jep = np.arange(1.2, 2.001, 0.05)
l_Jpe = np.arange(0.6, 1.401, 0.05)

m_unstable = np.zeros((len(l_Jep), len(l_Jpe)))
m_response_reversal_index = np.zeros((len(l_Jep), len(l_Jpe)))


for j, Jep in enumerate(l_Jep):
    for k, Jpe in enumerate(l_Jpe):
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
                    m_unstable[j,k] = 1
                    r_e = 400
                if r_p > 400:
                    unstable = True
                    m_unstable[j,k] = 1
                    r_p = 400
                if r_s > 400:
                    unstable = True
                    m_unstable[j,k] = 1
                    r_s = 400
                if r_v > 400:
                    unstable = True
                    m_unstable[j,k] = 1
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
                l_change_SST.append(np.mean(l_r_s[60000:65000]) - np.mean(l_r_s[40000:45000]))
            else:
                l_change_SST.append(0)

            if len(l_r_e) > 45000:
                if any([np.mean(l_r_e[40000:45000]) <= 0.1, np.mean(l_r_p[40000:45000]) <= 0.1, np.mean(l_r_v[40000:45000]) <= 0.1, np.mean(l_r_s[40000:45000]) <= 0.1]):
                    unstable = True
                    m_unstable[j,k] = 2


        # compute response reversal index
        if not unstable:
            m_response_reversal_index[j,k] = compute_response_reversal_index(l_change_SST)
        else:
            m_response_reversal_index[j,k] = -10

sio.savemat('m_response_reversal_index_Jep_Jpe.mat', mdict={'m_response_reversal_index': m_response_reversal_index})
sio.savemat('m_regime_Jep_Jpe.mat', mdict={'m_regime': m_unstable})

###################### Jes / Jse ######################

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

l_alpha = [0, 20]
c = 3

g_1 = 4
g_2 = 3

l_Jes = np.arange(0.9, 1.101, 0.0125)
l_Jse = np.arange(0.7, 0.9001, 0.0125)

m_unstable_Jes_Jse = np.zeros((len(l_Jes), len(l_Jse)))
m_response_reversal_index_Jes_Jse = np.zeros((len(l_Jes), len(l_Jse)))

for j, Jes in enumerate(l_Jes):
    for k, Jse in enumerate(l_Jse):
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
                    m_unstable_Jes_Jse[j,k] = 1
                    r_e = 400
                if r_p > 400:
                    unstable = True
                    m_unstable_Jes_Jse[j,k] = 1
                    r_p = 400
                if r_s > 400:
                    unstable = True
                    m_unstable_Jes_Jse[j,k] = 1
                    r_s = 400
                if r_v > 400:
                    unstable = True
                    m_unstable_Jes_Jse[j,k] = 1
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
                l_change_SST.append(np.mean(l_r_s[60000:65000]) - np.mean(l_r_s[40000:45000]))
            else:
                l_change_SST.append(0)

            if len(l_r_e) > 45000:
                if any([np.mean(l_r_e[40000:45000]) <= 0.1, np.mean(l_r_p[40000:45000]) <= 0.1, np.mean(l_r_v[40000:45000]) <= 0.1, np.mean(l_r_s[40000:45000]) <= 0.1]):
                    unstable = True
                    m_unstable_Jes_Jse[j,k] = 2

        # compute response reversal index
        if not unstable:
            m_response_reversal_index_Jes_Jse[j,k] = compute_response_reversal_index(l_change_SST)
        else:
            m_response_reversal_index_Jes_Jse[j,k] = -10
            
sio.savemat('m_response_reversal_index_Jes_Jse.mat', mdict={'m_response_reversal_index': m_response_reversal_index_Jes_Jse})
sio.savemat('m_regime_Jes_Jse.mat', mdict={'m_regime': m_unstable_Jes_Jse})

# ###################### Jpe / Jse ######################

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

l_alpha = [0, 20]
c = 3

g_1 = 4
g_2 = 3

l_Jse = np.arange(0.7, 1.001, 0.02)
l_Jpe = np.arange(0.7, 1.001, 0.02)

m_unstable = np.zeros((len(l_Jse), len(l_Jpe)))
m_response_reversal_index = np.zeros((len(l_Jse), len(l_Jpe)))

for j, Jse in enumerate(l_Jse):
    for k, Jpe in enumerate(l_Jpe):
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
                    m_unstable[j,k] = 1
                    r_e = 400
                if r_p > 400:
                    unstable = True
                    m_unstable[j,k] = 1
                    r_p = 400
                if r_s > 400:
                    unstable = True
                    m_unstable[j,k] = 1
                    r_s = 400
                if r_v > 400:
                    unstable = True
                    m_unstable[j,k] = 1
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
                l_change_SST.append(np.mean(l_r_s[60000:65000]) - np.mean(l_r_s[40000:45000]))
            else:
                l_change_SST.append(0)

            if len(l_r_e) > 45000:
                if any([np.mean(l_r_e[40000:45000]) <= 0.1, np.mean(l_r_p[40000:45000]) <= 0.1, np.mean(l_r_v[40000:45000]) <= 0.1, np.mean(l_r_s[40000:45000]) <= 0.1]):
                    unstable = True
                    m_unstable[j,k] = 2


        # compute response reversal index
        if not unstable:
            m_response_reversal_index[j,k] = compute_response_reversal_index(l_change_SST)
        else:
            m_response_reversal_index[j,k] = -10
            
sio.savemat('m_response_reversal_index_Jse_Jpe.mat', mdict={'m_response_reversal_index': m_response_reversal_index})
sio.savemat('m_regime_Jse_Jpe.mat', mdict={'m_regime': m_unstable})


l_file_name = ['_Jep_Jes', '_Jep_Jpe', '_Jes_Jse', '_Jse_Jpe']
sections = ['D', 'F', 'G', 'H']

for file_name, section in zip(l_file_name, sections):

    m_response_reversal_index = sio.loadmat('m_response_reversal_index' + file_name + '.mat')['m_response_reversal_index']
    m_regime = sio.loadmat('m_regime' + file_name + '.mat')['m_regime']

    plt.figure()
    ax = plt.gca()
    g = sns.heatmap(m_response_reversal_index, cmap='bwr', vmin=-10, vmax=10, linewidths=3, linecolor='black')

    indices_one_dead = m_regime == 2

    for i,j in zip(*np.where(indices_one_dead)):
        g.add_patch(plt.Rectangle((j, i), 1, 1, facecolor='grey', edgecolor='black', linewidth=3))

    g.set_facecolor('gray')
    ax.axhline(y=0, color='k',linewidth=10)
    ax.axhline(y=m_response_reversal_index.shape[1], color='k',linewidth=12)
    ax.axvline(x=0, color='k',linewidth=10)
    ax.axvline(x=m_response_reversal_index.shape[0], color='k',linewidth=12)
    g.set_xticklabels(g.get_xticklabels(), rotation=0)
    g.set_yticklabels(g.get_yticklabels(), rotation=0)

    plt.xticks([])
    plt.yticks([])

    if file_name == '_Jep_Jpe':
        plt.xlabel(r"$J_{PE}$")
        plt.ylabel(r"$J_{EP}$")
        plt.xticks(np.arange(0.5, 16 + 1, 8), [0.6, 1.0, 1.4])
        plt.yticks(np.arange(0.5, 16 + 1, 8), [1.2, 1.6, 2.0])
        plt.xlim([0, 17])
        plt.ylim([0, 17])
    elif file_name == '_Jep_Jes':
        plt.xlabel(r"$J_{ES}$")
        plt.ylabel(r"$J_{EP}$")
        plt.xticks(np.arange(0.5, 16 + 1, 8), [0.9, 1.3, 1.7])
        plt.yticks(np.arange(0.5, 16 + 1, 8), [0.9, 1.3, 1.7])
        plt.xlim([0, 17])
        plt.ylim([0, 17])
    elif file_name == '_Jes_Jse':
        plt.xlabel(r"$J_{ES}$")
        plt.ylabel(r"$J_{SE}$")
        plt.xticks(np.arange(0.5, 16 + 1, 8), [0.9, 1.0, 1.1])
        plt.yticks(np.arange(0.5, 16 + 1, 8), [0.7, 0.8, 0.9])
        plt.xlim([0, 17])
        plt.ylim([0, 17])
    elif file_name == '_Jse_Jpe':
        plt.xlabel(r"$J_{PE}$")
        plt.ylabel(r"$J_{SE}$")
        plt.xticks(np.arange(0.5, 15 + 1, 7.5), [0.7, 0.85, 1.0])
        plt.yticks(np.arange(0.5, 15 + 1, 7.5), [0.7, 0.85, 1.0])
        plt.xlim([0, 16])
        plt.ylim([0, 16])

        ax2 = plt.twinx()
        sns.lineplot(data=np.arange(0, m_response_reversal_index.shape[1] + 1, 1), color='black', ax=ax2)
        plt.yticks([])
        plt.ylim([0, m_response_reversal_index.shape[1]])
        ax2.lines[0].set_linestyle("--")
    else:
         pass

    cax = plt.gcf().axes[-1]
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(axis='both', which='both', length=0)
    cbar.set_ticks([-10, 0, 10])

    plt.savefig(f'Fig_S12{section}.png')
    plt.savefig(f'Fig_S12{section}.pdf')