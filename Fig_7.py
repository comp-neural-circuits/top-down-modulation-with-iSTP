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
# set the marker size
marker_size = 25

def get_limits(low:float, high:float):
    """ Returns offset limits for axes"""
    offset = (high - low) / 40
    return [low - offset, high + offset]

# simulation setup
dt = 0.0001
T = int(9 / dt)

# neuronal parameters
tau_e, tau_p, tau_s, tau_v = 0.020, 0.010, 0.010, 0.010
alpha_e, alpha_p, alpha_s, alpha_v = 1.0, 1.0, 1.0, 1.0

# network connectivity
Jee = 1.8
Jep = 2.0
Jes = 1.0
Jev = 0

Jpe = 1.4
Jpp = 1.3
Jps = 0.8
Jpv = 0

Jse = 0.9
Jsp = 0
Jss = 0
Jsv = 0.6

Jve = 1.1
Jvp = 0.4
Jvs = 0.4
Jvv = 0

g_1 = 7
g_2 = 5

init_u_s_ee = 0.3
init_tau_x_ee = 0.01

l_alpha = np.arange(0,100.1,1)
c = 0.1

l_SST_change = []

l_R_SS, l_R_SV = [], []
l_eig_E_PV_VIP_STP = []
l_eig_E_PV_VIP_STP_2 = []
l_eig_E_SST_VIP_STP = []
l_eig_E_SST_VIP_STP_2 = []
l_eig_E_E_STP = []
l_eig_E_VIP_STP = []

unstable_modes_E_PV_VIP = []

for alpha in l_alpha:
    # depression variables
    x_ep, x_pp, x_vp, x_ee = 1, 1, 1, 1
    u_s = 1
    tau_x = 0.10
    u_s_ee = init_u_s_ee
    tau_x_ee = init_tau_x_ee

    # facilitation variables
    u_vs = 1
    U, U_max = 1, 3
    tau_u = 0.40

    r_e, r_p, r_s, r_v = 0, 0, 0, 0
    z_e, z_p, z_s, z_v = 0, 0, 0, 0

    l_r_e, l_r_p, l_r_s, l_r_v = [], [], [], []
    
    J_E_PV_VIP = np.zeros((7, 7))
    J_E_SST_VIP = np.zeros((5, 5))
    J_E_E = np.zeros((2, 2))
    J_E_VIP = np.zeros((3, 3))
    
    l_R_SV_temp = []
    
    for i in range(T):
        if 50000 <= i < 70000:
            g_e, g_p, g_s, g_v = g_1 + alpha, g_1 + alpha, g_2, g_1 + c
        else:
            g_e, g_p, g_s, g_v = g_1 + alpha, g_1 + alpha, g_2, g_1

        z_e = x_ee * Jee * r_e - x_ep * Jep * r_p - Jes * r_s - Jev * r_v + g_e
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
        x_ee = x_ee + ((1 - x_ee) / tau_x_ee - u_s_ee * x_ee * r_e) * dt
        x_ee = np.clip(x_ee, 0, 1)

        x_ep = x_ep + ((1 - x_ep) / tau_x - u_s * x_ep * r_p) * dt
        x_ep = np.clip(x_ep, 0, 1)

        x_pp = x_pp + ((1 - x_pp) / tau_x - u_s * x_pp * r_p) * dt
        x_pp = np.clip(x_pp, 0, 1)

        x_vp = x_vp + ((1 - x_vp) / tau_x - u_s * x_vp * r_p) * dt
        x_vp = np.clip(x_vp, 0, 1)

        # STF
        u_vs = u_vs + ((1 - u_vs) / tau_u + U * (U_max - u_vs) * r_s) * dt
        u_vs = np.clip(u_vs, 1, U_max)


            
        if i == 49999:
            # Jacobian of the weight matrix
            J_E_E[0, 0] = (x_ee * Jee - 1) / tau_e
            J_E_E[0, 1] = Jee * r_e / tau_e
            J_E_E[1, 0] = -u_s_ee * x_ee
            J_E_E[1, 1] = -1/tau_x_ee - u_s_ee * r_e

            J_E_VIP[0, 0] = (x_ee * Jee - 1) / tau_e
            J_E_VIP[0, 1] = -Jev / tau_e
            J_E_VIP[0, 2] = Jee * r_e / tau_e
            J_E_VIP[1, 0] = Jve / tau_v
            J_E_VIP[1, 1] = (-Jvv - 1) / tau_v
            J_E_VIP[1, 2] = 0
            J_E_VIP[2, 0] = -u_s_ee * x_ee
            J_E_VIP[2, 1] = 0
            J_E_VIP[2, 2] = -1/tau_x_ee - u_s_ee * r_e

            J_E_PV_VIP[0, 0] = (x_ee * Jee - 1) / tau_e
            J_E_PV_VIP[0, 1] = - x_ep * Jep / tau_e
            J_E_PV_VIP[0, 2] = -Jev / tau_e
            J_E_PV_VIP[0, 3] = Jee * r_e / tau_e
            J_E_PV_VIP[0, 4] = -Jep * r_p / tau_e
            J_E_PV_VIP[0, 5] = 0
            J_E_PV_VIP[0, 6] = 0
            J_E_PV_VIP[1, 0] = Jpe/ tau_p
            J_E_PV_VIP[1, 1] = (-1 - x_pp * Jpp)  / tau_p
            J_E_PV_VIP[1, 2] = -Jpv/ tau_p
            J_E_PV_VIP[1, 3] = 0
            J_E_PV_VIP[1, 4] = 0
            J_E_PV_VIP[1, 5] = -Jpp * r_p/ tau_p
            J_E_PV_VIP[1, 6] = 0
            J_E_PV_VIP[2, 0] = Jve / tau_v
            J_E_PV_VIP[2, 1] = -x_vp * Jvp / tau_v
            J_E_PV_VIP[2, 2] = (-1 - Jvv)  / tau_v
            J_E_PV_VIP[2, 3] = 0
            J_E_PV_VIP[2, 4] = 0
            J_E_PV_VIP[2, 5] = 0
            J_E_PV_VIP[2, 6] = -Jvp * r_p / tau_v
            J_E_PV_VIP[3, 0] = -u_s_ee * x_ee
            J_E_PV_VIP[3, 1] = 0
            J_E_PV_VIP[3, 2] = 0
            J_E_PV_VIP[3, 3] = -1/tau_x_ee - u_s_ee * r_e
            J_E_PV_VIP[3, 4] = 0
            J_E_PV_VIP[3, 5] = 0
            J_E_PV_VIP[3, 6] = 0
            J_E_PV_VIP[4, 0] = 0
            J_E_PV_VIP[4, 1] = -u_s * x_ep
            J_E_PV_VIP[4, 2] = 0
            J_E_PV_VIP[4, 3] = 0
            J_E_PV_VIP[4, 4] = -1/tau_x - u_s * r_p
            J_E_PV_VIP[4, 5] = 0
            J_E_PV_VIP[4, 6] = 0
            J_E_PV_VIP[5, 0] = 0
            J_E_PV_VIP[5, 1] = -u_s * x_pp
            J_E_PV_VIP[5, 2] = 0
            J_E_PV_VIP[5, 3] = 0
            J_E_PV_VIP[5, 4] = 0
            J_E_PV_VIP[5, 5] = -1/tau_x - u_s * r_p
            J_E_PV_VIP[5, 6] = 0
            J_E_PV_VIP[6, 0] = 0
            J_E_PV_VIP[6, 1] = -u_s * x_vp
            J_E_PV_VIP[6, 2] = 0
            J_E_PV_VIP[6, 3] = 0
            J_E_PV_VIP[6, 4] = 0
            J_E_PV_VIP[6, 5] = 0
            J_E_PV_VIP[6, 6] = -1/tau_x - u_s * r_p

            J_E_SST_VIP[0, 0] = (x_ee * Jee - 1) / tau_e
            J_E_SST_VIP[0, 1] = -Jes / tau_e
            J_E_SST_VIP[0, 2] = -Jev / tau_e
            J_E_SST_VIP[0, 3] = Jee * r_e / tau_e
            J_E_SST_VIP[0, 4] = 0
            J_E_SST_VIP[1, 0] = Jse/ tau_s
            J_E_SST_VIP[1, 1] = (-Jss - 1)/tau_s
            J_E_SST_VIP[1, 2] = -Jsv/ tau_s
            J_E_SST_VIP[1, 3] = 0
            J_E_SST_VIP[1, 4] = 0
            J_E_SST_VIP[2, 0] = Jve / tau_v
            J_E_SST_VIP[2, 1] = -u_vs * Jvs / tau_v
            J_E_SST_VIP[2, 2] = (-Jvv - 1) / tau_v
            J_E_SST_VIP[2, 3] = 0
            J_E_SST_VIP[2, 4] = - Jvs * r_s / tau_v
            J_E_SST_VIP[3, 0] = -u_s_ee * x_ee
            J_E_SST_VIP[3, 1] = 0
            J_E_SST_VIP[3, 2] = 0
            J_E_SST_VIP[3, 3] = - 1/tau_x_ee - u_s_ee * r_e
            J_E_SST_VIP[3, 4] = 0
            J_E_SST_VIP[4, 0] = 0
            J_E_SST_VIP[4, 1] = U * (U_max - u_vs)
            J_E_SST_VIP[4, 2] = 0
            J_E_SST_VIP[4, 3] = 0
            J_E_SST_VIP[4, 4] = -1/tau_u - U * r_s
            
            eig_E_E = np.sort(np.real(np.linalg.eig(J_E_E)[0]))
            l_eig_E_E_STP.append(np.max(eig_E_E))
            
            eig_E_VIP = np.sort(np.real(np.linalg.eig(J_E_VIP)[0]))
            l_eig_E_VIP_STP.append(np.max(eig_E_VIP))
                        
            eig_E_PV_VIP = np.real(np.linalg.eig(J_E_PV_VIP)[0])
            eig_E_SST_VIP = np.real(np.linalg.eig(J_E_SST_VIP)[0])
            
            # cound the number of eigenvalues that are positive
            unstable_modes_E_PV_VIP.append(np.sum(eig_E_PV_VIP > 0))
            
            # get the first eigenvalue
            eig_E_PV_VIP_1 = np.sort(eig_E_PV_VIP)[-1]
            eig_E_SST_VIP_1 = np.sort(eig_E_SST_VIP)[-1]
            
            l_eig_E_PV_VIP_STP.append(eig_E_PV_VIP_1)
            l_eig_E_SST_VIP_STP.append(eig_E_SST_VIP_1)

            # get the second eigenvalue
            eig_E_PV_VIP_2 = np.sort(eig_E_PV_VIP)[-2]
            eig_E_SST_VIP_2 = np.sort(eig_E_SST_VIP)[-2]
            
            l_eig_E_PV_VIP_STP_2.append(eig_E_PV_VIP_2)
            l_eig_E_SST_VIP_STP_2.append(eig_E_SST_VIP_2)

            # K-value
            p_pp = 1 / (1+u_s*tau_x *r_p)
            p_pp_prime = - (u_s * tau_x) / np.power(1 + u_s * tau_x * r_p, 2)

            p_ep = 1 / (1+u_s*tau_x *r_p)
            p_ep_prime = - (u_s * tau_x) / np.power(1 + u_s * tau_x * r_p, 2)

            p_vp = 1 / (1+u_s*tau_x *r_p)
            p_vp_prime = - (u_s * tau_x) / np.power(1 + u_s * tau_x * r_p, 2)

            p_ee = 1 / (1+u_s_ee*tau_x_ee *r_e)
            p_ee_prime = - (u_s_ee * tau_x_ee) / np.power(1 + u_s_ee * tau_x_ee * r_e, 2)

            p_vs = (1 + U * U_max * tau_u * r_s) / (1 + U * tau_u * r_s)
            p_vs_prime = (U * (U_max - 1) * tau_u) / np.power(1 + U * tau_u * r_s, 2)

            K_SV = (p_pp + p_pp_prime * r_p) * (p_ee + p_ee_prime * r_e) * Jee * Jpp * Jsv - (p_ep + p_ep_prime * r_p) * Jsv * Jpe * Jep - (p_pp + p_pp_prime * r_p) * Jpp * Jsv + (p_ee + p_ee_prime * r_e) * Jee * Jsv - Jsv
            K_SS = ((p_ep + p_ep_prime * r_p) * Jpe * Jep - (p_pp + p_pp_prime * r_p) * (p_ee + p_ee_prime * r_e) * Jee * Jpp + (p_pp + p_pp_prime * r_p) * Jpp - (p_ee + p_ee_prime * r_e ) * Jee + 1)
            
            mat_R = np.array(
                [[1-(p_ee + p_ee_prime * r_e) * Jee, (p_ep + p_ep_prime * r_p) * Jep, Jes, Jev],
                [-Jpe, 1 + (p_pp + p_pp_prime * r_p) * Jpp, Jps, Jpv],
                [-Jse, Jsp, 1+Jss, Jsv],
                [-Jve, (p_vp + p_vp_prime * r_p) * Jvp, (p_vs + p_vs_prime * r_s) * Jvs, 1 + Jvv]
                ]
            )

            det_mat_R = np.linalg.det(mat_R)
            R_SV = 1/det_mat_R * K_SV * c
            R_SS = 1/det_mat_R * K_SS * c
            l_R_SV.append(R_SV)
            l_R_SS.append(R_SS)
            
        l_r_e.append(r_e)
        l_r_p.append(r_p)
        l_r_s.append(r_s)
        l_r_v.append(r_v)
        
    l_r_e = np.asarray(l_r_e)
    l_r_p = np.asarray(l_r_p)
    l_r_s = np.asarray(l_r_s)
    l_r_v = np.asarray(l_r_v)

    l_SST_change.append(np.mean(l_r_s[60000:65000]) - np.mean(l_r_s[40000:45000]))
    
# calculate the zero crossing of R_SV
l_R_SV_sign = np.sign(l_R_SV)
l_R_SV_sign_change = np.diff(l_R_SV_sign)
l_R_SV_sign_change_idx = np.where(l_R_SV_sign_change != 0)[0]

l_R_SV_sign_change_idx = [3, 75]

# Change in SST activity and R_SV (Fig. 7A)
plt.figure()

plt.plot(l_SST_change)
plt.plot(l_R_SV, '--')

plt.axhline(y=0, color='k', linestyle='--')
for loc_R_SV_change in l_R_SV_sign_change_idx:
    plt.axvline(x=loc_R_SV_change, color='k', linestyle='--')

plt.xticks([0, 20, 40, 60, 80, 100])
plt.xlim(get_limits(0,100))

plt.yticks(np.array([-0.2, -0.1, 0, 0.1, 0.2]))
plt.ylim([-0.2, 0.2])

plt.xlabel(r'$\alpha$')
plt.ylabel('Change in SST activity (a.u.)')

plt.legend(['simulation', 'analytics $\mathbf{R}_{SV}$'], loc='lower right')

plt.savefig('Fig_7A.png')

# Leading eigenvalues of E and E-VIP subnetworks (Fig. 7B)
plt.figure()

l_eig_E_E_STP_sign = np.sign(l_eig_E_E_STP)
l_eig_E_E_STP_sign_change = np.diff(l_eig_E_E_STP_sign)
l_eig_E_E_STP_sign_change_idx = np.where(l_eig_E_E_STP_sign_change != 0)[0]
l_eig_E_E_STP_sign_change_idx = l_eig_E_E_STP_sign_change_idx + 1

plt.plot(l_eig_E_VIP_STP, color='cyan', label='E-VIP subnetwork')
plt.plot(l_eig_E_E_STP, '--', color='purple', label='E subnetwork')
plt.plot(l_eig_E_E_STP_sign_change_idx[0], 0, marker='o', color='purple', markersize=marker_size)

plt.axhline(y=0, color='k', linestyle='--')
for loc_R_SV_change in l_R_SV_sign_change_idx:
    plt.axvline(x=loc_R_SV_change, color='k', linestyle='--')

plt.xticks([0, 20, 40, 60, 80, 100])
plt.xlim(get_limits(0,100))

plt.yticks([-10, 0, 10, 20, 30, 40, 50])
plt.ylim([-10, 50])

plt.xlabel(r'$\alpha$')
plt.ylabel('Leading eigenvalue')

plt.legend(loc='upper right')

plt.savefig('Fig_7B.png')

# R_SS and R_SV (Fig. 7C)
plt.figure()

plt.plot(l_R_SV, color='orange', label='$\mathbf{R}_{SV}$')
plt.plot(l_R_SS, color='green', label='$\mathbf{R}_{SS}$')

# plot the marker
plt.plot(0, l_R_SV[0], marker='^', color='orange', markersize=marker_size)
plt.plot(0, l_R_SS[0], marker='^', color='green', markersize=marker_size)

plt.plot(5, l_R_SV[5], marker='D', color='orange', markersize=marker_size)
plt.plot(5, l_R_SS[5], marker='D', color='green', markersize=marker_size)

plt.plot(15, l_R_SV[15], marker='o', color='orange', markersize=marker_size)
plt.plot(15, l_R_SS[15], marker='o', color='green', markersize=marker_size)

plt.plot(77, l_R_SV[77], marker='*', color='orange', markersize=marker_size)
plt.plot(77, l_R_SS[77], marker='*', color='green', markersize=marker_size)

plt.plot(90, l_R_SV[90], marker='s', color='orange', markersize=marker_size)
plt.plot(90, l_R_SS[90], marker='s', color='green', markersize=marker_size)

plt.axhline(y=0, color='k', linestyle='--')
for loc_R_SV_change in l_R_SV_sign_change_idx:
    plt.axvline(x=loc_R_SV_change, color='k', linestyle='--')

plt.yticks([-0.4, -0.3, -0.2, -0.1, 0, 0.1, 0.2])
plt.xticks([0, 20, 40, 60, 80, 100])
plt.xlabel(r'$\alpha$')
plt.ylabel('$R_{ij}$ magnitude (a.u.)')
plt.ylim([-0.4, 0.2])
plt.xlim(get_limits(0,100))

plt.legend(loc='lower right')

plt.savefig('Fig_7C.png')

# Leading eigenvalues of E-PV-VIP and E-SST-VIP subnetworks (Fig. 7D)
plt.figure()

plt.plot(l_eig_E_PV_VIP_STP, '--', color='blue')
plt.plot(l_eig_E_SST_VIP_STP, '--', color='red')

plt.axhline(y=0, color='k', linestyle='--')
for loc_R_SV_change in l_R_SV_sign_change_idx:
    plt.axvline(x=loc_R_SV_change, color='k', linestyle='--')

plt.xticks([0, 20, 40, 60, 80, 100])
plt.xlim(get_limits(0,100))

plt.yticks([-30, -20, -10, 0, 10, 20])
plt.ylim([-30, 20])

plt.xlabel(r'$\alpha$')
plt.ylabel('Leading eigenvalue')

plt.legend(['E-PV-VIP subnetwork', 'E-SST-VIP subnetwork'], loc='best')

plt.savefig('Fig_7D.png')

# Second largest eigenvalues of E-PV-VIP and E-SST-VIP subnetworks (Fig. 7E)
plt.figure()

plt.plot(l_eig_E_PV_VIP_STP_2, '--', color='blue')
plt.plot(l_eig_E_SST_VIP_STP_2, '--', color='red')

plt.axhline(y=0, color='k', linestyle='--')
for loc_R_SV_change in l_R_SV_sign_change_idx:
    plt.axvline(x=loc_R_SV_change, color='k', linestyle='--')

plt.xticks([0, 20, 40, 60, 80, 100])
plt.xlim(get_limits(0,100))

plt.yticks([-120, -80, -40, 0])
plt.ylim([-120, 10])

plt.xlabel(r'$\alpha$')
plt.ylabel('Leading eigenvalue')

plt.legend(['E-PV-VIP subnetwork', 'E-SST-VIP subnetwork'], loc='best')

plt.savefig('Fig_7E.png')


# Parity of unstable modes (Fig. 7F)
plt.figure()

l_even_odd = np.asarray(unstable_modes_E_PV_VIP) % 2

# get the indices where l_even_odd changes
l_change = np.diff(l_even_odd)
l_change = np.where(l_change != 0)[0]

plt.scatter(l_alpha, l_even_odd, color='grey', s=800)

for change_pos in l_change:
    plt.axvline(x=change_pos + 1, color='k', linestyle='--')

plt.xticks([0, 20, 40, 60, 80, 100])
plt.xlim(get_limits(0,100))

plt.yticks([0,1], ['even', 'odd'])
plt.ylim([-0.25, 1.25])

plt.xlabel(r'$\alpha$')
plt.ylabel('Parity of the number of unstable modes\nin E-PV-VIP subnetwork')

plt.savefig('Fig_7F.png')



