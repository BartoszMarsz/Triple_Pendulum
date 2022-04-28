import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
print('Put constant values')
m1 = float(input('m1 = '))
m2 = float(input('m2 = '))
m3 = float(input('m3 = '))
l1 = float(input('l1 = '))
l2 = float(input('l2 = '))
l3 = float(input('l3 = '))
g = 9.81

print('Put initial values in degrees and rad/s')
th1 = np.radians(float(input('theta_1 = ')))
th2 = np.radians(float(input('theta_2 = ')))
th3 = float(input('theta_3 = '))
om1 = float(input('omega_1 = '))
om2 = float(input('omega_2 = '))
om3 = float(input('omega_3 = '))

h = 0.01
t = 10

TH = np.array([[th1], [th2], [th3]])
OM = np.array([[om1], [om2], [om3]])


def F(TH, OM):
    th1 = TH[0][0]
    th2 = TH[1][0]
    th3 = TH[2][0]
    om1 = OM[0][0]
    om2 = OM[1][0]
    om3 = OM[2][0]

    a11 = (m1 + m2 + m3) * l1
    a12 = (m2 + m3) * l2 * np.cos(th2 - th1)
    a13 = m3 * l3 * np.cos(th3 - th1)

    A1 = np.array([a11, a12, a13])

    a21 = (m2 + m3) * l1 * np.cos(th1 - th2)
    a22 = (m2 + m3) * l2
    a23 = m3 * l3 * np.cos(th3 - th2)

    A2 = np.array([a21, a22, a23])

    a31 = l1 * np.cos(th1 - th3)
    a32 = l2 * np.cos(th2 - th3)
    a33 = l3

    A3 = np.array([a31, a32, a33])

    A = np.array([A1, A2, A3])

    b1 = (m2 + m3) * l2 * om2 ** 2 * np.sin(th2 - th1) + m3 * l3 * om3 ** 2 * np.sin(th3 - th1) - (
            m1 + m2 + m3) * g * np.sin(th1)
    b2 = (m2 + m3) * l1 * om1 ** 2 * np.sin(th1 - th2) + m3 * l3 * om3 ** 2 * np.sin(th3 - th2) - (
            m2 + m3) * g * np.sin(th2)
    b3 = l1 * om1 ** 2 * np.sin(th1 - th3) + l2 * om2 ** 2 * np.sin(th2 - th3) - g * np.sin(th3)

    B = np.array([[b1], [b2], [b3]])

    InvA = np.linalg.inv(A)

    return np.matmul(InvA, B)


# F = [[f1()],
#      [f2()],
#      [f3()]]

def runge_kutta4(TH, OM):
    K1TH = OM
    K1OM = F(TH, OM)

    K2TH = OM + 0.5 * h * K1TH
    K2OM = F(TH + 0.5 * h * K1TH, OM + 0.5 * h * K1OM)

    K3TH = OM + 0.5 * h * K2TH
    K3OM = F(TH + 0.5 * h * K2TH, OM + 0.5 * h * K2OM)

    K4TH = OM + h * K2TH
    K4OM = F(TH + h * K3TH, OM + h * K3OM)

    TH = TH + (1 / 6) * h * (K1TH + 2 * K2TH + 2 * K3TH + K4TH)
    OM = OM + (1 / 6) * h * (K1OM + 2 * K2OM + 2 * K3OM + K4OM)
    return TH, OM


def runge_kutta2(TH, OM):
    K1TH = OM
    K1OM = F(TH, OM)

    K2TH = OM + h * K1TH
    K2OM = F(TH + h * K1TH, OM + h * K1OM)

    TH = TH + 0.5 * h * (K1TH + K2TH)
    OM = OM + 0.5 * h * (K1OM + K2OM)
    return TH, OM


DATA = open('trajectory4.pdb', 'w')
for i in range(0, int(t / h)):
    var = np.concatenate((TH, OM), axis=0)
    DATA.write(str(round(i * h, 2)))
    for j in var:
        DATA.write(';' + str(round(*j, 4)))
    DATA.write('\n')
    TH, OM = runge_kutta4(TH, OM)
DATA.close()

TH[0][0] = th1
TH[1][0] = th2
TH[2][0] = th3
OM[0][0] = om1
OM[1][0] = om2
OM[2][0] = om3

DATA = open('trajectory2.pdb', 'w')
for i in range(0, int(t / h)):
    var = np.concatenate((TH, OM), axis=0)
    DATA.write(str(round(i * h, 2)))
    for j in var:
        DATA.write(';' + str(round(*j, 4)))
    DATA.write('\n')
    TH, OM = runge_kutta2(TH, OM)
DATA.close()

data = np.loadtxt('trajectory4.pdb', delimiter=';')
time = np.array(data[:, 0])
th1_4 = np.array(data[:, 1])
th2_4 = np.array(data[:, 2])
th3_4 = np.array(data[:, 3])
om1_4 = np.array(data[:, 4])
om2_4 = np.array(data[:, 5])
om3_4 = np.array(data[:, 6])

data = np.loadtxt('trajectory2.pdb', delimiter=';')
th1_2 = np.array(data[:, 1])
th2_2 = np.array(data[:, 2])
th3_2 = np.array(data[:, 3])
om1_2 = np.array(data[:, 4])
om2_2 = np.array(data[:, 5])
om3_2 = np.array(data[:, 6])

X1_4 = l1 * np.sin(th1_4)
Y1_4 = -l1 * np.cos(th1_4)
X2_4 = l1 * np.sin(th1_4) + l2 * np.sin(th2_4)
Y2_4 = -l1 * np.cos(th1_4) - l2 * np.cos(th2_4)
X3_4 = l1 * np.sin(th1_4) + l2 * np.sin(th2_4) + l3 * np.sin(th3_4)
Y3_4 = -l1 * np.cos(th1_4) - l2 * np.cos(th2_4) - l3 * np.cos(th3_4)
X1_2 = l1 * np.sin(th1_2)
Y1_2 = -l1 * np.cos(th1_2)
X2_2 = l1 * np.sin(th1_2) + l2 * np.sin(th2_2)
Y2_2 = -l1 * np.cos(th1_2) - l2 * np.cos(th2_2)
X3_2 = l1 * np.sin(th1_2) + l2 * np.sin(th2_2) + l3 * np.sin(th3_2)
Y3_2 = -l1 * np.cos(th1_2) - l2 * np.cos(th2_2) - l3 * np.cos(th3_2)


def animation_4():
    # defining figure and plots
    fig = plt.figure(figsize=(15, 8))
    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()
    ax1 = fig.add_subplot(2, 2, 1)
    ax2 = fig.add_subplot(2, 2, 3)
    ax3 = fig.add_subplot(2, 2, (2, 4))
    ax3.set_title('Motion of triple pendulum', fontsize=20)
    ax1.set_title(r'Graph of $\theta_1(t)$, $\theta_2(t)$ and $\theta_3(t)$', fontsize=20)
    ax2.set_title(r'Graph of $\omega_1(t)$, $\omega_2(t)$ and $\omega_3(t)$', fontsize=20)
    # defining axises
    th_max = max(10, np.amax(np.concatenate((th1_4, th2_4, th3_4, th1_2, th2_2, th3_2))))
    th_min = min(-10, np.amin(np.concatenate((th1_4, th2_4, th3_4, th1_2, th2_2, th3_2))))
    om_max = max(10, np.amax(np.concatenate((om1_4, om2_4, om3_4, om1_2, om2_2, om3_2))))
    om_min = min(-10, np.amin(np.concatenate((om1_4, om2_4, om3_4, om1_2, om2_2, om3_2))))

    ax1.set_xlim(0, t)
    ax1.set_ylim(th_min - 0.5, th_max + 0.5)
    ax2.set_xlim(0, t)
    ax2.set_ylim(om_min - 0.5, om_max + 0.5)
    ax3.set_xlim(-1.1 * (l1 + l2 + l3), 1.1 * (l1 + l2 + l3))
    ax3.set_ylim(-1.1 * (l1 + l2 + l3), 1.1 * (l1 + l2 + l3))

    ax1.set_ylabel('[rad]', fontsize=15)
    ax2.set_ylabel(r'[$\frac{rad}{s}$]', fontsize=15)
    ax2.set_xlabel('t[s]', fontsize=15)
    ax3.set_xlabel('[m]', fontsize=15)
    ax3.set_ylabel('[m]', fontsize=15)

    # specifying appearance
    ax1.grid(color='dimgrey')
    ax1.set_facecolor(color='black')
    ax2.grid(color='dimgrey')
    ax2.set_facecolor(color='black')
    ax3.grid(color='dimgrey')
    ax3.set_facecolor(color='black')

    ratio = 1.0
    x_left, x_right = ax3.get_xlim()
    y_low, y_high = ax3.get_ylim()
    ax3.set_aspect(abs((x_right - x_left) / (y_low - y_high)) * ratio)

    # defining objects

    # theta and omega (graphs)
    TH1, = ax1.plot(0, 0)
    TH2, = ax1.plot(0, 0)
    TH3, = ax1.plot(0, 0)
    OM1, = ax2.plot(0, 0)
    OM2, = ax2.plot(0, 0)
    OM3, = ax2.plot(0, 0)
    # pendulum (motion)
    trajectory3_4, = ax3.plot(0, 0)
    line1_4, = ax3.plot(0, 0)
    line2_4, = ax3.plot(0, 0)
    line3_4, = ax3.plot(0, 0)
    # timer
    timer = ax3.text(0.05, 0.95, '', transform=ax3.transAxes, verticalalignment='top', color='black', fontsize=20,
                     bbox=(dict(facecolor='wheat', boxstyle='round')))

    # specifying objects appearance
    TH1.set_color('darkorange')
    TH2.set_color('mediumblue')
    TH3.set_color('green')
    OM1.set_color('darkorange')
    OM2.set_color('mediumblue')
    OM3.set_color('green')

    trajectory3_4.set_color('lightgrey')
    trajectory3_4.set_alpha(0.4)

    line1_4.set_color('darkorange')
    line2_4.set_color('mediumblue')
    line3_4.set_color('green')
    line1_4.set_linewidth(2)
    line2_4.set_linewidth(2)
    line3_4.set_linewidth(2)

    # defining legends
    ax1.legend((TH1, TH2, TH3), (r'$\theta_1$', r'$\theta_2$', r'$\theta_3$'), loc='upper right', shadow=True,
               labelcolor='white',
               facecolor='black', fontsize=20)
    ax2.legend((OM1, OM2, OM3), (r'$\omega_1$', r'$\omega_2$', r'$\omega_3$'), loc='upper right', shadow=True,
               labelcolor='white',
               facecolor='black', fontsize=20)

    def animation_frame(i):
        TH1.set_xdata(time[:i])
        TH1.set_ydata(th1_4[:i])
        TH2.set_xdata(time[:i])
        TH2.set_ydata(th2_4[:i])
        TH3.set_xdata(time[:i])
        TH3.set_ydata(th3_4[:i])
        OM1.set_xdata(time[:i])
        OM1.set_ydata(om1_4[:i])
        OM2.set_xdata(time[:i])
        OM2.set_ydata(om2_4[:i])
        OM3.set_xdata(time[:i])
        OM3.set_ydata(om3_4[:i])

        line1_4.set_xdata([0.0, X1_4[i]])
        line1_4.set_ydata([0.0, Y1_4[i]])
        line2_4.set_xdata([X1_4[i], X2_4[i]])
        line2_4.set_ydata([Y1_4[i], Y2_4[i]])
        line3_4.set_xdata([X2_4[i], X3_4[i]])
        line3_4.set_ydata([Y2_4[i], Y3_4[i]])
        trajectory3_4.set_xdata(X3_4[:i])
        trajectory3_4.set_ydata(Y3_4[:i])

        timer.set_text(str(time[i]))

        return TH1, TH2, TH3, OM1, OM2, OM3, line1_4, line2_4, line3_4, trajectory3_4, timer,

    anim = FuncAnimation(fig, func=animation_frame, frames=range(1, int(t / h + 1), 2), interval=1, repeat=False,
                         blit=True)
    plt.show()


def animation_4_2():
    # defining figure and plots
    fig = plt.figure(figsize=(15, 8))
    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()
    ax = fig.add_subplot()
    ax.set_title('Motion of triple pendulum', fontsize=20)
    # defining axis
    ax.set_xlim(-1.1 * (l1 + l2 + l3), 1.1 * (l1 + l2 + l3))
    ax.set_ylim(-1.1 * (l1 + l2 + l3), 1.1 * (l1 + l2 + l3))

    ax.set_xlabel('[m]', fontsize=15)
    ax.set_ylabel('[m]', fontsize=15)

    # specifying appearance
    ax.grid(color='dimgrey')
    ax.set_facecolor(color='black')

    ratio = 1.0
    x_left, x_right = ax.get_xlim()
    y_low, y_high = ax.get_ylim()
    ax.set_aspect(abs((x_right - x_left) / (y_low - y_high)) * ratio)

    # defining objects

    trajectory3_2, = ax.plot(0, 0)
    trajectory3_4, = ax.plot(0, 0)

    line1_2, = ax.plot(0, 0)
    line2_2, = ax.plot(0, 0)
    line3_2, = ax.plot(0, 0)
    line1_4, = ax.plot(0, 0)
    line2_4, = ax.plot(0, 0)
    line3_4, = ax.plot(0, 0)

    # timer
    timer = ax.text(0.05, 0.95, '', transform=ax.transAxes, verticalalignment='top', color='black', fontsize=20,
                    bbox=(dict(facecolor='wheat', boxstyle='round')))

    # specifying objects appearance
    trajectory3_4.set_color('#FFBE32')
    trajectory3_4.set_alpha(0.4)
    trajectory3_2.set_color('#9696FF')
    trajectory3_2.set_alpha(0.4)

    line1_4.set_color('darkorange')
    line2_4.set_color('darkorange')
    line3_4.set_color('darkorange')
    line1_4.set_linewidth(2)
    line2_4.set_linewidth(2)
    line3_4.set_linewidth(2)
    line1_2.set_color('mediumblue')
    line2_2.set_color('mediumblue')
    line3_2.set_color('mediumblue')
    line1_2.set_linewidth(2)
    line2_2.set_linewidth(2)
    line3_2.set_linewidth(2)

    # defining legend
    ax.legend((line1_4, line1_2), ('Runge Kutta 4th order', 'Runge Kutta 2th order'), loc='upper right', shadow=True,
              labelcolor='white',
              facecolor='black', fontsize=20)

    def animation_frame(i):
        line1_2.set_xdata([0.0, X1_2[i]])
        line1_2.set_ydata([0.0, Y1_2[i]])
        line2_2.set_xdata([X1_2[i], X2_2[i]])
        line2_2.set_ydata([Y1_2[i], Y2_2[i]])
        line3_2.set_xdata([X2_2[i], X3_2[i]])
        line3_2.set_ydata([Y2_2[i], Y3_2[i]])
        line1_4.set_xdata([0.0, X1_4[i]])
        line1_4.set_ydata([0.0, Y1_4[i]])
        line2_4.set_xdata([X1_4[i], X2_4[i]])
        line2_4.set_ydata([Y1_4[i], Y2_4[i]])
        line3_4.set_xdata([X2_4[i], X3_4[i]])
        line3_4.set_ydata([Y2_4[i], Y3_4[i]])
        trajectory3_2.set_xdata(X3_2[:i])
        trajectory3_2.set_ydata(Y3_2[:i])
        trajectory3_4.set_xdata(X3_4[:i])
        trajectory3_4.set_ydata(Y3_4[:i])

        timer.set_text(str(time[i]))

        return line1_4, line2_4, line3_4, line1_2, line2_2, line3_2, trajectory3_4, trajectory3_2, timer,

    anim = FuncAnimation(fig, func=animation_frame, frames=range(1, int(t / h + 1), 2), interval=1, repeat=False,
                         blit=True)
    plt.show()


def graph_TH_4():
    fig = plt.figure(figsize=(15, 8))
    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()
    ax = fig.add_subplot(1, 1, 1)

    ax.set_title(r'Graph of $\theta_1(t)$, $\theta_2(t)$ and $\theta_3(t)$ for Runge Kutta 4th order', fontsize=30)

    th_max = max(10, np.amax(np.concatenate((th1_4, th2_4, th3_4))))
    th_min = min(-10, np.amin(np.concatenate((th1_4, th2_4, th3_4))))

    ax.set_xlim(0, t)
    ax.set_ylim(th_min - 0.5, th_max + 0.5)

    ax.tick_params(axis='both', which='major', labelsize=25)

    ax.set_ylabel('[rad]', fontsize=25)
    ax.set_xlabel('t[s]', fontsize=25)

    ax.grid(color='dimgrey')
    ax.set_facecolor(color='white')

    TH1, = ax.plot(0, 0)
    TH2, = ax.plot(0, 0)
    TH3, = ax.plot(0, 0)

    TH1.set_color('darkorange')
    TH2.set_color('mediumblue')
    TH3.set_color('green')

    ax.legend((TH1, TH2, TH3), (r'$\theta_1$', r'$\theta_2$', r'$\theta_3$'), loc='upper right', shadow=True,
              labelcolor='black',
              facecolor='white', fontsize=25)

    TH1.set_xdata(time)
    TH1.set_ydata(th1_4)
    TH2.set_xdata(time)
    TH2.set_ydata(th2_4)
    TH3.set_xdata(time)
    TH3.set_ydata(th3_4)

    plt.show()


def graph_OM_4():
    fig = plt.figure(figsize=(15, 8))
    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()
    ax = fig.add_subplot(1, 1, 1)

    ax.set_title(r'Graph of $\omega_1(t)$, $\omega_2(t)$ and $\omega_3(t)$ for Runge Kutta 4th order', fontsize=30)

    om_max = max(10, np.amax(np.concatenate((om1_4, om2_4, om3_4))))
    om_min = min(-10, np.amin(np.concatenate((om1_4, om2_4, om3_4))))

    ax.set_xlim(0, t)
    ax.set_ylim(om_min - 0.5, om_max + 0.5)

    ax.tick_params(axis='both', which='major', labelsize=25)

    ax.set_ylabel(r'$[\frac{rad}{s}]$', fontsize=25)
    ax.set_xlabel('t[s]', fontsize=25)

    ax.grid(color='dimgrey')
    ax.set_facecolor(color='white')

    OM1, = ax.plot(0, 0)
    OM2, = ax.plot(0, 0)
    OM3, = ax.plot(0, 0)

    OM1.set_color('darkorange')
    OM2.set_color('mediumblue')
    OM3.set_color('green')

    ax.legend((OM1, OM2, OM3), (r'$\omega_1$', r'$\omega_2$', r'$\omega_3$'), loc='upper right', shadow=True,
              labelcolor='black',
              facecolor='white', fontsize=25)

    OM1.set_xdata(time)
    OM1.set_ydata(om1_4)
    OM2.set_xdata(time)
    OM2.set_ydata(om2_4)
    OM3.set_xdata(time)
    OM3.set_ydata(om3_4)

    plt.show()


def graph_traj_4():
    fig = plt.figure(figsize=(15, 8))
    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()
    ax = fig.add_subplot(1, 1, 1)

    ax.set_title('Trajectory for Runge Kutta 4th order', fontsize=30)

    ax.set_xlim(-1.1 * (l1 + l2 + l3), 1.1 * (l1 + l2 + l3))
    ax.set_ylim(-1.1 * (l1 + l2 + l3), 1.1 * (l1 + l2 + l3))

    ax.set_xlabel('[m]', fontsize=25)
    ax.set_ylabel('[m]', fontsize=25)

    ax.tick_params(axis='both', which='major', labelsize=25)

    ax.grid(color='dimgrey')
    ax.set_facecolor(color='white')

    ratio = 1.0
    x_left, x_right = ax.get_xlim()
    y_low, y_high = ax.get_ylim()
    ax.set_aspect(abs((x_right - x_left) / (y_low - y_high)) * ratio)

    trajectory_4, = ax.plot(0, 0)
    line_4, = ax.plot(0, 0)

    line_4.set_color('black')
    trajectory_4.set_color('orange')

    line_4.set_linewidth(2)

    line_4.set_xdata([0.0, X1_4[int(t / h - 1)], X2_4[int(t / h - 1)], X3_4[int(t / h - 1)]])
    line_4.set_ydata([0.0, Y1_4[int(t / h - 1)], Y2_4[int(t / h - 1)], Y3_4[int(t / h - 1)]])
    trajectory_4.set_xdata(X3_4)
    trajectory_4.set_ydata(Y3_4)

    plt.show()


def graph_th3_4_2():
    fig = plt.figure(figsize=(15, 8))
    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()
    ax = fig.add_subplot(1, 1, 1)

    ax.set_title(r'Graph of $\theta_3(t)$ for Runge Kutta 4th and 2nd order', fontsize=30)

    th_max = max(10, np.amax(np.concatenate((th3_4, th3_2))))
    th_min = min(-10, np.amin(np.concatenate((th3_4, th3_2))))

    ax.set_xlim(0, t)
    ax.set_ylim(th_min - 0.5, th_max + 0.5)

    ax.tick_params(axis='both', which='major', labelsize=25)

    ax.set_ylabel('[rad]', fontsize=25)
    ax.set_xlabel('t[s]', fontsize=25)

    ax.grid(color='dimgrey')
    ax.set_facecolor(color='white')

    TH3_4, = ax.plot(0, 0)
    TH3_2, = ax.plot(0, 0)

    TH3_4.set_color('darkorange')
    TH3_2.set_color('green')

    ax.legend((TH3_4, TH3_2), ('4th order', '2nd order'), loc='upper right', shadow=True, labelcolor='black',
              facecolor='white', fontsize=25)

    TH3_4.set_xdata(time)
    TH3_4.set_ydata(th3_4)
    TH3_2.set_xdata(time)
    TH3_2.set_ydata(th3_2)

    plt.show()


def traj_4_2():
    # defining figure and plots
    fig = plt.figure(figsize=(15, 8))
    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()
    ax = fig.add_subplot()
    ax.set_title('Trajectory for Runge-Kutta 4th and 2nd order', fontsize=30)
    # defining axis
    ax.set_xlim(-1.1 * (l1 + l2 + l3), 1.1 * (l1 + l2 + l3))
    ax.set_ylim(-1.1 * (l1 + l2 + l3), 1.1 * (l1 + l2 + l3))

    ax.set_xlabel('[m]', fontsize=25)
    ax.set_ylabel('[m]', fontsize=25)

    ax.tick_params(axis='both', which='major', labelsize=25)

    # specifying appearance
    ax.grid(color='dimgrey')
    ax.set_facecolor(color='white')

    ratio = 1.0
    x_left, x_right = ax.get_xlim()
    y_low, y_high = ax.get_ylim()
    ax.set_aspect(abs((x_right - x_left) / (y_low - y_high)) * ratio)

    # defining objects

    trajectory_2, = ax.plot(0, 0)
    trajectory_4, = ax.plot(0, 0)

    line_2, = ax.plot(0, 0)
    line_4, = ax.plot(0, 0)

    # specifying objects appearance
    trajectory_4.set_color('#FFBE32')
    trajectory_2.set_color('#00FF00')

    line_4.set_color('darkorange')
    line_4.set_linewidth(2)

    line_2.set_color('green')
    line_2.set_linewidth(2)

    # defining legend
    ax.legend((line_4, line_2), ('4th order', '2th order'), loc='upper right', shadow=True, labelcolor='black',
              facecolor='white', fontsize=20)

    line_4.set_xdata([0.0, X1_4[int(t / h - 1)], X2_4[int(t / h - 1)], X3_4[int(t / h - 1)]])
    line_4.set_ydata([0.0, Y1_4[int(t / h - 1)], Y2_4[int(t / h - 1)], Y3_4[int(t / h - 1)]])
    trajectory_4.set_xdata(X3_4)
    trajectory_4.set_ydata(Y3_4)
    line_2.set_xdata([0.0, X1_2[int(t / h - 1)], X2_2[int(t / h - 1)], X3_2[int(t / h - 1)]])
    line_2.set_ydata([0.0, Y1_2[int(t / h - 1)], Y2_2[int(t / h - 1)], Y3_2[int(t / h - 1)]])
    trajectory_2.set_xdata(X3_2)
    trajectory_2.set_ydata(Y3_2)

    plt.show()


animation_4()
animation_4_2()
graph_TH_4()
graph_OM_4()
graph_traj_4()
graph_th3_4_2()
traj_4_2()
