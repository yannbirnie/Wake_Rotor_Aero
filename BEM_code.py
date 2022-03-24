import numpy as np
import scipy.integrate as spig
import matplotlib.pyplot as plt


relaxation = 0.1
rho = 1.225


class DU95W150:
    def __init__(self):
        data = read_from_file('DU95W150.csv')
        self.alpha_lst = data[:, 0]
        self.cl_lst = data[:, 1]
        self.cd_lst = data[:, 2]
        self.cm_lst = data[:, 3]

    def cl(self, alpha): return np.interp(alpha, self.alpha_lst, self.cl_lst)

    def cd(self, alpha): return np.interp(alpha, self.alpha_lst, self.cd_lst)

    def cm(self, alpha): return np.interp(alpha, self.alpha_lst, self.cm_lst)


class BladeElement:
    def __init__(self, pos_r: float, chord: float, relative_pitch: float, airfoil):
        self.r = pos_r
        self.c = chord
        self.beta = relative_pitch

        self.a = None
        self.a_prime = None
        self.axial_induction = None
        self.azimuthal_induction = None
        self.phi = None
        self.alpha = None
        self.p_n = None
        self.p_t = None
        self.u_tangential = None
        self.u_normal = None

        self.af = airfoil
        self.airfoil = airfoil()

    def __repr__(self):
        return f"<Blade Element at r={self.r}, c={self.c}, beta={self.beta}>"

    def determine_loads(self, v_0, omega, theta_p, b, r_blade, yaw, azimuth):
        yaw = np.radians(yaw)
        azimuth = np.radians(azimuth)
        # Set initial loop values
        self.a = 0
        self.a_prime = 0
        error_a = 1
        error_a_dash = 1
        i = 0
        # Iterative solver for a and a_prime until the difference between the iterations becomes very small
        while True:
            self.u_tangential = omega * self.r * (1 + self.a_prime)
            self.u_normal = v_0 * (1 - self.a)

            # For the previous a and a_prime, find the flow angle and angle of attack
            self.phi = np.arctan2(self.u_normal, self.u_tangential)
            self.alpha = np.degrees(self.phi) - self.beta - theta_p

            # With the angle of attack, determine the lift and drag coefficient from airfoil data interpolation
            cl = self.airfoil.cl(self.alpha)
            cd = self.airfoil.cd(self.alpha)

            # Use these to find the normal and tangential force coefficients
            cn = cl * np.cos(self.phi) + cd * np.sin(self.phi)
            ct = cl * np.sin(self.phi) - cd * np.cos(self.phi)

            # Break conditions for the a-loop
            if error_a <= 1e-6 and error_a_dash <= 1e-6:
                break
            elif i > 1e3:
                raise ValueError(f"r={self.r}: Solution for a and a' not converging. a={self.a}, a' = {self.a_prime}.")

            # Determine the solidity and Prandtl’s tip loss correction
            solidity = self.c * b / (2 * np.pi * self.r)
            f = (2/np.pi) * np.arccos(np.exp(-(b * (r_blade - self.r) / (2 * self.r * np.sin(abs(self.phi))))))

            # Determine the new a and a_prime
            if self.a >= 0.33:
                c_thrust = ((1 - self.a) ** 2 * cn * solidity) / (np.sin(self.phi) ** 2)

                a_star = c_thrust / (4 * f * (1 - 0.25*(5 - 3 * self.a) * self.a))
                a_new = relaxation * a_star + (1-relaxation) * self.a

            else:
                a_new = 1 / ((4 * f * np.sin(self.phi)**2) / (solidity * cn) + 1)

            a_prime_new = 1 / ((4 * f * np.sin(self.phi) * np.cos(self.phi)) / (solidity * ct) - 1)

            # Determine the difference between this and the previous iteration
            error_a = abs(a_new - self.a)
            error_a_dash = abs(a_prime_new - self.a_prime)

            # Get ready for the next iteration
            self.a = a_new
            self.a_prime = a_prime_new
            i += 1

        x = xi(self.a, yaw)
        K_xi = 2 * np.tan(x / 2)
        self.axial_induction = self.a * (1 + K_xi * self.r * np.sin(azimuth - np.pi / 2) / r_blade)
        self.azimuthal_induction = self.a_prime

        self.u_tangential = (omega * self.r - v_0 * np.sin(yaw) * np.sin(azimuth)) * (1 + self.a_prime)
        self.u_normal = v_0 * (np.cos(yaw) - self.axial_induction)

        # For the previous a and a_prime, find the flow angle and angle of attack
        self.phi = np.arctan2(self.u_normal, self.u_tangential)
        self.alpha = np.degrees(self.phi) - self.beta - theta_p

        # With the angle of attack, determine the lift and drag coefficient from airfoil data interpolation
        cl = self.airfoil.cl(self.alpha)
        cd = self.airfoil.cd(self.alpha)

        # Use these to find the normal and tangential force coefficients
        cn = cl * np.cos(self.phi) + cd * np.sin(self.phi)
        ct = cl * np.sin(self.phi) - cd * np.cos(self.phi)

        # Determine the relative velocity with the velocity triangle
        v_rel = np.sqrt(self.u_normal**2 + self.u_tangential**2)

        # Using the previous calculations, find the forces on the blade element
        self.p_n = 0.5 * rho * v_rel ** 2 * self.c * cn
        self.p_t = 0.5 * rho * v_rel ** 2 * self.c * ct

    def get_loads(self):
        if self.p_t is None or self.p_n is None:
            raise ValueError(f"Loads have not been determined. Run .determine_loads() first.")
        else:
            return self.p_n, self.p_t

    def reset(self):
        self.__init__(self.r, self.c, self.beta, self.af)


class Blade:
    def __init__(self, no_blades, airfoil, r_start, r_end, blade_pitch, n_elements):
        self.b = no_blades

        self.power = None
        self.thrust = None
        self.c_power = None
        self.c_thrust = None

        self.r_list = []
        self.p_n_list = None
        self.p_t_list = None

        self.blade_elements = list()
        # Divide the blade up in n_elements pieces;
        for i in range(n_elements + 1):
            r = r_start + (r_end - r_start)/n_elements * i
            self.r_list.append(r)
            # Sorry for hardcoding the equations below- taken from the assignment description :)
            twist = 14*(1-r/r_end)
            chord = (3*(1-r/r_end)+1)

            # BladeElement takes in argument relative_pitch, I assume that this means total? So offset with the blade pitch
            relative_pitch = blade_pitch + twist

            self.blade_elements.append(BladeElement(r, chord, relative_pitch, airfoil))

        self.r_list = np.array(self.r_list)
        self.r = r_end

    def find_pn_pt(self, v_0, theta_p, omega, yaw, azimuth):
        # Initialise the lists for p_n and p_t
        p_n_list, p_t_list = list(), list()
        for blade in self.blade_elements:
            if blade.r < self.r:
                blade.determine_loads(v_0, omega, theta_p, self.b, self.r, yaw, azimuth)
                p_n, p_t = blade.get_loads()

                p_n_list.append(p_n)
                p_t_list.append(p_t)

        # Add zero load at the blade tip
        p_n_list.append(0)
        p_t_list.append(0)

        return np.array(p_n_list), np.array(p_t_list), self.r_list

    def determine_cp_ct(self, v_0, lamda, theta_p, yaw, azimuth):
        # Determine the rotational speed of the turbine
        omega = lamda * v_0 / self.r
        # Get the loads on the blade elements
        self.p_n_list, self.p_t_list, r_list = self.find_pn_pt(v_0, theta_p, omega, yaw, azimuth)

        # Determine the thrust and power of the turbine
        self.thrust = self.b * spig.trapz(self.p_n_list, self.r_list)
        self.power = omega * self.b * spig.trapz(self.p_t_list * self.r_list, self.r_list)

        # Determine the thrust and power coefficient
        self.c_thrust = self.thrust / (0.5 * rho * np.pi * self.r**2 * v_0**2)
        self.c_power = self.power / (0.5 * rho * np.pi * self.r**2 * v_0**3)

    def reset(self):
        for be in self.blade_elements:
            be.reset()


class Turbine:
    def __init__(self):
        self.blade = Blade(3, DU95W150, .2 * 50, 50, -2, 80)

    def cp_lamda(self):
        tsr = np.round(np.arange(4, 12.1, 0.1), 1)
        cp = np.zeros(tsr.shape)
        for i, lamda in enumerate(tsr):
            self.blade.determine_cp_ct(10, lamda, 0, 0, 0)
            cp[i] = self.blade.c_power

            if lamda in (6, 8, 10):
                plt.plot(lamda, self.blade.c_power, 'k^')

            self.blade.reset()

        plt.xlabel("$\\lambda\\ [-]$")
        plt.ylabel("$C_P\\ [-]$")
        plt.tight_layout()
        plt.plot(tsr, cp, 'k')
        plt.show()

    def spanwise_distributions(self):
        self.blade.determine_cp_ct(10, 8, 0, 0, 0)
        pn, pt = self.blade.p_n_list, self.blade.p_t_list
        alpha, phi, a, a_prime, twist = np.zeros((5, len(self.blade.blade_elements)))
        for i, be in enumerate(self.blade.blade_elements):
            alpha[i] = be.alpha
            phi[i] = be.phi
            a[i] = be.a
            a_prime[i] = be.a_prime
            twist[i] = be.beta

        plt.figure(1)
        plt.plot(self.blade.r_list, alpha, label='Angle of Attack ($\\alpha$)')
        plt.plot(self.blade.r_list, np.degrees(phi), label='Inflow Angle ($\\phi$)')
        plt.plot(self.blade.r_list, twist, label='Twist Angle ($\\beta$)')
        plt.xlabel('$r$ [m]')
        plt.ylabel('$Angle$ [$^{\\circ}$]')
        plt.grid()
        plt.legend()

        plt.figure(2)
        plt.plot(self.blade.r_list, a, label='Axial Induction ($a$)')
        plt.plot(self.blade.r_list, a_prime, label="Azimuthal Induction ($a'$)")
        plt.xlabel('$r$ [m]')
        plt.ylabel('$Induction\\ factor$ [-]')
        plt.yticks(np.arange(0, 0.6, 0.1))
        plt.grid()
        plt.legend()

        plt.figure(3)
        plt.plot(self.blade.r_list, pn, label='Thrust Loading ($p_n$)')
        plt.plot(self.blade.r_list, pt, label="Azimuthal Loading ($p_t$)")
        plt.xlabel('$r$ [m]')
        plt.ylabel('$p$ [N/m]')
        plt.grid()
        plt.legend()

        plt.show()

    def yaw_polar_plots(self):
        yaw = (15, 30,)
        azimuth = np.arange(0, 360 + 1, 1)
        r_grid, az_grid = np.meshgrid(self.blade.r_list[:-1], np.radians(azimuth))

        for i, y in enumerate(yaw):
            fig1, (ax1, ax2) = plt.subplots(1, 2, num=1, subplot_kw=dict(projection='polar'), figsize=(9, 4.5))
            fig2, (ax3, ax4) = plt.subplots(1, 2, num=2, subplot_kw=dict(projection='polar'), figsize=(9, 4.5))
            fig3, (ax5, ax6) = plt.subplots(1, 2, num=3, subplot_kw=dict(projection='polar'), figsize=(9, 4.5))
            fig4, (ax7, ax8) = plt.subplots(1, 2, num=4, subplot_kw=dict(projection='polar'), figsize=(9, 4.5))

            alpha, phi, a, a_prime, pn, pt, un, ut = np.zeros((8, azimuth.size, len(self.blade.blade_elements[:-1])))
            for j, az in enumerate(azimuth):
                # print(az)
                self.blade.determine_cp_ct(10, 8, 0, y, az)

                pn[j, :], pt[j, :] = self.blade.p_n_list[:-1], self.blade.p_t_list[:-1]
                for k, be in enumerate(self.blade.blade_elements[:-1]):
                    alpha[j, k] = be.alpha
                    phi[j, k] = be.phi
                    a[j, k] = be.axial_induction
                    a_prime[j, k] = be.azimuthal_induction
                    un[j, k] = be.u_normal
                    ut[j, k] = be.u_tangential

                self.blade.reset()

            color = 'k'
            cmap = 'hot'
            linewidth = .5

            contour_plot(az_grid, r_grid, alpha, ax1, fig1,
                         (cmap, color, linewidth, 'Angle of Attack $\\alpha$ [$^{\\circ}$]'))
            contour_plot(az_grid, r_grid, np.degrees(phi), ax2, fig1,
                         (cmap, color, linewidth, 'Inflow Angle $\\phi$ [$^{\\circ}$]'))

            contour_plot(az_grid, r_grid, a, ax3, fig2,
                         (cmap, color, linewidth, 'Axial Induction $a_{total}$ [-]'))
            contour_plot(az_grid, r_grid, a_prime, ax4, fig2,
                         (cmap, color, linewidth, "Azimuthal Induction $a'$ [-]"))

            contour_plot(az_grid, r_grid, pn, ax5, fig3,
                         (cmap, color, linewidth, 'Normal Force $p_n$ [N/m]'))
            contour_plot(az_grid, r_grid, pt, ax6, fig3,
                         (cmap, color, linewidth, 'Tangential Force $p_t$ [N/m]'))

            contour_plot(az_grid, r_grid, un, ax7, fig4,
                         (cmap, color, linewidth, 'Normal Velocity $u_n$ [m/s]'))
            contour_plot(az_grid, r_grid, ut, ax8, fig4,
                         (cmap, color, linewidth, 'Tangential Velocity $u_t$ [m/s]'))

            fig1.tight_layout()
            fig2.tight_layout()
            fig3.tight_layout()
            fig4.tight_layout()
            plt.show()


def contour_plot(az_grid, r_grid, values, axis, figure, options: tuple):
    def contour_levels():
        base_levels = np.linspace(0, 1, 1000)

        val_min = np.min(values)
        val_max = np.max(values)

        return base_levels * (val_max - val_min) + val_min

    cmap, color, linewidth, label, *_ = options

    axis.set_title(label)
    ctr = axis.contourf(az_grid, r_grid, values, contour_levels(), cmap=cmap)
    ctr1 = axis.contour(az_grid, r_grid, values, colors=color, linewidths=linewidth)
    axis.set_yticks(np.arange(0, 50 + 10, 10))
    cbar = figure.colorbar(ctr, ax=axis, ticks=ctr1.levels)


def xi(a, yaw):
    return (0.6 * a + 1) * yaw
#     val = yaw.copy()
#     diff = 1
#     c = 0
#     relax = 0.1
#     while diff > 1e-3 and c < 1e2:
#         # new = np.arctan2(2 * np.tan(val / 2), 1 - np.tan(val / 2)**2)
#         new = relax * np.arctan2(np.sin(yaw) - a * np.tan(val/2), np.cos(yaw) - a) + (1 - relax) * val
#         diff = abs(new - val)
#         val = new
#         c += 1
#
#     # print(c)
#     if c < 1e3:
#         return val
#     else:
#         print(f'Not converged for yaw={yaw}, a={a}.')
#         return yaw


def interpolate(value1, value2, co1, co2, co_interpolation):
    dy_dx = (value2 - value1) / (co2 - co1)
    return dy_dx * (co_interpolation - co1) + value1


def read_from_file(path):
    f = open(path)
    lines = f.readlines()
    out_list = [[float(num) for num in line.strip('\n').split(',')] for line in lines]
    return np.array(out_list)


if __name__ == '__main__':
    turbine = Turbine()
    # turbine.cp_lamda()
    # turbine.spanwise_distributions()
    turbine.yaw_polar_plots()

    # a = .82
    # yaw = np.radians(30)
    # print(np.degrees((.6 * a + 1) * yaw))
    # print(np.degrees(xi(a, yaw)))
