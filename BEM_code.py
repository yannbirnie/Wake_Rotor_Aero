import numpy as np
import scipy.integrate as spig
import scipy.interpolate as spip

relaxation = 0.25
rho = 1.225


class Airfoils:
    def __init__(self):
        # list of thicknesses of the given airfoils
        thickness_list = (241, 301, 360, 480, 600, 1000)

        # Initialise data lists
        data_raw = list()
        cl_list = list()
        cd_list = list()
        cm_list = list()

        # Iterate through the list of thicknesses
        for i, t in enumerate(thickness_list):
            # Read the corresponding airfoil data file
            file = open(f"airfoil_data/FFA-W3-{t}.txt")
            file_lines = file.readlines()
            file.close()
            # Format the data to usable format
            data_raw = [line.strip("\n").split("\t") for line in file_lines]

            # Isolate the values for lift, drag and moment at different aoa's
            # Add each list of values to the correct 2d array
            cl_list.append([float(line[1]) for line in data_raw])
            cd_list.append([float(line[2]) for line in data_raw])
            cm_list.append([float(line[3]) for line in data_raw])

        # List all the aoa's, assuming all airfoil data has the same amount of points
        aoa_list = (*(float(line[0]) for line in data_raw),)

        # Make the lists numpy arrays for use in interpolation
        # These are 2d arrays with the variation with thickness in axis 0 and variation in angle of attack in axis 1
        self.cl_list = np.array(cl_list)
        self.cd_list = np.array(cd_list)
        self.cm_list = np.array(cm_list)

        # The tuple with the data points for the interpolation function
        self.points = (thickness_list, aoa_list)

    # Function to get the lift coefficient interpolated withing the range of thickness and aoa
    def cl(self, thickness, alpha):
        return spip.interpn(self.points, self.cl_list, (thickness, alpha), method="linear")[0]

    # Function to get the drag coefficient interpolated withing the range of thickness and aoa
    def cd(self, thickness, alpha):
        return spip.interpn(self.points, self.cd_list, (thickness, alpha), method="linear")[0]

    # Function to get the moment coefficient interpolated withing the range of thickness and aoa
    def cm(self, thickness, alpha):
        return spip.interpn(self.points, self.cm_list, (thickness, alpha), method="linear")[0]


class BladeElement:
    def __init__(self, pos_r: float, chord: float, relative_pitch: float, thickness: float, airfoil):
        self.r = pos_r
        self.c = chord
        self.beta = relative_pitch
        self.tc = 10 * thickness

        self.a = None
        self.a_prime = None
        self.p_n = None
        self.p_t = None

        self.airfoil = airfoil()

    def __repr__(self):
        return f"<Blade Element at r={self.r}, c={self.c}, beta={self.beta}, t/c={self.tc}>"

    def determine_loads(self, v_0, omega, theta_p, b, r_blade):
        # Set initial loop values
        self.a = 0
        self.a_prime = 0
        error_a = 1
        error_a_dash = 1
        i = 0
        # Iterative solver for a and a_prime until the difference between the iterations becomes very small
        while True:
            # For the previous a and a_prime, find the flow angle and angle of attack
            phi = np.arctan2(((1 - self.a) * v_0), ((1 + self.a_prime) * omega * self.r))
            alpha = np.degrees(phi) - self.beta - theta_p

            # With the angle of attack, determine the lift and drag coefficient from airfoil data interpolation
            cl = self.airfoil.cl(self.tc, alpha)
            cd = self.airfoil.cd(self.tc, alpha)

            # Use these to find the normal and tangential force coefficients
            cn = cl * np.cos(phi) + cd * np.sin(phi)
            ct = cl * np.sin(phi) - cd * np.cos(phi)

            # Break conditions for the a-loop
            if error_a <= 10 ** (-6) and error_a_dash <= 10 ** (-6):
                break
            elif i > 10**3:
                raise ValueError(f"Solution for a and a' not converging. a={self.a}, a' = {self.a_prime}.")

            # Determine the solidity and Prandtl’s tip loss correction
            solidity = self.c * b / (2 * np.pi * self.r)
            f = (2/np.pi) * np.arccos(np.exp(-(b * (r_blade - self.r) / (2 * self.r * np.sin(abs(phi))))))

            # Determine the new a and a_prime
            if self.a >= 0.33:
                c_thrust = ((1 - self.a) ** 2 * cn * solidity) / (np.sin(phi) ** 2)

                a_star = c_thrust / (4 * f * (1 - 0.25*(5 - 3 * self.a) * self.a))
                a_new = relaxation * a_star + (1-relaxation) * self.a

            else:
                a_new = 1 / ((4 * f * np.sin(phi)**2) / (solidity * cn) + 1)

            a_prime_new = 1 / ((4 * f * np.sin(phi) * np.cos(phi)) / (solidity * ct) - 1)

            # Determine the difference between this and the previous iteration
            error_a = abs(a_new - self.a)
            error_a_dash = abs(a_prime_new - self.a_prime)

            # Get ready for the next iteration
            self.a = a_new
            self.a_prime = a_prime_new
            i += 1

        # Determine the relative velocity with the velocity triangle
        v_n = (1 + self.a_prime) * omega * self.r
        v_t = (1 - self.a) * v_0
        v_rel = np.sqrt(v_n**2 + v_t**2)

        # Using the previous calculations, find the forces on the blade element
        self.p_n = 0.5 * rho * v_rel ** 2 * self.c * cn
        self.p_t = 0.5 * rho * v_rel ** 2 * self.c * ct

    def get_loads(self):
        if self.p_t is None or self.p_n is None:
            raise ValueError(f"Loads have not been determined. Run .determine_loads() first.")
        else:
            return self.p_n, self.p_t


class Blade:
    def __init__(self, no_blades, path, airfoils):
        self.b = no_blades

        self.power = None
        self.thrust = None
        self.c_power = None
        self.c_thrust = None

        self.r_list = []
        self.p_n_list = None
        self.p_t_list = None

        # Load the blade data from the file
        file = open(path)
        file_lines = file.readlines()
        file.close()

        data_raw = [0, 0, 0, 0]
        self.blade_elements = {}
        for line in file_lines:
            data_raw = [float(value) for value in line.strip("\n").split("\t")]
            self.blade_elements[data_raw[0]] = BladeElement(data_raw[0], data_raw[2], data_raw[1], data_raw[3],
                                                            airfoils)
            self.r_list.append(data_raw[0])

        self.r_list = np.array(self.r_list)
        self.r = data_raw[0]

    def find_pn_pt(self, v_0, theta_p, omega):
        # Initialise the lists for p_n and p_t
        p_n_list, p_t_list = list(), list()
        for r, blade in self.blade_elements.items():
            if r < self.r:
                blade.determine_loads(v_0, omega, theta_p, self.b, self.r)
                p_n, p_t = blade.get_loads()

                p_n_list.append(p_n)
                p_t_list.append(p_t)

        # Add zero load at the blade tip
        p_n_list.append(0)
        p_t_list.append(0)

        return np.array(p_n_list), np.array(p_t_list), self.r_list

    def determine_cp_ct(self, v_0, lamda, theta_p):
        # Determine the rotational speed of the turbine
        omega = lamda * v_0 / self.r
        # Get the loads on the blade elements
        p_n_list, p_t_list, r_list = self.find_pn_pt(v_0, theta_p, omega)

        # Determine the thrust and power of the turbine
        self.thrust = self.b * spig.trapz(p_n_list, self.r_list)
        self.power = omega * self.b * spig.trapz(p_t_list * self.r_list, self.r_list)

        # Determine the thrust and power coefficient
        self.c_thrust = self.thrust / (0.5 * rho * np.pi * self.r**2 * v_0**2)
        self.c_power = self.power / (0.5 * rho * np.pi * self.r**2 * v_0**3)


def interpolate(value1, value2, co1, co2, co_interpolation):
    dy_dx = (value2 - value1) / (co2 - co1)
    return dy_dx * (co_interpolation - co1) + value1
