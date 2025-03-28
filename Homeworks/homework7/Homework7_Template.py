# -*- coding: utf-8 -*-
"""
Created on Fri Mar 28 13:02:28 2025

@author: harni
"""

# import necessary modules
# numpy provides powerful multi-dimensional arrays to hold and manipulate data
import numpy as np
# matplotlib provides powerful functions for plotting figures
import matplotlib.pyplot as plt
# astropy provides unit system and constants for astronomical calculations
import astropy.units as u
import astropy.constants as const
# import Latex module so we can display the results with symbols
from IPython.display import Latex


# **** import CenterOfMass to determine the COM pos/vel of M33
import CenterOfMass2 as COM


# **** import the GalaxyMass to determine the mass of M31 for each component
import GalaxyMass as GM

class M33AnalyticOrbit:
    """Calculate the analytical orbit of M33 around M31."""
    
    def __init__(self, filename):
        """
        Initialize the orbit integrator.

        Parameters:
        -----------
        filename : str
            Name of the file where the integrated orbit will be saved.
        """
        # Get the gravitational constant in units of kpc^3/Msun/Gyr^2
        self.G = const.G.to(u.kpc**3 / u.Msun / u.Gyr**2).value
        
        # Store the output file name (using a consistent attribute name)
        self.fileout = filename

        # Use raw strings for Windows file paths to avoid escape sequence issues
        M33_file = r"C:\Users\harni\OneDrive - University of Arizona\Astro400B\Astro400BClone\Homeworks\homework7"
        M31_file = r"C:\Users\harni\OneDrive - University of Arizona\Astro400B\Astro400BClone\Homeworks\homework7"
        
        # Get the current position/velocity of M33 
        M33COM = COM.CenterOfMass(M33_file, 2)
        self.PM33 = M33COM.COM_P(0.1, 4).value  # position vector (without units)
        self.VM33 = M33COM.COM_V(self.PM33[0]*u.kpc,
                                  self.PM33[1]*u.kpc,
                                  self.PM33[2]*u.kpc).value  # velocity vector (without units)
        
        # Get the current position/velocity of M31 
        M31COM = COM.CenterOfMass(M31_file, 2)
        self.PM31 = M31COM.COM_P(0.1, 2).value  # position vector (without units)
        self.VM31 = M31COM.COM_V(self.PM31[0]*u.kpc,
                                  self.PM31[1]*u.kpc,
                                  self.PM31[2]*u.kpc).value  # velocity vector (without units)
        
        # Store the relative position and velocity vectors (M33 relative to M31)
        self.r = self.PM33 - self.PM31 
        self.v = self.VM33 - self.VM31
        
        # Get the mass components of M31
        # Disk
        self.rdisk = 5
        self.Mdisk = GM.ComponentMass(M31_file, 2) * 1e12
        # Bulge
        self.rbulge = 1
        self.Mbulge = GM.ComponentMass(M31_file, 3) * 1e12 
        # Halo
        self.rhalo = 60
        self.Mhalo = GM.ComponentMass(M31_file, 1) * 1e12
    
    def HernquistAccel(self, M, r_a, r):
        """
        Calculates the acceleration for the halo or bulge using the Hernquist profile.

        Parameters:
        -----------
        M : float
            Mass of the halo or bulge.
        r_a : float
            Scale length of the halo or bulge.
        r : array_like
            Position vector.

        Returns:
        --------
        ndarray
            Acceleration vector.
        """
        rmag = np.sqrt(np.sum(r**2))
        Hern = -(self.G * M) * r / (rmag * (r_a + rmag)**2)
        return Hern
    
    def MiyamotoNagaiAccel(self, M, r_d, r):
        """
        Calculates the disk acceleration using the Miyamoto-Nagai profile.

        Parameters:
        -----------
        M : float
            Mass of the disk.
        r_d : float
            Scale length of the disk.
        r : array_like
            Position vector.

        Returns:
        --------
        ndarray
            Acceleration vector.
        """
        # Calculate the cylindrical radius in the x-y plane
        R = np.sqrt(r[0]**2 + r[1]**2)
        z_d = r_d / 5.0
        B = r_d + np.sqrt(r[2]**2 + z_d**2)
        # Compute the acceleration vector; note that for the z-component we could include a different scaling.
        MNa = -(self.G * M) * r / ((R**2 + B**2)**(3/2))
        return MNa
    
    def M31Accel(self, r):
        """
        Sums the acceleration contributions from the halo, bulge, and disk.

        Parameters:
        -----------
        r : array_like
            Position vector in 3D space (x, y, z).

        Returns:
        --------
        ndarray
            Total acceleration vector from M31.
        """
        Halo_acc = self.HernquistAccel(self.Mhalo, self.rhalo, r)
        Bulge_acc = self.HernquistAccel(self.Mbulge, self.rbulge, r)
        Disk_acc = self.MiyamotoNagaiAccel(self.Mdisk, self.rdisk, r)
        total_acc = np.sum([Halo_acc, Bulge_acc, Disk_acc], axis=0)
        return total_acc
    
    def LeapFrog(self, dt, r, v):
        """
        Advances the orbit by one time step using the LeapFrog integrator.

        Parameters:
        -----------
        dt : float
            Time step.
        r : array_like
            Current position vector.
        v : array_like
            Current velocity vector.

        Returns:
        --------
        rnew : ndarray
            Updated position vector.
        vnew : ndarray
            Updated velocity vector.
        """
        # Predict the position at the half timestep
        rhalf = r + v * dt / 2
        a = self.M31Accel(rhalf)
        # Update the velocity using the acceleration at rhalf
        vnew = v + a * dt
        # Update the position using the new velocity
        rnew = rhalf + vnew * dt / 2
        return rnew, vnew
    
    def OrbitIntegration(self, t0, dt, tmax):
        """
        Integrates the orbit using the LeapFrog scheme.

        Parameters:
        -----------
        t0 : float
            Initial time (Gyr).
        dt : float
            Time step (Gyr).
        tmax : float
            Maximum time (Gyr).

        Returns:
        --------
        orbit : ndarray
            Array containing the time, position, and velocity at each time step.
            Each row is [t, x, y, z, vx, vy, vz].
        """
        t = t0
        n_steps = int(tmax / dt) + 2
        orbit = np.zeros((n_steps, 7))
        # Set initial conditions
        orbit[0] = t0, *tuple(self.r), *tuple(self.v)
        i = 1
        
        # Integration loop using LeapFrog
        while t < tmax:
            t += dt
            orbit[i, 0] = t
            r_old = orbit[i-1, 1:4]
            v_old = orbit[i-1, 4:7]
            rnew, vnew = self.LeapFrog(dt, r_old, v_old)
            orbit[i, 1:4] = rnew
            orbit[i, 4:7] = vnew
            i += 1
        
        # Build the full file path using os.path.join to ensure proper path separation
        directory = r"C:\Users\harni\OneDrive - University of Arizona\Astro400B\Astro400BClone\Homeworks\homework7"
        filepath = os.path.join(directory, self.fileout)
        
        # Save the orbit data to the file
        np.savetxt("/Users/harni/OneDrive - University of Arizona/Astro400B/Astro400BClone/Homeworks/homework7/", orbit, fmt="%11.3f"*7, comments='#',
                   header="{:>10s}{:>11s}{:>11s}{:>11s}{:>11s}{:>11s}{:>11s}"
                          .format('t', 'x', 'y', 'z', 'vx', 'vy', 'vz'))

M33_orbit = M33AnalyticOrbit("M33_predicted_orbit.txt")

M33_orbit.OrbitIntegration(0, 0.1, 10)
M33_pred_orbit = np.genfromtxt("M33_predicted_orbit.txt", comments='#', names=True)
mag_pos = np.sqrt(M33_pred_orbit['x']**2 + M33_pred_orbit['y']**2 + M33_pred_orbit['z']**2)
mag_vel = np.sqrt(M33_pred_orbit['vx']**2 + M33_pred_orbit['vy']**2 + M33_pred_orbit['vz']**2)

M31 = np.genfromtxt("/Users/swapnaneeldey/Desktop/ASTR400Bfiles/ASTR400B/Homeworks/Homework6/Orbit_M31.txt", comments='#', names=True)
M33 = np.genfromtxt("/Users/swapnaneeldey/Desktop/ASTR400Bfiles/ASTR400B/Homeworks/Homework6/Orbit_M33.txt", comments='#', names=True)

# function to compute the magnitude of the difference between two vectors 
# You can use this function to return both the relative position and relative velocity for two 
# galaxies over the entire orbit  
def relative_mag(galaxy_1, galaxy_2):
    """Calculate the relative position and relative velocity of the galaxies
    
    Inputs:
    galaxy_1: the name of the first galaxy
    galaxy_2: the name of the second galaxy     
    
    Returns:
    relative_position: the relative position of the two galaxies
    relative_velocity: the relative velocity of the two galaxies
    """
    # Calculate the relative position and relative velocity of the two galaxies
    relative_position = np.sqrt((galaxy_1["x"] - galaxy_2["x"])**2 + (galaxy_1["y"] - galaxy_2["y"])**2 + (galaxy_1["z"] - galaxy_2["z"])**2)
    relative_velocity = np.sqrt((galaxy_1["vx"] - galaxy_2["vx"])**2 + (galaxy_1["vy"] - galaxy_2["vy"])**2 + (galaxy_1["vz"] - galaxy_2["vz"])**2)

    return relative_position, relative_velocity

    
# of M33 and M31
rel_pos_M33_M31, rel_vel_M33_M31 = relative_mag(M33, M31)

plt.rcParams.update({
    "font.family": "DejaVu Serif",   # specify font family here
    "font.serif": ["cm"],
    "mathtext.fontset" : "dejavuserif" ,
    "font.size":17,
    "axes.titlesize" : 20,
    "axes.labelsize" : 20,
    "axes.linewidth" : 1.5,
    "lines.linewidth" : 2.0,
    "xtick.labelsize" :15,
    "ytick.labelsize" : 15,
    "xtick.major.size" : 15,
    "xtick.minor.size" : 0,
    "ytick.major.size" : 15,
    "ytick.minor.size" : 0,
    "xtick.major.width" : 2,
    "xtick.minor.width" : 2,
    "ytick.major.width" : 2,
    "ytick.minor.width" : 2})  
plt.rcParams["legend.frameon"] = True

#plotting relative position

fig= plt.figure(figsize=(11,8))
plt.plot(M33_pred_orbit['t'], mag_pos, label = 'M33-M31 predicted', color = 'blue')
plt.plot(M31["t"], rel_pos_M33_M31, label='M33-M31 simulation', color='r')
plt.legend(loc='upper right', fontsize=15)
plt.xlabel('Time (Gyr)', fontsize=20)
plt.ylabel('Relative Position (kpc)', fontsize=20)


#plotting relative velocities
fig= plt.figure(figsize=(11,8))
plt.plot(M33_pred_orbit['t'], mag_vel, label = 'M33-M31 predicted', color = 'blue')
plt.plot(M31["t"], rel_vel_M33_M31, label='M33-M31 simulation', color='r')
plt.legend(loc='upper right', fontsize=15)
plt.xlabel('Time (Gyr)', fontsize=20)
plt.ylabel('Relative velocity (km/s)', fontsize=20)
