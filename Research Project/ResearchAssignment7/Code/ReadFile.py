# -*- coding: utf-8 -*-
"""
Created on Sun Jan 26 21:31:34 2025

@author: harni
"""


import numpy as np
import astropy.units as u

# Defines the function
def Read(MW_801):
    """
    Function to read data from the MW_000.txt file.

    Parameters:
    filename : str
        The name of the file to read.

    Returns:
    time : astropy.units.Quantity
        Time in Myr.
    total_particles : int
        Total number of particles.
    data : 
        Array containing particle type, mass, and coordinates (x, y, z, vx, vy, vz).
    """
    # Opens the file
    file = open(MW_801, 'r')

    # Reads the first line to get the time
    line1 = file.readline()
    label, value = line1.split()
    time = float(value) * u.Myr  # Convert time to Myr using Astropy units

    # Reads the second line to get the total number of particles
    line2 = file.readline()
    label, value = line2.split()
    total_particles = int(value)

    # Closes the file
    file.close()

    # Reads the remaining data into a structured array
    data = np.genfromtxt(MW_801, dtype=None, names=True, skip_header=3)


   #returns the data; time, total particles, and the rest of the data
    return time, total_particles, data


def Read(M31_801):
    """
    Function to read data from the MW_000.txt file.

    Parameters:
    filename : str
        The name of the file to read.

    Returns:
    time : astropy.units.Quantity
        Time in Myr.
    total_particles : int
        Total number of particles.
    data : 
        Array containing particle type, mass, and coordinates (x, y, z, vx, vy, vz).
    """
    # Opens the file
    file = open(M31_801, 'r')

    # Reads the first line to get the time
    line1 = file.readline()
    label, value = line1.split()
    time = float(value) * u.Myr  # Convert time to Myr using Astropy units

    # Reads the second line to get the total number of particles
    line2 = file.readline()
    label, value = line2.split()
    total_particles = int(value)

    # Closes the file
    file.close()

    # Reads the remaining data into a structured array
    data = np.genfromtxt(M31_801, dtype=None, names=True, skip_header=3)


   #returns the data; time, total particles, and the rest of the data
    return time, total_particles, data
    