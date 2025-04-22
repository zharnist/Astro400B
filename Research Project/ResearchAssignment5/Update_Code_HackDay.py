# -*- coding: utf-8 -*-
"""
Created on Mon Apr 21 21:46:08 2025

@author: harni

GOAL:
---------------------
This Python script analyzes the final state of a simulated dark matter halo formed from the merger of the Milky Way and M31 galaxies.
It does the following:
1. Loads in two simulation snapshots (MW and M31).
2. Merges only the dark matter particles.
3. Calculates the center of mass (COM) of the system.
4. Computes radial distances and radial velocities of each particle from the COM.
5. Bins the particles into radial shells to compute a mass density profile.
6. Determines the R200 radius, where the average density = 200x the critical density of the universe.
7. Fits Hernquist and NFW models to the density profile inside R200.
8. Plots the R200 determination curve and the density and radial velocity profiles with fitted models.

Used code from Homework 2,4,6 and Lab2
Referenced Swapnaneel R200 code to help implement in mine

"""

# Import required scientific libraries
import numpy as np  # Numerical computing
import matplotlib.pyplot as plt  # Plotting library
from astropy import units as u  # Astrophysical unit handling
from scipy.optimize import curve_fit  # Function fitting tool

# Import custom modules
from ReadFile import Read  # Reads simulation snapshot data
from CenterOfMass2 import CenterOfMass  # Computes COM for particle data

# Define Hernquist density profile model
# Inputs: r (radius), rho0 (density scale), a (scale radius)
def hernquist_density(r, rho0, a):
    return rho0 / ((r / a) *(1 + r/a)**3)  # Return Hernquist profile evaluated at r

# Define NFW (Navarro-Frenk-White) density profile model
# Inputs: r (radius), rho0 (density scale), rs (scale radius)
def nfw_density(r, rho0, rs):
    return rho0/((r / rs)*(1 + r/rs)**2)  # Return NFW profile evaluated at r

# Merge two dark matter snapshots
# Inputs: file1, file2 - snapshot file paths; ptype = 1 for dark matter particles
# Output: merged dictionary with positions, velocities, and masses

def merge_snapshots(file1, file2, ptype=1):
    _, _, data1 = Read(file1) #Read first file
    _, _, data2 = Read(file2) #Read second file
    idx1 = np.where(data1['type'] == ptype)  #Select dark matter from file 1
    idx2 = np.where(data2['type'] == ptype)  # Select dark matter from file 2

    m = np.concatenate((data1['m'][idx1], data2['m'][idx2]))*1e10 #Combine masses and convert to Msun
    x = np.concatenate((data1['x'][idx1], data2['x'][idx2]))#Combine x-positions
    y = np.concatenate((data1['y'][idx1], data2['y'][idx2]))#Combine y-positions
    z = np.concatenate((data1['z'][idx1], data2['z'][idx2]))#Combine z-positions
    vx = np.concatenate((data1['vx'][idx1], data2['vx'][idx2])) #Combine x-velocities
    vy = np.concatenate((data1['vy'][idx1], data2['vy'][idx2])) #Combine y-velocities
    vz = np.concatenate((data1['vz'][idx1], data2['vz'][idx2])) #Combine z-velocities

    return {'m': m, 'x': x, 'y': y, 'z': z, 'vx': vx, 'vy': vy, 'vz': vz}  # Return combined dictionary

# Compute Center of Mass (COM) of the entire system
# Input: merged dictionary
# Output: COM position as a numpy array [x, y, z]
def compute_combined_COM(merged):
    m, x, y, z = merged['m'], merged['x'], merged['y'], merged['z']  # Unpack arrays
    COM_x = np.sum(x *m)/np.sum(m) #Weighted COM x
    COM_y = np.sum(y *m)/np.sum(m) #Weighted COM y
    COM_z = np.sum(z *m)/np.sum(m)#Weighted COM z
    return np.array([COM_x, COM_y, COM_z])  # Return COM vector

#Compute radial distance and radial velocity of each particle from the COM
#Input: merged data + COM
#Output: arrays of r (distance) and v_rad (radial velocity)
def compute_radial_quantities(merged, COM):
    x = merged['x']-COM[0] #x relative to COM
    y = merged['y']-COM[1] #y relative to COM
    z = merged['z']-COM[2] #z relative to COM
    vx, vy, vz = merged['vx'], merged['vy'], merged['vz']  #Get velocities
    r = np.sqrt(x**2 + y**2 + z**2)  # Distance to COM
    with np.errstate(divide='ignore', invalid='ignore'):
        v_rad = (vx *x +vy*y + vz *z)/r  # Project velocity onto r
        v_rad[r == 0] = 0.0  # Avoid NaNs at r=0
    return r, v_rad #Return distances and radial velocities

#Bin particles into shells to compute density and average radial velocity
#Input: r, v_rad, mass array, number of bins, min/max radius
#Output: r_mid (bin centers), density, v_rad_avg

def compute_profiles(r, v_rad, m, nbins=50, rmin=1, rmax=300):
    r_bins = np.logspace(np.log10(rmin), np.log10(rmax), nbins)  # Log-spaced radial bins
    r_mid = 0.5 * (r_bins[:-1] + r_bins[1:])  # Bin centers
    density = np.zeros(nbins-1) #Empty array for density
    v_rad_avg = np.zeros(nbins-1)  # Empty array for avg. radial velocity
    volume_shells = (4/3)*np.pi * (r_bins[1:]**3 - r_bins[:-1]**3)  # Volume of shells

    for i in range(nbins-1):  #Loop through bins
        in_bin = (r >= r_bins[i]) & (r<r_bins[i+1])  #Select particles in bin
        mass_shell = np.sum(m[in_bin])  #total mass in shell
        density[i] = mass_shell/volume_shells[i]#Compute density
        if np.sum(in_bin) > 0:
            v_rad_avg[i] = np.sum(v_rad[in_bin] * m[in_bin]) / np.sum(m[in_bin])#Mass-weighted avg. velocity
        else:
            v_rad_avg[i] = 0.0#If no particles, set to zero

    return r_mid, density, v_rad_avg  #Return bin midpoints and profiles

#Determine R200 — radius where average density = 200x critical density
#Input: data, COM
#Output: R200 radius

def compute_R200(data, COM):
    x = data['x']-COM[0]#x relative to COM
    y = data['y']-COM[1]#y relative to COM
    z = data['z']-COM[2]#z relative to COM
    m = data['m']  #Masses
    r = np.sqrt(x**2 +y**2 +z**2)#Distance to COM
    radii = np.sort(r.copy()) #Sorted distances
    mass_enclosed = np.zeros(len(radii)) # Init array for enclosed mass
    for i in range(len(radii)): #Loop over radius steps
        mass_enclosed[i] = np.sum(m[np.where(r < radii[i])])  # Sum mass inside radius
    density_enclosed = mass_enclosed/(4/3 *np.pi *radii**3)  # Enclosed density
    p_c = 8.5e-27*u.kg/u.m**3 #Critical density in SI
    p_c = p_c.to(u.Msun/u.kpc**3)  #Convert to Msun/kpc^3
    limit = 200*p_c.value #Define 200x critical density
    idx = np.argmin(abs(density_enclosed - limit))#Find radius closest to threshold

    # Plot R200 determination curve
    plt.figure(figsize=(6, 5))
    plt.loglog(radii, density_enclosed, label='Enclosed Density')  # Enclosed density vs r
    plt.axhline(limit, color='red', linestyle='--', label='200 × ρ_crit')  # Horizontal threshold
    plt.axvline(radii[idx], color='gray', linestyle=':', label=f'R200 = {radii[idx]:.1f} kpc')  # R200 vertical line
    plt.xlabel("Radius (kpc)")
    plt.ylabel("Density (Msun/kpc³)")
    plt.title("Determining R200")
    plt.grid(True, which='both', ls='--')
    plt.legend()
    plt.tight_layout()
    plt.savefig("R200_determination_plot.png", dpi=300)
    plt.show()  # Show plot

    return radii[idx]  #Return R200 radius

# Plot the final results: density profile, velocity profile, and model fits
# Input: r_mid (bin centers), density, v_rad, and R200

def plot_profiles_with_R200(r_mid, density, v_rad_avg, R200):
    plt.figure(figsize=(12, 5))  # Create wide figure

    valid = (density > 0) & (~np.isnan(density)) & (~np.isinf(density))  # Filter valid data
    r_valid = r_mid[valid]  # Clean radii
    d_valid = density[valid]  # Clean densities
    v_valid = v_rad_avg[valid]  # Clean velocities

    fit_mask = (np.abs(v_valid) < 30) & (r_valid > 10) & (r_valid < R200)  # Define fit region
    r_fit = r_valid[fit_mask]  # Radii to fit
    d_fit = d_valid[fit_mask]  # Densities to fit

    plt.subplot(1, 2, 1)  #First subplot: Density
    plt.loglog(r_mid, density, 'k-', label='Simulated Density')  # Raw density profile
    plt.loglog(r_fit, d_fit, 'go', label='Fit Region')  # Highlight fit region

    try:
        popt_h, _ = curve_fit(hernquist_density, r_fit, d_fit, p0=[1e7, 30], maxfev=100000)  # Fit Hernquist model
        rho0_h, a_h = popt_h
        r_model = np.logspace(np.log10(1), np.log10(R200), 200)  # Radii to plot model
        plt.loglog(r_model, hernquist_density(r_model, *popt_h), 'r--',
                   label=f'Hernquist Fit (a={a_h:.1f} kpc)')  # Plot fit

        popt_n, _ = curve_fit(nfw_density, r_fit, d_fit, p0=[1e7, 30], maxfev=100000)  # Fit NFW model
        rho0_n, rs_n = popt_n
        plt.loglog(r_model, nfw_density(r_model, *popt_n), 'b--',
                   label=f'NFW Fit (r$_s$={rs_n:.1f} kpc)')  # Plot fit
    except Exception as e:
        print("Profile fitting failed:", e)  #Error handling
    plt.xlabel('Radius (kpc)')
    plt.ylabel('Density (Msun/kpc$^3$)')
    plt.title('Radial Density Profile with R200')
    plt.grid(True, which='both', ls='--')
    plt.axvline(R200, color='gray', linestyle=':', label=f'R200 = {R200:.1f} kpc')  # R200 line
    plt.legend()

    plt.subplot(1, 2, 2)  # Second subplot: Radial Velocity
    plt.plot(r_mid, v_rad_avg, 'b-', label='Average $v_{{rad}}$')  # Plot velocity profile
    plt.axhline(30, color='gray', ls='--', lw=0.5)  # Upper cutoff
    plt.axhline(-30, color='gray', ls='--', lw=0.5)  # Lower cutoff
    plt.xlabel('Radius (kpc)')
    plt.ylabel('Average Radial Velocity (km/s)')
    plt.title('Radial Velocity Profile')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.savefig("MWM31_HaloProfiles_R200.png", dpi=300)  # Save figure
    plt.show()  # Show plot


file1 = 'MW_801.txt'  # Milky Way snapshot
file2 = 'M31_801.txt'  # M31 snapshot
merged = merge_snapshots(file1, file2)  # Merge particle data
COM = compute_combined_COM(merged)  # Compute center of mass
r, v_rad = compute_radial_quantities(merged, COM)  # Compute r and v_rad
r_mid, density, v_rad_avg = compute_profiles(r, v_rad, merged['m'])  # Bin into radial shells
R200 = compute_R200(merged, COM)  # Determine R200
plot_profiles_with_R200(r_mid, density, v_rad_avg, R200)  # Plot everything
