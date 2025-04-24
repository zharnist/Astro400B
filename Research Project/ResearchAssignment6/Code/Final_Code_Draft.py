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
9. Additionally, it produces a dark matter particle contour plot (from Lab 2 and Lab 7 template) to visualize the spatial structure.

Used code from Homework 2, 4, 6 and Lab 2 and Lab 7.
Referenced Swapnaneel's R200 code implementation.
Formatting guided by ChatGPT.
"""



# %% Import libraries
import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u
from scipy.optimize import curve_fit
from ReadFile import Read  
from CenterOfMass2 import CenterOfMass  #Homework 4

# %% Define Hernquist density model
def hernquist_density(r, rho0, a):
    """Hernquist profile function
    Inputs:
        r: float, radius
        rho0: float, scale density
        a: float, scale radius
    Returns:
        Density values at r
    Used from Homework 6
    """
    return rho0/((r/a)*(1 + r/a)**3)

# %% Define NFW density model
def nfw_density(r, rho0, rs):
    """
    NFW profile function
    Inputs:
        r: float, radius
        rho0: float, scale density
        rs: float, scale radius
    Returns:
        Density values at r
    Used from Homework 6
    """
    return rho0/((r/rs)*(1+r/rs)**2)




# %% Merge two halo snapshots
file1 = 'MW_801.txt'
file2 = 'M31_801.txt'

def merge_snapshots(file1, file2, ptype=1):
    """
    Merge dark matter particle data from two snapshots
    Inputs:
        file1: str, first snapshot file
        file2: str, second snapshot file
        ptype: int, particle type 1 (dark matter)
    Returns:
        dictionary with merged m, x, y, z, vx, vy, vz arrays (position and velocities)
    From Homework2
    """
    _, _, data1 = Read(file1)
    _, _, data2 = Read(file2)
    idx1 = np.where(data1['type'] == ptype)
    idx2 = np.where(data2['type'] == ptype)
    m = np.concatenate((data1['m'][idx1], data2['m'][idx2]))*1e10
    x = np.concatenate((data1['x'][idx1], data2['x'][idx2]))
    y = np.concatenate((data1['y'][idx1], data2['y'][idx2]))
    z = np.concatenate((data1['z'][idx1], data2['z'][idx2]))
    vx = np.concatenate((data1['vx'][idx1], data2['vx'][idx2]))
    vy = np.concatenate((data1['vy'][idx1], data2['vy'][idx2]))
    vz = np.concatenate((data1['vz'][idx1], data2['vz'][idx2]))
    return {'m': m, 'x': x, 'y': y, 'z': z, 'vx': vx, 'vy': vy, 'vz': vz}

# %% Compute COM
def compute_combined_COM(merged):
    """
    Compute center of mass position of merged halo
    Inputs:
        merged: dictionary of particle data
    Returns:
        np.array of COM[x, y, z]
    From Homework 4
    """
    m,x,y,z = merged['m'], merged['x'], merged['y'], merged['z']
    COM_x = np.sum(x*m)/np.sum(m)
    COM_y = np.sum(y*m)/np.sum(m)
    COM_z = np.sum(z*m)/np.sum(m)
    return np.array([COM_x, COM_y, COM_z])

# %% Compute radial distances and velocities
def compute_radial_quantities(merged, COM):
    """
    Computes radial distance and radial velocity of particles
    Inputs:
        merged: dict of merged particle data
        COM: array of center of mass [x, y, z]
    Returns:
        r: radius array
        v_rad: radial velocity array
    Used from Homework 6
    """
    x = merged['x']-COM[0]
    y = merged['y']-COM[1]
    z = merged['z']-COM[2]
    vx, vy, vz = merged['vx'], merged['vy'], merged['vz']
    r =np.sqrt(x**2 +y**2 +z**2)
    with np.errstate(divide='ignore', invalid='ignore'):
        v_rad = (vx*x+ vy+ y+vz*z)/r
        v_rad[r== 0] = 0.0
    return r, v_rad

# %% Compute density and radial velocity profiles
def compute_profiles(r, v_rad, m, rmin=1, rmax=300):
    """
    Computes radial density and velocity profiles
    
    Inputs:
        r: radius array
        v_rad: radial velocity array
        m: mass array
        rmin, rmax : floats, min and max radius
    Returns:
        r_mid: array of bin centers
        density: density per bin
        v_rad_avg: average velocity per bin
    From Homework 6
    """
    r_bins = np.arange(0.1, 300, 1)
    r_mid = 0.5*(r_bins[:-1] +r_bins[1:])
    density = np.zeros(np.size(r_bins)-1)
    v_rad_avg = np.zeros(np.size(r_bins)-1)
    volume_shells = (4/3)*np.pi*(r_bins[1:]**3 -r_bins[:-1]**3)
    for i in range(np.size(r_bins)-1):
        in_bin =(r >= r_bins[i]) & (r < r_bins[i+1])
        mass_shell =np.sum(m[in_bin])
        density[i] =mass_shell / volume_shells[i]
        if np.sum(in_bin) > 0:
            v_rad_avg[i] = np.sum(v_rad[in_bin]*m[in_bin])/np.sum(m[in_bin])
        else:
            v_rad_avg[i] = 0.0
    return r_mid, density, v_rad_avg

# %% Compute R200 and plot determination curve
def compute_R200(data, COM):
    """
    Determine R200: where enclosed density=200 × critical density
    
    Inputs:
        data: dict, merged data
        COM: array, center of mass
    Returns:
        R200: float, radius in kpc
    (Help/Worked with Swapnaneel: implementation reference from R200)
    """
    x = data['x']-COM[0]
    y = data['y']-COM[1]
    z = data['z']-COM[2]
    m = data['m']
    r = np.sqrt(x**2+y**2+z**2)
    radii = np.sort(r.copy())
    mass_enclosed = np.zeros(len(radii))
    for i in range(len(radii)):
        mass_enclosed[i] = np.sum(m[r < radii[i]])
    density_enclosed = mass_enclosed/((4/3)*np.pi*radii**3)
    p_c = (8.5e-27 *u.kg/u.m**3).to(u.Msun/u.kpc**3)
    limit = 200*p_c.value
    idx = np.argmin(np.abs(density_enclosed-limit))
    plt.figure(figsize=(6, 5))
    plt.loglog(radii, density_enclosed, label='Enclosed Density')
    plt.axhline(limit, color='red', linestyle='--', label='200 × ρ_crit')
    plt.axvline(radii[idx], color='purple', linestyle=':', label=f'R200 = {radii[idx]:.1f} kpc')
    plt.xlabel("Radius (kpc)")
    plt.ylabel("Density (Msun/kpc³)")
    plt.title("Determining R200")
    plt.grid(True, which='both', ls='--')
    plt.legend()
    plt.tight_layout()
    plt.savefig("R200_determination_plot.png", dpi=300)
    plt.show()
    return radii[idx]


# %% Plot density and velocity profiles
def plot_density_with_fits(r_mid, density, v_rad_avg, R200):
    """
    Plot density profile with Hernquist and NFW fits.
    Used some parts from Homework 6
    """
    plt.figure(figsize=(6, 5))

    valid =(density > 0) & (~np.isnan(density)) & (~np.isinf(density))
    r_valid = r_mid[valid]
    d_valid = density[valid]
    v_valid = v_rad_avg[valid]

    fit_mask = (np.abs(v_valid) < 30) & (r_valid > 10)
    r_fit = r_valid[fit_mask]
    d_fit = d_valid[fit_mask]

    r_model = np.logspace(np.log10(min(r_valid)), np.log10(max(r_valid)), 200)

    # Plot data
    plt.loglog(r_mid,density, 'k-', label='Simulated Density')

    #plot Hernquist and NFW
    popt_h, _ = curve_fit(hernquist_density, r_fit, d_fit, p0=[1e7, 30], maxfev=100000)
    plt.loglog(r_model, hernquist_density(r_model, *popt_h), 'r--',
               label=f'Hernquist Fit (a={popt_h[1]:.1f} kpc)')
    popt_n, _ = curve_fit(nfw_density, r_fit, d_fit, p0=[1e7, 30], maxfev=100000)
    plt.loglog(r_model, nfw_density(r_model, *popt_n), 'b--',
               label=f'NFW Fit (r$_s$={popt_n[1]:.1f} kpc)')

    #Add R200
    plt.axvline(R200, color='purple', linestyle=':', label=f'R200 = {R200:.1f} kpc')
    plt.xlabel('Radius (kpc)')
    plt.ylabel('Density (Msun/kpc$^3$)')
    plt.title('Radial Density Profile with Fits')
    plt.grid(True, which='both', ls='--')
    plt.legend()
    plt.tight_layout()
    plt.savefig("Density_Profile_Fits.png", dpi=300)
    plt.show()

    
def plot_radial_velocity_profile(r_mid, v_rad_avg):
    """
    Plot average radial velocity profile.
    """
    
    plt.figure(figsize=(6, 5))
    plt.plot(r_mid, v_rad_avg, 'blue', label='Average $v_{{rad}}$')
    plt.axhline(30, color='purple', ls='--', lw=0.5)
    plt.axhline(-30, color='purple', ls='--', lw=0.5)
    plt.xlabel('Radius (kpc)')
    plt.ylabel('Average Radial Velocity (km/s)')
    plt.title('Radial Velocity Profile')
    plt.grid(True)
    plt.legend()
    plt.xlim(0, 300)
    plt.ylim(-100, 100)
    plt.tight_layout()
    plt.savefig("Radial_Velocity_Profile.png", dpi=300)
    plt.show()


# %% Plot dark matter spatial contour
def plot_dm_contour(merged, COM):
    """
    Create 2D contour of dark matter distribution
    
    Inputs:
        merged: dict, merged particle data
        COM: array, center of mass
    From Lab 7
    """
    x = merged['x']-COM[0]
    y = merged['y']-COM[1]
    bins = 500
    extent =300
    H, xedges, yedges = np.histogram2d(x, y, bins=bins, range=[[-extent, extent], [-extent, extent]])
    H = np.log10(H + 1)
    plt.figure(figsize=(8, 6))
    plt.imshow(H.T, origin='lower', cmap='magma', extent=[-extent, extent, -extent, extent])
    plt.xlabel('x [kpc]')
    plt.ylabel('y [kpc]')
    plt.title('Dark Matter Particle Distribution')
    plt.colorbar(label='log₁₀(N)')
    plt.tight_layout()
    plt.savefig("DarkMatter_Contour.png", dpi=300)
    plt.show()

# %% Execute all analysis steps
merged=merge_snapshots(file1, file2)
COM=compute_combined_COM(merged)
r,v_rad =compute_radial_quantities(merged, COM)
r_mid, density, v_rad_avg = compute_profiles(r, v_rad, merged['m'])
R200 =compute_R200(merged, COM)
plot_density_with_fits(r_mid,density, v_rad_avg,R200)
plot_radial_velocity_profile(r_mid, v_rad_avg)
plot_dm_contour(merged,COM)