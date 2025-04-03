"""
MWM31_HaloDensityProfile.py
Zach Harnist
Research Assignment 3

Project Topic and Summary:
This code supports my research project in trying to find the structure of the dark matter halo 
that forms after the major merger between the Milky Way (MW) and Andromeda (M31). 
Using the provided data, my goal is to analyze how the density profile of the merged 
halo evolves, looking specifically at whether it shows a classical dark matter halo model like 
the Hernquist and NFW profiles.

Research Question:
''Does the dark matter halo that forms after the MW-M31 merger follow a Hernquist profile, 
an NFW profile, or something different? How does violent relaxation and energy redistribution 
during the merger impact the inner density slope?''


What I expect the code to do/what the code does
Loads the simulation snapshot of the MW-M31 merger remnant
Takes only the dark matter particles from the data
Calculates the center of mass (COM) using the shrinking sphere method
Recenters all the dark matter particles based on the center of mass
Computes the radial distance of each dark matter particle from the center
Bins particles in logarithmically spaced spherical shells
Calculates the mass density in each shell
Plots the density profile on a loglog plot
Fits the profile with both Hernquist and NFW models for comparison
Finally saves the figure

"""
#Already attempted to implement some code, will probably need to be revised tomorrow after some discussion
# import modules


# import numpy as np                             
# import matplotlib.pyplot as plt
# from astropy import units as u
# from scipy.optimize import curve_fit           
# from ReadFile import Read                      
# from CenterOfMass2 import CenterOfMass         

#Density profiles

# def hernquist_density(values here):
#     write hernquist equation
#     return the equation

# def nfw_density(values here):
#     
#     Write NFW equation
#     return NFW equation



#Need to compute a density profile
# def compute_density_profile():
    
#     Parameters:
#         the filename
#         the particle type, for dark matter type is 1
#         the number of bins 

#     Returns:
#         the bin midpoints in kpc
#         the density
#
#     Steps:
#     Read a simulation snapshot
#     Filters dark matter particles by type=1
#     Compute the center of mass using shrinking sphere method which was done in homework 4
#     Recenter particles on the center of mass
#     Bin the particles into spherical shells (I think this was done in lab 4)
#     Compute mass density in each shell

#     #step 1: Load the file using the read function from Homework 2
#     time, total, data = Read(filename)

#     #step 2: Select dark matter particles
#     index = np.where(data['type'] == ptype)
#     m = data['m'][index] * 1e10  #this Converts the value to Msun (simulation units are 1e10 Msun)
#     x = data['x'][index]
#     y = data['y'][index]
#     z = data['z'][index]

#     #step 3: Compute the center of mass and recenter positions using code from Homework 4
#     COM=CenterOfMass(filename, ptype)
#     COM_pos = com.COM_P(delta=0.1, volDec=2)  #shrinking sphere method
#     x -= COM_pos[0].value
#     y -= COM_pos[1].value
#     z -= COM_pos[2].value

#     #step 4: Compute ther radius of each particle
#     r=np.sqrt(x**2 + y**2 + z**2)

#     #step 5: Logarithmic radial bins, done in Lab 4
#     r_bins= np.logspace(np.log10(1), np.log10(300), nbins)  # 1 kpc to 300 kpc
#     r_mid = 0.5*(r_bins[1:]+r_bins[:-1])    #Midpoints for plotting
#     volume_shells =(4/3)*np.pi*(r_bins[1:]**3 - r_bins[:-1]**3)  # volumes

#     #step 6: Compute density in each shell
#     density = np.zeros(nbins - 1)
#     for i in range(nbins - 1):
#         in_bin = (r >= r_bins[i]) and (r < r_bins[i+1])
#         mass_shell = np.sum(m[in_bin])
#         density[i] = mass_shell/volume_shells[i]

#     return ...





# def plot_density_profile():
#    

#     Parameters:
#         radial bin centers in kpc (r)
#         the density values (Msun/kpc^3) (rho)
#         the model, either hernquist, nfw, or both (model)
#

#     plt.figure(figsize=(8, 6))
#     plt.loglog(r, rho, 'ko', label='Simulated Data')  # Plots the simulation points

#     #could also try and fit the Hernquist profile
#     if model in ['hernquist', 'both']:
#         popt_h, _ = curve_fit(hernquist_density, r, rho, p0=[1e7, 30])
#         plt.loglog(r, hernquist_density(r, *popt_h), 'r--',
#                    label=f'Hernquist Fit (a={popt_h[1]:.1f} kpc)')

#     #Same with trying to fit the NFW profile
#     if model in ['nfw', 'both']:
#         popt_n, _ = curve_fit(nfw_density, r, rho, p0=[1e7, 30])
#         plt.loglog(r, nfw_density(r, *popt_n), 'b-.',
#                    label=f'NFW Fit (rs={popt_n[1]:.1f} kpc)')

#
#     plt.xlabel('Radius [kpc]', fontsize=14)
#     plt.ylabel(r'Density [$M_\odot$ / kpc$^3$]', fontsize=14)
#     plt.title('Dark Matter Density Profile (MW+M31 Remnant)', fontsize=15)
#     plt.legend()
#     plt.grid(True)
#     plt.tight_layout()
#     plt.savefig("DensityProfile_MWM31.png", dpi=300)
#     plt.show()





