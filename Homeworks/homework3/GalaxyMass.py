# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 15:29:28 2025

@author: harni
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pandas.plotting import table

from ReadFile import Read 
 

def ComponentMass(file, particle_type):
    """
    Computes total mass of a given galaxy.

    Parameters:
        file (str): The galaxy data file.
        particle_type (int): The particle type (1 = Halo, 2 = Disk, 3 = Bulge).

    Returns:
        float: Total mass in units of 10^12 M_sun, rounded to 3 decimal places.
    """
    _, _, data = Read(file)  
    index = np.where(data['type'] == particle_type)  
    mass_total = np.sum(data['m'][index]) * 1e-2#Convert to 10^12 M_sun
    return np.round(mass_total, 3)

def MassBreakdown(galaxies, files):
    """
    Computes mass breakdown for each galaxy and the Local Group.

    Parameters:
        galaxies (list): List of galaxy names.
        files (list): List of corresponding filenames.

    Returns:
        DataFrame: The table that contains mass and baryon fractions
    """
    results = []
    total_halo, total_disk, total_bulge, total_mass, total_stellar = (0, 0, 0, 0, 0)

    for i in range(len(galaxies)):
        g, f = galaxies[i], files[i]  
        halo = ComponentMass(f, 1)  
        disk = ComponentMass(f, 2)  
        bulge = ComponentMass(f, 3)  

        total= halo+disk+bulge
        fbar=np.round((disk+bulge)/total, 3)
        results.append([g, halo, disk, bulge, total, fbar])

        
        total_halo += halo
        total_disk += disk
        total_bulge += bulge
        total_mass += total
        total_stellar += (disk + bulge)
    local_fbar = np.round(total_stellar/total_mass, 3) 
    results.append(["Local Group", np.round(total_halo, 3), np.round(total_disk, 3),np.round(total_bulge, 3), np.round(total_mass, 3),np.round(local_fbar, 3)])

    return pd.DataFrame(results, columns=["Galaxy", "Halo Mass", "Disk Mass", "Bulge Mass", "Total Mass", "f_bar"])




def SaveAsPDF(dataframe, questions, filename="MassBreakdown.pdf"):
    """
    Saves the mass breakdown table and the questions as a PDF.

    Parameters:
        datafra,e (DataFrame): The dataframe containing the table.
        questions (str): The text containing the questions and answers.
        filename (str): The output filename.
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    tbl = table(ax, dataframe, loc='upper center', cellLoc='center', colWidths=[0.15] * len(dataframe.columns))
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1.2, 1.2)
    text_x = 0.05  
    text_y = -0.6  
    ax.text(text_x, text_y, questions, fontsize=8, verticalalignment='top', wrap=True)


    plt.savefig(filename, bbox_inches='tight', format="pdf")
    
    plt.close()



galaxies = ["MW", "M31", "M33"]
files = ["MW_000.txt", "M31_000.txt", "M33_000.txt"]


dataframe = MassBreakdown(galaxies, files)


questions = '''
1. How does the total mass of the MW and M31 compare in this simulation? What galaxy component dominates this total mass?
   - MW and M31 share the same mass for this simulation. The most dominant component in either galaxy is the halo mass.

2. How does the stellar mass of the MW and M31 compare? Which galaxy do you expect to be more luminous?
   - M31 has a higher stellar mass than MW. I think M31 would be more luminous than MW because it has a higher stellar mass.

3. How does the total dark matter mass of MW and M31 compare in this simulation (ratio)? Is this surprising, given their difference in stellar mass?
   - The ratio of MW and M31 is 1.975/1.921 = 1.028 meaning that both galaxies have nearly the same dark matter.
   - This is surprising because M31 has more stellar mass, yet shares the same dark matter with MW that has a lower stellar mass.

4. What is the ratio of stellar mass to total mass for each galaxy (i.e., the Baryon fraction)? 
   - The computed baryon fractions are: MW = 4.1%, M31 = 6.7%, and M33 = 4.6%.
   - In the Universe, Ωb/Ωm = 16% of all mass is locked up in baryons (gas & stars) vs. dark matter.
   - The values above are much lower. Some possible reasons include:
     - Not all baryons in galaxies form stars.
     - Gas loss from explosions may eject gas, lowering the baryon fraction.
'''


SaveAsPDF(dataframe, questions, "MassBreakdown.pdf")




print(dataframe)














