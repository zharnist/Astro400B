

# Homework 6 Template
# G. Besla & R. Li




# import modules
import numpy as np
import astropy.units as u
from astropy.constants import G

# import plotting modules
import matplotlib.pyplot as plt
import matplotlib

# my modules
from ReadFile import Read
# Step 1: modify CenterOfMass so that COM_P now takes a parameter specifying 
# by how much to decrease RMAX instead of a factor of 2
from CenterOfMass2 import CenterOfMass



import os
print(os.path.exists("VLowRes/MW/MW_000.txt"))
print(os.path.exists("VLowRes/M31/M31_000.txt"))
print(os.path.exists("VLowRes/M33/M33_000.txt"))




def OrbitCOM(galaxy, start, end, n):
    """function that loops over all the desired snapshots to compute the COM pos and vel as a function of time.
     Parameters:
    galaxy : str
        Name of the galaxy (e.g., 'MW', 'M31', 'M33').
    start : int
        The first snapshot number to read.
    end : int
        The last snapshot number to read.
    n : int
        Interval over which snapshots are processed.
    """
    
    # compose the filename for output
    fileout = f"Orbit_{galaxy}.txt"
    #  set tolerance and VolDec for calculating COM_P in CenterOfMass
    # for M33 that is stripped more, use different values for VolDec
    delta = 0.1
    volDec = 2
    if galaxy == "M33":
        volDec = 4  # M33 is more tidally stripped
    
    # generate the snapshot id sequence 
    # it is always a good idea to also check if the input is eligible (not required)
    snap_ids = np.arange(start, end+1, n)
    if len(snap_ids) == 0:
        raise ValueError("Invalid snapshot range or interval.")
    
    # initialize the array for orbital info: t, x, y, z, vx, vy, vz of COM
    orbit = np.zeros((len(snap_ids), 7))
    
    # a for loop 
    for i, snap_id in enumerate(snap_ids):
        
        # Compose the data filename (be careful about the folder)
        ilbl = f"{snap_id:03d}"
        filename = f"VLowRes/{galaxy}/{galaxy}_{ilbl}.txt"
        
        # Initialize an instance of CenterOfMass class, using disk particles
        COM = CenterOfMass(filename, 2)
        
        # Store the COM pos and vel. Remember that now COM_P requires VolDec
        COM_p = COM.COM_P(delta, volDec)
        COM_v = COM.COM_V(COM_p[0], COM_p[1], COM_p[2])
        
        # Store the time, pos, vel in ith element of the orbit array, without units (.value)
        orbit[i] = [COM.time.to(u.Myr).value/1000, COM_p[0].value, COM_p[1].value, COM_p[2].value,
            COM_v[0].value, COM_v[1].value, COM_v[2].value]

        
        # Print snap_id to see the progress
        print(f"Processed snapshot {snap_id}")
        
    # write the data to a file
    # we do this because we don't want to have to repeat this process 
    # this code should only have to be called once per galaxy.
    np.savetxt(fileout, orbit, fmt = "%11.3f"*7, comments='#',
               header="{:>10s}{:>11s}{:>11s}{:>11s}{:>11s}{:>11s}{:>11s}"\
                      .format('t', 'x', 'y', 'z', 'vx', 'vy', 'vz'))




# Recover the orbits and generate the COM files for each galaxy
# read in 800 snapshots in intervals of n=5
# Note: This might take a little while - test your code with a smaller number of snapshots first! 

OrbitCOM("MW", 0, 800, 5)
OrbitCOM("M31", 0, 800, 5)
OrbitCOM("M33", 0, 800, 5)


# Read in the data files for the orbits of each galaxy that you just created
# headers:  t, x, y, z, vx, vy, vz
# using np.genfromtxt

MW_data = np.genfromtxt("Orbit_MW.txt", names=True)
M31_data = np.genfromtxt("Orbit_M31.txt", names=True)
M33_data = np.genfromtxt("Orbit_M33.txt", names=True)


# function to compute the magnitude of the difference between two vectors 
# You can use this function to return both the relative position and relative velocity for two 
# galaxies over the entire orbit  

def magnitude_difference(data1, data2):
    """
    Function to compute the magnitude of the difference between two position vectors.

    Parameters:
    ----------
    data1 : numpy structured array
        Contains position data for the first galaxy (x, y, z).
    data2 : numpy structured array
        Contains position data for the second galaxy (x, y, z).

    Returns:
    -------
    numpy.ndarray
        Array of magnitudes representing the relative distance between the two galaxies.
    """
    
    return np.sqrt((data1['x'] - data2['x'])**2 + (data1['y'] - data2['y'])**2 + (data1['z'] - data2['z'])**2)


# Determine the magnitude of the relative position and velocities 

# of MW and M31
MW_M31_separation = magnitude_difference(MW_data, M31_data)
# of M33 and M31
M33_M31_separation = magnitude_difference(M33_data, M31_data)



# Plot the Orbit of the galaxies 
#################################
plt.figure()
plt.plot(MW_data['t'], MW_M31_separation, label='MW-M31 Separation')
plt.plot(M33_data['t'], M33_M31_separation, label='M33-M31 Separation')
plt.xlabel('Time (Gyr)')
plt.ylabel('Separation (kpc)')
plt.legend()
plt.title('Galactic Orbits')
plt.show()



# Plot the orbital velocities of the galaxies 
#################################
plt.figure()
plt.plot(MW_data['t'], np.linalg.norm([MW_data['vx'] - M31_data['vx'], MW_data['vy'] - M31_data['vy'], MW_data['vz'] - M31_data['vz']], axis=0), label='MW-M31 Velocity')
plt.plot(M33_data['t'], np.linalg.norm([M33_data['vx'] - M31_data['vx'], M33_data['vy'] - M31_data['vy'], M33_data['vz'] - M31_data['vz']], axis=0), label='M33-M31 Velocity')
plt.xlabel('Time (Gyr)')
plt.ylabel('Velocity (km/s)')
plt.legend()
plt.title('Galactic Velocities')
plt.show()

#Questions:
    '''''
   How many close encounters will the MW and M31 experience?
--------------------------------------------------------------
From the orbital separation plot, we see that MW and M31 
experience **two** close encounters before they finally merge.

--------------------------------------------------------------
2.) How is the time evolution of separation and velocity related?
--------------------------------------------------------------
- The **separation and relative velocity are anti-correlated**.
- When MW and M31 approach (separation decreases), their 
  velocity **increases** due to gravitational attraction.
- When they move apart after the first encounter, velocity 
  **decreases** again.
- This pattern repeats until the final merger.

--------------------------------------------------------------
3.) When do MW and M31 merge? What happens to M33?
--------------------------------------------------------------
- MW and M31 will **merge around 6–7 Gyr into the future** 
  based on the separation plot.
- The relative velocity approaches zero at this time.
- M33's orbit **becomes unstable** after MW and M31 merge:
  - It may **spiral in and merge** with MW+M31.
  - It could be **ejected into a new orbit** due to changes 
    in the gravitational potential.

--------------------------------------------------------------
4.) What is the decay rate of M33’s orbit after 6 Gyr?
--------------------------------------------------------------
- The orbital decay rate is estimated by comparing successive 
  apocenters after 6 Gyr.
- If the **apocenter distance decreases by a few kpc per Gyr**, 
  we estimate:

  Decay rate = (Change in Apocenter Distance) / (Orbital Period)

- If M33 is currently **~75 kpc** from MW+M31:
 It will merge in **~10–15 Gyr** if the decay is steady.
    ''