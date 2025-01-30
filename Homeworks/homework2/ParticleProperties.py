# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 17:31:48 2025

@author: harni
"""
import numpy as np
from astropy import units as u
from ReadFile import Read


"""
This function will extract the properties of the 100th disk particle using the data from MW_000
It calcluates the Distance, the velocity and the mass of the particle.
The distance is converted into lightyears from Astropy
It will return the parameters of the 100th particle using the data set from our file

"""

def particleproperties(MW_000): #Defines the function particle properties using MW_000 (our filename)
    
    
    _,_, data = Read(MW_000) #skips the time and number of particles and only gives the data from MW_000

    

    x=(data['x']*u.kpc) #Finds the distance for all particles in x coordinate and converts it to kpc using astropy package
    y=(data['y']*u.kpc) #Finds the distance in y coordinate and converts it to kpc using astropy package
    z=(data['z']*u.kpc) #Finds the distance in z coordinate and converts it to kpc using astropy package
    


    vx=data['vx']*u.km/u.s #Finds the velocity in x coordinate and converts it to km/s using astropy package
    vy=data['vy']*u.km/u.s #Finds the velocity in y coordinate and converts it to km/s using astropy package
    vz=data['vz']*u.km/u.s #Finds the velocity in z coordinate and converts it to km/s using astropy package

    particletype=2.00000 #type 2 is when disk particle data begins
    index=np.where(data['type']==particletype) #finds all disk particles and stores its indices
#Choosing index 99 because we want the 100th particle
#Finding the distance in x,y,z coordinates
    x_type=x[index][99] #extracts distance in x only for 100th particle
    y_type=y[index][99] #extracts distance in y only for 100th particle
    z_type=z[index][99] #extracts dinstance in z only for 100th particle
#Choosing index 99 because we want the 100th particle
#Finding the velocities in x, y, z directions
    vx_type=vx[index][99] #velocity in x
    vy_type=vy[index][99] #velocity in y
    vz_type=vz[index][99] #velocity in z

    mass_type=data['m'][index][99]#extracts the mass of the 100th particle from the data

    distance = np.sqrt(x_type**2 + y_type**2 +z_type**2) #This is the equation we use to solve for distance with the particles x,y,z coordinates
    distance_ly = distance.to(u.lyr) #Here we convert the units to light years from kpc
    
    distance_ly =np.around(distance,3) #we are rounding the light year distance by three decimals
    Velocity = np.sqrt(vx_type**2 + vy_type**2 + vz_type**2) #This is the equation we use to solve for velocity of the particle in km/s
    Velocity=np.around(Velocity,3) #we round the velocity value by a decimal of three
    

    return distance_ly, Velocity, mass_type #return the calculated distance in light years, velocity of the particle in km/s and the mass of the particle to the sun
MW_000 = 'MW_000.txt' #MW_000 is the filename and MW_000.txt is the text file that has our data
distance, Velocity, mass_type = particleproperties(MW_000) #calls the particleproperties function

print("The following properties of the 100th disk particle consist of:") 
print(f"Distance: {distance}") #prints distance  in lightyears
print(f"Velocity: {Velocity}") #prints velocity of particle
print(f"Mass: {mass_type} M_sun") #prints the mass of particle to mass sun