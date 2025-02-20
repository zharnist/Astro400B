import numpy as np
import astropy.units as u
from astropy.constants import G
import matplotlib.pyplot as plt

from ReadFile import Read
from CenterOfMass import CenterOfMass

class GalaxyMassProfile:

    def __init__(self, galaxy_name, snapshot):
        """
        Loads the data and computes the galaxy center of mass.
        
        Inputs:
            galaxy_name(str): Name of the galaxy (MW, M31, M33)
            snapshot(int): Snapshot number
        
        Outputs:
            Initializes class with galaxy properties and the data of each galaxy
        """
        self.galaxy= galaxy_name
        snap_str= f"{snapshot:03d}"
        self.file = f"{galaxy_name}_{snap_str}.txt"
        
        self.time, self.total_particles, self.dataset =Read(self.file)
        
        self.coords = np.vstack((self.dataset['x'], self.dataset['y'], self.dataset['z']))*u.kpc
        self.mass =self.dataset['m']*1e10*u.Msun
        
        com_finder= CenterOfMass(self.file, 2)
        self.center_of_mass = com_finder.COM_P(0.1)
        
        self.G = G.to(u.kpc*u.km**2/(u.s**2*u.Msun))
    
    
    
    def enclosed_mass(self, ptype, radii):
        """
        Calculates enclosed mass for a particle type.
        
        Inputs:
            ptype(int): Particle type (1=Halo, 2=Disk, 3=Bulge)
            radii(array): Radii in kpc where mass is computed
        
        Outputs:
            enclosed_mass(quantity): array of enclosed masses in Msun at given radii
        """
        particle_mask=self.dataset['type'] == ptype
        m_selected =self.mass[particle_mask]  
    
        positions =self.coords[:, particle_mask] - self.center_of_mass[:, None]
        r_part = np.linalg.norm(positions, axis=0)  
    
        radii_kpc = np.array(radii)*u.kpc
    
        enclosed_mass = u.Quantity([np.sum(m_selected[r_part < r]) for r in radii_kpc], u.Msun)
        return enclosed_mass
    
    
    
    
    def total_enclosed_mass(self, radii):
        """
        Computes total enclosed mass from all components.
        
        Inputs:
            radii(array): radii in kpc where total mass is calculated
        
        Outputs:
            total_mass(Quantity): Total enclosed mass in Msun at given radii
        """
        halo_mass = self.enclosed_mass(1, radii)
        disk_mass = self.enclosed_mass(2, radii)
        bulge_mass = self.enclosed_mass(3, radii) if self.galaxy != 'M33' else 0 * halo_mass
        
        
        return halo_mass + disk_mass + bulge_mass
    
    def hernquist_mass(self, r, scale_length, halo_mass):
        """
        Compute the Hernquist mass profile.
        
        Inputs:
            r(array): Radii in kpc
            scale_length(float): scale radius in kpc
            halo_mass(float): total halo mass in Msun
        
        Outputs:
            Hernquist mass at given radii in Msun
        """
        
        return halo_mass*(r**2/(r + scale_length)**2)*u.Msun
    
    def circular_velocity(self, ptype, radii):
        """
        Compute circular velocity.
        
        Inputs:
            ptype(int): Particle type (1=Halo, 2=Disk, 3=Bulge)
            radii(array-like): Radii in kpc
        
        Outputs:
            V_circ(quantity): circular velocity in km/s
        """
        mass_enc =self.enclosed_mass(ptype, radii)
        radii_kpc = np.array(radii)*u.kpc
        
        return np.sqrt(self.G*mass_enc/radii_kpc).to(u.km/u.s)
    
    def total_circular_velocity(self, radii):
        """
        Compute total circular velocity including all components.
        
        Inputs:
            radii(array): Radii in kpc
        
        Outputs:
            V_circ_total(quantity): Total circular velocity in km/s
        """
        total_mass = self.total_enclosed_mass(radii)
        radii_kpc = np.array(radii)*u.kpc
        
        
        return np.sqrt(self.G*total_mass/radii_kpc).to(u.km/u.s)

def plot_mass_profiles(galaxy_profile, radii, scale_length):
    """
    Plot enclosed mass profiles for a given galaxy.
    
    Inputs:
        galaxy_profile: GalaxyMassProfile (previous code)
        radii(array): Radii in kpc
        scale_length(float): Scales radius in kpc
    
    Outputs:
        Saves a plot showing mass profiles
    """
    halo_mass = galaxy_profile.enclosed_mass(1, radii)
    disk_mass = galaxy_profile.enclosed_mass(2, radii)
    bulge_mass = galaxy_profile.enclosed_mass(3, radii) 
    total_mass = galaxy_profile.total_enclosed_mass(radii)
    
    hernquist_mass = galaxy_profile.hernquist_mass(radii, scale_length, halo_mass[-1].value)
    
    plt.figure(figsize=(8, 6))
    plt.semilogy(radii, halo_mass, 'b', label='Halo')
    plt.semilogy(radii, disk_mass, 'r', label='Disk')
    plt.semilogy(radii, bulge_mass, 'g', label='Bulge')
    plt.semilogy(radii, total_mass, 'k', label='Total')
    plt.semilogy(radii, hernquist_mass, 'b--', label=f'Hernquist (a={scale_length} kpc)')
    
    plt.title(f'Mass Profile: {galaxy_profile.galaxy}', fontsize=15)
    plt.xlabel('Radius (kpc)', fontsize=14)
    plt.ylabel('Enclosed Mass (Msun)', fontsize=14)
    plt.legend()
    plt.show()
    plt.savefig(f"Mass_Profile_{galaxy_profile.galaxy}.png", dpi=300, bbox_inches="tight")

#profiles for each galaxy
galaxies = {name: GalaxyMassProfile(name, 0) for name in ["MW", "M31", "M33"]}
radii_range = np.linspace(0.1, 30, 100)
scale_lengths = {"MW": 60, "M31": 60, "M33": 25}

for name, profile in galaxies.items():
    plot_mass_profiles(profile, radii_range, scale_lengths[name])

