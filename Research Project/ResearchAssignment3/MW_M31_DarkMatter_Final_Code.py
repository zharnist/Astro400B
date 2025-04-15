import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u
from scipy.optimize import curve_fit

from ReadFile import Read
from CenterOfMass2 import CenterOfMass

# ------------------------------
# Define density models
# ------------------------------
def hernquist_density(r, rho0, a):
    return rho0 / ((r / a) * (1 + r / a)**3)

def nfw_density(r, rho0, rs):
    return rho0 / ((r / rs) * (1 + r / rs)**2)

# ------------------------------
# Load and merge snapshot data
# ------------------------------
def merge_snapshots(file1, file2, ptype=1):
    _, _, data1 = Read(file1)
    _, _, data2 = Read(file2)
    idx1 = np.where(data1['type'] == ptype)
    idx2 = np.where(data2['type'] == ptype)

    m = np.concatenate((data1['m'][idx1], data2['m'][idx2])) * 1e10
    x = np.concatenate((data1['x'][idx1], data2['x'][idx2]))
    y = np.concatenate((data1['y'][idx1], data2['y'][idx2]))
    z = np.concatenate((data1['z'][idx1], data2['z'][idx2]))
    vx = np.concatenate((data1['vx'][idx1], data2['vx'][idx2]))
    vy = np.concatenate((data1['vy'][idx1], data2['vy'][idx2]))
    vz = np.concatenate((data1['vz'][idx1], data2['vz'][idx2]))

    return {'m': m, 'x': x, 'y': y, 'z': z, 'vx': vx, 'vy': vy, 'vz': vz}

# ------------------------------
# Compute Center of Mass
# ------------------------------
def compute_combined_COM(merged):
    m, x, y, z = merged['m'], merged['x'], merged['y'], merged['z']
    COM_x = np.sum(x * m) / np.sum(m)
    COM_y = np.sum(y * m) / np.sum(m)
    COM_z = np.sum(z * m) / np.sum(m)
    return np.array([COM_x, COM_y, COM_z])

# ------------------------------
# Radial Distance and Velocity
# ------------------------------
def compute_radial_quantities(merged, COM):
    x = merged['x'] - COM[0]
    y = merged['y'] - COM[1]
    z = merged['z'] - COM[2]
    vx, vy, vz = merged['vx'], merged['vy'], merged['vz']

    r = np.sqrt(x**2 + y**2 + z**2)
    with np.errstate(divide='ignore', invalid='ignore'):
        v_rad = (vx * x + vy * y + vz * z) / r
        v_rad[r == 0] = 0.0
    return r, v_rad

# ------------------------------
# Compute density & velocity profiles
# ------------------------------
def compute_profiles(r, v_rad, m, nbins=50, rmin=1, rmax=300):
    r_bins = np.logspace(np.log10(rmin), np.log10(rmax), nbins)
    r_mid = 0.5 * (r_bins[:-1] + r_bins[1:])
    density = np.zeros(nbins-1)
    v_rad_avg = np.zeros(nbins-1)
    volume_shells = (4/3) * np.pi * (r_bins[1:]**3 - r_bins[:-1]**3)

    for i in range(nbins-1):
        in_bin = (r >= r_bins[i]) & (r < r_bins[i+1])
        mass_shell = np.sum(m[in_bin])
        density[i] = mass_shell / volume_shells[i]
        if np.sum(in_bin) > 0:
            v_rad_avg[i] = np.sum(v_rad[in_bin] * m[in_bin]) / np.sum(m[in_bin])
        else:
            v_rad_avg[i] = 0.0

    print("First 5 radial bins:", r_mid[:5])
    print("First 5 densities:", density[:5])
    return r_mid, density, v_rad_avg

# ------------------------------
# Plot profiles and model fits
# ------------------------------
def plot_profiles(r_mid, density, v_rad_avg, model='both'):
    plt.figure(figsize=(12, 5))

    # Clean data
    valid = (density > 0) & (~np.isnan(density)) & (~np.isinf(density))
    r_valid = r_mid[valid]
    d_valid = density[valid]
    v_valid = v_rad_avg[valid]

    # Auto-select fit region: relaxed, stable halo
    fit_mask = (np.abs(v_valid) < 30) & (r_valid > 10) & (r_valid < 150)
    r_fit = r_valid[fit_mask]
    d_fit = d_valid[fit_mask]

    # --- Subplot 1: Density Profile ---
    plt.subplot(1, 2, 1)
    plt.loglog(r_mid, density, 'k-', label='Simulated Density')
    plt.loglog(r_fit, d_fit, 'go', label='Fit Region')

    # Hernquist fit
    if model in ['hernquist', 'both']:
        try:
            popt_h, _ = curve_fit(hernquist_density, r_fit, d_fit, p0=[1e7, 30], maxfev=100000)
            rho0_h, a_h = popt_h
            r_model = np.logspace(np.log10(1), np.log10(300), 200)
            plt.loglog(r_model, hernquist_density(r_model, *popt_h), 'r--',
                       label=f'Hernquist Fit (a={a_h:.1f} kpc)')
        except Exception as e:
            print("Hernquist fit failed:", e)

    # NFW fit
    if model in ['nfw', 'both']:
        try:
            popt_n, _ = curve_fit(nfw_density, r_fit, d_fit, p0=[1e7, 30], maxfev=100000)
            rho0_n, rs_n = popt_n
            plt.loglog(r_model, nfw_density(r_model, *popt_n), 'b--',
                       label=f'NFW Fit (r$_s$={rs_n:.1f} kpc)')
        except Exception as e:
            print("NFW fit failed:", e)

    plt.xlabel('Radius (kpc)')
    plt.ylabel('Density (Msun/kpc$^3$)')
    plt.title('Radial Density Profile')
    plt.grid(True, which='both', ls='--')
    plt.legend()

    # --- Subplot 2: Radial Velocity Profile ---
    plt.subplot(1, 2, 2)
    plt.plot(r_mid, v_rad_avg, 'b-', label='Average $v_{{rad}}$')
    plt.axhline(30, color='gray', ls='--', lw=0.5)
    plt.axhline(-30, color='gray', ls='--', lw=0.5)
    plt.xlabel('Radius (kpc)')
    plt.ylabel('Average Radial Velocity (km/s)')
    plt.title('Radial Velocity Profile')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.savefig("MWM31_HaloProfiles.png", dpi=300)
    plt.show()





def plot_normalized_profile(r_mid, density, label="Simulated"):
    """
    Plot a normalized density profile on log-log axes for shape comparison.
    """
    # Normalize radius and density
    r_norm = r_mid / np.max(r_mid)  # Optional: or r_mid / a, if Hernquist a is known
    rho_norm = density / np.max(density)

    plt.figure(figsize=(6, 5))
    plt.loglog(r_norm, rho_norm, 'k-', label=label)
    plt.xlabel("Normalized Radius (r / r_max)")
    plt.ylabel("Normalized Density (ρ / ρ_max)")
    plt.title("Normalized Density Profile")
    plt.grid(True, which='both', ls='--')
    plt.legend()
    plt.tight_layout()
    plt.show()



# ------------------------------
# MAIN EXECUTION
# ------------------------------
if __name__ == "__main__":
    file_MW = "MW_801.txt"
    file_M31 = "M31_801.txt"

    merged = merge_snapshots(file_MW, file_M31)
    COM = compute_combined_COM(merged)
    print("Center of Mass (kpc):", COM)

    r, v_rad = compute_radial_quantities(merged, COM)
    r_mid, density, v_rad_avg = compute_profiles(r, v_rad, merged['m'])

    # Now compare both Hernquist and NFW
    plot_profiles(r_mid, density, v_rad_avg, model='both')

    print("Plot saved as MWM31_HaloProfiles.png")
    
    plot_normalized_profile(r_mid, density)

