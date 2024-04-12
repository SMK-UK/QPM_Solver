"""
Quasi-Phase Matching Finder

Author: Sean Keenan
GitHub: SMK-UK
Date: 15/02/2024

"""

import qpm_funcs as qpm
from numpy import  array, linspace, round

# specify units to work in
input = 'freq'
# reverse 'engineer' poling period using known QPM temperature
find_P = True
# enter known polling periods [(e-wave), (o-wave)]
P_known = [10.603212266659575, 9.046371465669914]
T_find = 53.96          # Temperature of optimum conversion
T_0 = 20                # Room temp of sample (where polling period is usually quoted)
alpha = 15.4E-6         # co-efficient of expansion for material
# input frequencies (GHz) or wavelengths (um) to use
units = [193375500, 301343200] # [1.55, 0.995] 
# Sellmeier co-efficients 
Alpha = [[5.756, 0.0983, 0.2020, 189.32, 12.52, 1.32E-2], [5.653, 0.1185, 0.2091, 89.61, 10.85, 1.97E-2]] 
Beta =  [[2.860E-6, 4.700E-8, 6.113E-8, 1.516E-4], [7.941E-7, 3.134E-8, -4.641E-9, -2.188E-6]]
# Temperature array
start = 0
stop = 100
step = 0.001
num = int(round((stop-start)/step))
temperatures = linspace(start=start, stop=stop, num=num, endpoint=True)
# create lists of wavelengths and frequencies
wavelengths, frequencies = qpm.create_lists(units, unit_type=input)
# print out visible frequency
print(f"Visible at (THz), {frequencies[2]*1e-6 :.2f}")
# calculate the refractive index for ordinary (o) and extraordinary axis (e) at each wavelength and temperature
ne = [qpm.calculate_n(wavelength, alphas=Alpha[0], betas=Beta[0], T=temperatures) for wavelength in wavelengths]
no = [qpm.calculate_n(wavelength, alphas=Alpha[1], betas=Beta[1], T=temperatures) for wavelength in wavelengths]
# calculate the wavevectors for each wavelength and axis
ke = [qpm.calculate_k(wavelength, n) for wavelength, n in zip(wavelengths, ne)]
ko = [qpm.calculate_k(wavelength, n) for wavelength, n in zip(wavelengths, no)]
if find_P == True:
    # find the polling period of your crystal given a known optimum temperature
    Pe_0 = qpm.find_P0(T_find, temperatures, ke, T_0, alpha)
    Po_0 = qpm.find_P0(T_find, temperatures, ko, T_0, alpha)
else:
    # enter known polling periods
    Pe_0 = P_known[0]
    Po_0 = P_known[1]
# calculate polling period over temperature range
Pe = array([qpm.calculate_P(Pe_0, temperature, alpha) for temperature in temperatures])
Po = array([qpm.calculate_P(Po_0, temperature, alpha) for temperature in temperatures])
# calculate the wavevector mismatch
delta_ke = qpm.calculate_dk(Pe, ke)
delta_ko = qpm.calculate_dk(Po_0, ko)
# plot and extract the dk = 0 temperature
qpm.plot_mismatch(temperatures, delta_ke, title='extraordinary dk')
qpm.plot_mismatch(temperatures, delta_ko, title='ordinary dk')