'''
Functions for use in the QPM solver

Author: Sean Keenan
Date: 15/02/2024
GitHub: SMK-UK

'''

import matplotlib.pyplot as mp
from numpy import argmin, meshgrid, pi, sqrt, where

c = 2.99792458E8

def calculate_dk(period, k, mode='SFG'):
    '''
    Calculate the diffence in wavevectors
    
    <period>:
        Polling period of the crystal
    <k>:
        Wavevector array of waves mixing in the crystal
        
    '''
    if mode == 'SFG':
        sign = 1
    else:
        sign = -1
        
    return k[2] - k[0] - sign*k[1] - sign*converter(period)

def calculate_k(wavelength:float, n_index:float):
    '''
    Calculate corresponding k-vector for each wave

    <wavelengths>:
        Wavelengths to determine refractive index in um
    
    '''
    return (2*pi*(n_index/wavelength))

def calculate_n(wavelength:float, alphas:list[float], betas:list[float], T:list[float]):
    '''
    Calculate the refractive index of a material based on the Sellmeir co-efficients

    <wavelength>:
        Wavelength to determine refractive index in um
    <alphas>:
        Sellmeier A-coefficients
    <betas>:
        Sellmeier B-coefficients
    <T>:
        Temperature range over which to calculate n
    
    '''
    # temperature dependant parameter F
    F = (T-24.5)*(T+24.5+2*273.16)
    # calculate the  refractive index for each wave and orientation
    return (sqrt(alphas[0] + betas[0]*F + ((alphas[1] + betas[1]*F)/(wavelength**2 - (alphas[2] + betas[2]*F)**2)) + ((alphas[3] + betas[3]*F)/(wavelength**2 - alphas[4]**2)) - alphas[5]*wavelength**2))

def calculate_P0(period, T_find, T_0, alpha):
    '''
    Calculate the polling period at a given temperature
    
    <period>:
        polling period at a known temperature
    <T-find>:
        temperature to calculate new polling period
    <T_0>:
        temperature of known polling <period>
    <alpha>:
        co-efficient of thermal expansion for crystal

    '''
    if T_find > T_0:
        P = period/(1 + T_find * alpha)
    else:
        P = period/(1 - T_find * alpha)
    return P

def calculate_P(P_0, alpha, T):
    '''
    Calculate a polling period over a range of temperatures
    
    <P_0>:
        Polling period at 0 temperature
    <alpha>:
        Co-efficient of expansion of crystal
    <T>:
        Temperatures over which to calculate P

    '''
    return P_0 + P_0*alpha*T
    
def converter(unit):
    '''
    Convert between polling period and wavevector

    <unit>:
        polling period or wavevector to convert
    
     '''
    return (2*pi)/unit

def create_lists(values:list[float], unit_type:str='freq', mode='SFG'):
    '''
    Create lists of frequency or wavelength data
    depending on user input

    <values>:
        units to convert 
    <unit_type>:
        choose either 'freq' or 'wavelength'
    
    '''
    if mode == 'SFG':
        sign = 1
    else:
        sign = -1
    wave_list = []
    freq_list = []
    if unit_type == 'freq':
        for value in values:
            wave_list.append(c/value)
            freq_list.append(value)
        wave_list.append(c/(sqrt((values[1]+sign*values[0])**2)))
        freq_list.append(values[0]+sign*values[1])
    else:
        for value in values:
            wave_list.append(value)
            freq_list.append(c/value)
        freq_list.append(sqrt((freq_list[1]+sign*freq_list[0])**2))
        wave_list.append(c/(freq_list[2]))

    return wave_list, freq_list

def find_P0(T_find, T, k, T_0, alpha, mode='SFG'):
    '''
    Find the polling period at room temperature

    <T_find>:
        Temperature of optimum conversion]
    <T>:
        Array of temperature values around T_find
    <k>:
        Wavevectors for the waves used in the mixing process
    
    '''
    P = find_P(T_find, T, k, mode=mode)

    return calculate_P0(period=P, T_find=T_find, T_0=T_0, alpha=alpha)

def find_P(T_find:float, T:list[float], k_v:list[float], mode):
    '''
    Find the poling period for a given temperature using 
    calculated wavevectors. Uses the refractive indexes
    as given by sellmeier equation over a temperature range

    <T_find>:
        Temperature at which to calculate the polling period
    <T>:
        Array of Temperature points
    <mismatch>:
        k-vector mismatch  

    '''
    # calculate k-mismatch
    mismatch = k_mismatch(k_vectors=k_v, mode=mode)
    # find index of phase-matched temperature
    if T_find >= min(T) and T_find <= max(T):
        where = argmin(abs(T-T_find))

    return -converter(mismatch[where])

def k_mismatch(k_vectors:list[float], mode='SFG'):
    '''
    Return the k-vector mismatch for 3 waves

    <k_vectors>:
        list of wave vectors for 3 waves (1/um) 
    
    '''
    if mode == 'SFG':
        sign = 1
    else:
        sign = -1

    return k_vectors[0] + sign*k_vectors[1] - k_vectors[2]

def plot_mismatch(x, y, title):
    '''
    Plot the k-vector mismatch and return the optimal phase matching temperature
    
    <x>:
        x-data to plot
    <y>:
        y-data to plot
    <title>:
        title of plot

    '''
    # plot the data
    fig, ax = mp.subplots()

    ax.set_title(f'{title}')
    ax.set(xlabel='Temperature $^{\circ}$C', ylabel='Wavevector (cm$^{-1}$)')
    ax.plot(x, y)
    ax.axhline(y=0, linestyle='--', color='orange')
    mp.show()
    
    QP_temp = x[argmin(abs(y))]

    print(f"Optimum Temperature = {QP_temp :.3f}")

    return fig, ax

def plot_sfg_cont(x, y, z, res, title):

    x, y = meshgrid(x, y)

    fig, ax = mp.subplots()

    cont = ax.contour(x, y, z, res)
    cbar = fig.colorbar(cont)
    ax.set(title='QPM Temperature at 605.99nm')
    ax.set(xlabel='SWIR Detuning kHz', ylabel='NIR Detuning kHz')



