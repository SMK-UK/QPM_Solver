import numpy as np
import matplotlib.pyplot as mp

# choose to use frequency ('freq') or wavelength ('wave') values
input = 'freq'
# reverse 'engineer' poling period using known QPM temperature
find_P = False

c = 2.99792458E8
# find period for a given temeperature (if you have physically measured the optimum)
T_find = 52.825
# original sample temperature and co-efficient of expansion
T_0 = 20
alpha = 15.4E-6

# Temperature array
start = 0
stop = 120
step = 0.001
num = int(np.round((stop-start)/step))
T = np.linspace(start=start, stop=stop, num=num, endpoint=True)
# temperature dependant parameter F
F = (T-24.5)*(T+24.5+2*273.16)

# values in microns or GHz
freq = [193991590, 302111070]
wave = [1.550, 0.995]
#empty lists for calculated values
freqs = []
waves = []
if input == 'freq':
    for value in freq:
        waves.append(c/value)
        freqs.append(value)
    waves.append(c/(np.sqrt((freq[1]+freq[0])**2)))
    freqs.append(freq[0]+freq[1])
else:
    for value in wave:
        waves.append(value)
        freqs.append(c/value)
    freqs.append(np.sqrt((freqs[1]+freqs[0])**2))
    waves.append(c/(freqs[2]))

# Sellmeier constants
Alpha = [[5.756, 0.0983, 0.2020, 189.32, 12.52, 1.32E-2], [5.653, 0.1185, 0.2091, 89.61, 10.85, 1.97E-2]] 
Beta =  [[2.860E-6, 4.700E-8, 6.113E-8, 1.516E-4], [7.941E-7, 3.134E-8, -4.641E-9, -2.188E-6]]

# calculate the  refractive index for each wave and orientation
n = []
for index in range(len(Alpha)):
    temp_n = []
    for value in waves:
        temp_n.append(np.sqrt(Alpha[index][0] + Beta[index][0]*F + ((Alpha[index][1] + Beta[index][1]*F)/(value**2 - (Alpha[index][2] + Beta[index][2]*F)**2)) + ((Alpha[index][3] + Beta[index][3]*F)/(value**2 - Alpha[index][4]**2)) - Alpha[index][5]*value**2))
    n.append(temp_n)

# calculate k-vectors
k = []
for index in range(len(n)):
    k_temp = []
    for index_2, data in enumerate(n[index]):
        k_temp.append(2*np.pi*(data/waves[index_2]))
    k.append(k_temp)
#calculate the mismatch
mismatch = []
for index in range(len(k)):
    mismatch.append(k[index][2] - k[index][1] - k[index][0])

# find index of phase-matched temperature
if find_P == True:
    if T_find >= start and T_find <= stop:
        P_find = []
        for index, temperature in enumerate(T):
            if temperature < T_find:
                continue
            else:
                where = index
                break
        # find poling period
        for data in mismatch:
            P_find.append((2*np.pi)/data[where])

            # poling period at T_0
P = []
k_qpm = []
if find_P == True:
    P_0 = []
    for index, period in enumerate(P_find):
        if T_find > T_0:
            P_0.append(period/(1 + T_find * alpha))
        else:
            P_0.append(period/(1 - T_find * alpha))
        # poling period array over temperature range
        P.append(P_0[index] + P_0[index]*alpha*T)
        k_qpm.append((2*np.pi)/P[index])
else:
    P_0 = [10.603212266659575, 9.046371465669914]
    for index, period in enumerate(P_0):
        P.append(P_0[index] + P_0[index]*alpha*T)
        k_qpm.append((2*np.pi)/P[index])

# print out poling period
print(P_0)

# evaluate phase-matching eqaution over all T
delta_k = []
for k_p in k_qpm:
    for k in mismatch:
        delta_k.append(k - k_p)

# plot the data
fig, ax = mp.subplots()

ax.set_title('Wavevector Mismatch Calibration')
ax.set(xlabel='Temperature $^{\circ}$C', ylabel='Wavevector (cm$^{-1}$)')
ax.plot(T, delta_k[0])
ax.axhline(y=0, linestyle='--', color='orange')

for index, data in enumerate(delta_k[0]):
    if data <= 0:
        QP_temp = T[index]

print("Optimum Temperature =", round(QP_temp, 3))
print("Visible at (THz)", freqs[2]*1e-6)

mp.show()