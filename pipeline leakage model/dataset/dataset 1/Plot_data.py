"""
@author: Joseph Cornelius van der Walt
         nelisvanderwalt@gmail.com
"""

import numpy as np
import matplotlib.pyplot as plt
fname='Leak 3_30m.npy'
data=np.load(fname)                              # Data structure [P1, Q1, P2, Q2]
plt.figure(1)
plt.subplot(4,1,1)
plt.plot(data[:,0])                             #Plot P1    [m]
plt.ylabel('P1 [m]')
plt.subplot(4,1,2)
plt.plot(data[:,1])                             #Plot Q1    [l/s]
plt.ylabel('Q1 [l/s]')

plt.subplot(4,1,3)
plt.plot(data[:,2])                             #Plot P2    [m]
plt.ylabel('P2 [m]')

plt.subplot(4,1,4)
plt.plot(data[:,3])                             #Plot Q2    [l/s]
plt.ylabel('Q2 [l/s]')
plt.xlabel('Sample')


def calculate_LL(sample):
    P1,Q1,P2,Q2=sample[0],sample[1]/1000.0,sample[2], sample[3]/1000.0          #Flows need to be in m^3/s(devide by 1000)
    C=100
    D=25e-3 #mm
    L=65    #m
    A=10.67/(C**1.852 * D**4.8704)
    LL=((P1-P2)-(A)*L*Q2**1.852)/(A*(Q1**1.852-Q2**1.852))
    return LL

print('Leak location: '+str(round(calculate_LL(np.mean(data,axis=0)),2))+'m')
print('Leak Size: '+str(round(np.mean(data[:,1])-np.mean(data[:,3]),3))+'l/s')      #QL=Q1-Q2

