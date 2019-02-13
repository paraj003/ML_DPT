# -*- coding: utf-8 -*-
## Module containing all definitions necessary to run a quench for the LMG model. Use python 3.
##v1.0 with new defined Hamiltonian with parameters γz and γy
import numpy as np
import os
import h5py
from scipy import linalg as LA
import numpy.matlib

## Class definition to define Hamiltonian
class TFIM_Ham_params:
    def __init__(self,J:float,Γ:float):
        self.J=float(J) #Ising hopping
        self.Γ=float(Γ) #Transverse field
    def paramstr(self):
        #returns a string that contains the parameters of the Hamiltonian
        return 'J_'+str(float(self.J))+',Γ_'+str(float(self.Γ))


####################Function Definitions#############
def Log_Loschmidt_Echo_TFIM(t:float,X0:TFIM_Ham_params,X:TFIM_Ham_params,L:int,η:int):
    #calculates |<ψ(t)|ψ(0)>|**2 and returns this value with |ψ(0)> determined by X0, and Hamiltonian evolution with X
    qarr=(2*np.arange(0,L/2)+0.5*(η-1))*np.pi/L+np.pi/L # array with only positive q values using global variables
    ωqarr= 2*np.sqrt(X.Γ**2+X.J**2-2*X.Γ*X.J*(np.cos(qarr))) #array of energies
    ωq0arr= 2*np.sqrt(X0.Γ**2+X0.J**2-2*X0.Γ*X0.J*(np.cos(qarr)))
    sin2θqarr=2*X.J*np.sin(qarr)/ωqarr
    cos2θqarr=2*(X.Γ-X.J*np.cos(qarr))/ωqarr
    sin2θ0qarr=2*X0.J*np.sin(qarr)/ωq0arr
    cos2θ0qarr=2*(X0.Γ-X0.J*np.cos(qarr))/ωq0arr
    sin2αqarr=sin2θqarr*cos2θ0qarr-cos2θqarr*sin2θ0qarr
    return np.sum(np.log(1-(sin2αqarr**2)*(np.sin(t*ωqarr/2)**2)))


####################Saving Data########################
def arrtostr(tarr):
    ##returns a string with the time array
    return '['+str(tarr[0])+'_'+str(np.divide((tarr[-1]-tarr[0]),(np.size(tarr)-1),out=np.zeros_like((tarr[-1]-tarr[0])), where=np.size(tarr)!=1))+'_'+str(tarr[-1])+']'

def save_data_LEt(paramvals0:TFIM_Ham_params,paramvalsf:TFIM_Ham_params,Λarr,tarr):
    ## saves data in a h5py dictionary
    directory='data/Loschmidt_echo/'
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename=directory+'LogLEt_'+arrtostr(tarr)+'_from_'+paramvals0.paramstr()+'_to_'+paramvalsf.paramstr()+'.hdf5'
    print(filename)
    with h5py.File(filename, "w") as f:
        f.create_dataset("logLEarr", Λarr.shape, dtype=Λarr.dtype, data=Λarr)
        f.create_dataset("tarr", tarr.shape, dtype=tarr.dtype, data=tarr)
        f.close()
    with open(directory+"list_of_logLEt.txt", "a") as myfile:
        myfile.write(filename+ "\n")

