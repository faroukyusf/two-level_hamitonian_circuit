#!/usr/bin/env python
# coding: utf-8

# In[24]:


import matplotlib.pyplot as plt
import numpy as np
from math import pi
from qiskit import *
from qiskit import QuantumCircuit,execute,Aer,IBMQ
from qiskit import IBMQ, Aer, transpile, assemble,execute
from qiskit import ClassicalRegister, QuantumRegister
from qiskit.visualization import plot_histogram, plot_bloch_multivector
from qiskit.compiler import transpile,assemble


# ##=================== KE function ===================##
# 
# For 3 qubits [q0, q1, q2], we first use q0 & q1 simulantously to undergo the kinetic circuit. 
# 
# Then, we use q1 & q2. The pattern goes on for bigger systems.
# 
# Typically,
# 
# 1) q0 is a control for X and q1 is a traget
# 
# 2) Rz(- pi /2) on q0
# 
# 3) q1 is a control for X and q0 is a traget
# 
# 4) Ry(- hopping angle) on q0
# 
# 5) q1 is a control for X and q0 is a traget
# 
# 6) Ry(+ hopping angle) on q0
# 
# 7) Rz(+ pi/2) on q0
# 
# 8) q0 is a control for X and q1 is a traget
# 
# 
# This makes sense because the kinetic term is just hopping from one location to the next.
# So, in general for n qubits, we will connect the i-th qubit with the i-th + 1 qubit..
# 
# It would be function of qc, hopping angle (hopping parameter h*dt), and the number of qubits.

# In[25]:


def Kinetic_Energy(qc, hopping_angle, num_qubits):
    for i in range(num_qubits - 1):
        qc.cx(i, i + 1)
        qc.rz(-np.pi/2.0, i)
        qc.cx(i + 1, i)
        qc.ry(-hopping_angle, i)
        qc.cx(i + 1, i)
        qc.ry(hopping_angle, i)
        qc.rz(np.pi/2.0, i)
        qc.cx(i, i + 1)

    
psi = QuantumCircuit(3)
Kinetic_Energy(psi, 0.002, 3)
psi.draw(output="mpl")


# ##=================== PE function ===================##
# 
# Acoording to how far the ith site is from origin, the potential acts ~ x^2 
# This fact translates to ((L - 1 - 2 i)/2)^2 being the elements on the diagonal
# 'L' is the total number of sites while 'i' is the site number.
# It would be function of qc, potential_angle = -l*dt (where l = m w^2 /2) and the number of qubits.
# 

# In[26]:


def Potential_Energy(qc, potential_angle, num_qubits): 
    for i in range(num_qubits):
        if i == int ((num_qubits-1)/2):
            continue
        r = (i-((num_qubits-1)/2))
        qc.p(potential_angle*r**2,i)

    
# sanity check
psi=QuantumCircuit(3)
Potential_Energy(psi, -0.001, 3)
psi.draw(output="mpl")


# Now, we aready to write a full time-evolution Hamiltonain quantum circuit
# This Hamiltonian is of one particle in a simple hamrnoic oscillator potential
# The time evoultion of any Hamiltonian is a unitary opertion 
# We are using frist order Lee_Torotter-Suzuki formula 

# In[27]:


##=================== Complete SHO N times ===================##

# 'time_steps' defines the number of time steps 'dt' to be taken 
# so that the system evolves with total time T = time_steps * dt 

def Uintary_Time_Evolution_SHO(qc, hopping_angle, potential_angle, time_steps, num_qubits):
    for i in range(time_steps):
        Kinetic_Energy(qc, hopping_angle*0.5, num_qubits)
        Potential_Energy(qc, potential_angle, num_qubits)
        Kinetic_Energy(qc, hopping_angle*0.5, num_qubits)


# Sanity check. This is a deep circuit
psi = QuantumCircuit(3)
Uintary_Time_Evolution_SHO(psi, 0.002, -0.001, 10, 3)
psi.draw(output="mpl")


# Now, let's add electromagnetic (E1) interaction to this system.
# Here, we are building the Unitary Time Evolution circuit for the E1 transition.
# The idea is to encode the entire electromagnetic field into one qubit, 
# which we refer to as an 'auxiliary qubit.' 
# Why only one? Because it has a narrow spectrum for interaction,
# and the interaction occurs within a cavity, 
# so we are only interested in photon modes of 1 (indicating the presence of a photon) 
# and photon modes of 0 (indicating no photon). It only takes one qubit!
# 
# We are going to sandwich the Unitary Time Evolution (U) circuit of the E1 transition between two X-gates.
# Why? Because using two X-gates transforms our circuit from its current form into the required form. Please check Nillsen&Chung (2nd Ed. Ch 4 section 3) for details. 
# Additionally, we need to apply U once and then U† (the conjugate transpose of U) afterward since everything must remain unitary

# In[28]:


#### ====================================== the E1 circuit  ====================================== ###

# beta, the E1 parameter or photon term (beta = alpha*R*dt = gamma/2)

def E1_circuit(qc, beta, num_qubits):
    for i in range(num_qubits-1):
        R = (i-((num_qubits-2)/2))  #R is the distance [L-(L-1)/2]
        qc.cx(num_qubits-1,i)     
        qc.cx(i, num_qubits-1)     
        #this is the start of the Uc circuit, an X gate.. and the Z-angle could also be pi/2
        qc.rz(-np.pi, i)   
        qc.cx(num_qubits-1, i)
        qc.ry(beta*R, i)
        qc.cx(num_qubits-1, i)
        qc.ry(-beta*R, i)
        qc.rz(np.pi, i)  #could be -pi/2
        qc.cx(i, num_qubits-1)
        #here is the end of the Uc circuit.
        qc.cx(num_qubits-1, i)



# Do U Want To See This? 
psi=QuantumCircuit(3+1) #you gotta add 1 becuase the whole thing is controled by the photon which is the nth+1 qubit
E1_circuit(psi, 0.002, 3)
psi.draw(output="mpl")


# Now, this following function applies the Unitary Time Evolution full Hamiltonian (Kinetic + Potential + E1)
# Let's call it Unitary_SHO_E1 

# In[31]:


##==================## THE FULL SHO + E1 FUNCTION ##==================##

def Unitary_SHO_E1(qc, hopping_angle, potential_angle, time_steps, beta, num_qubits):
    for i in range(time_steps):
        Kinetic_Energy(qc, hopping_angle*0.5, num_qubits-1)
        Potential_Eenergy(qc, potential_angle*0.5, num_qubits-1)
        #photon energy term (E is the photon kinetic energy)
        qc.rz(-E*0.5, num_qubits-1) 
        #photon E1 term 
        E1_circuit(qc, beta, num_qubits)
        qc.rz(-E*0.5, num_qubits-1)
        Potential_Eenergy(qc, potential_angle*0.5, num_qubits-1)
        Kinetic_Energy(qc, hopping_angle*0.5, num_qubits-1)



# In[34]:


#Sanity Check; let's chose some physical parameters to see this system in action
a = 1.0/197.33   #lattice spacing = 1 fermi
w = 8*a          #omega, the harmonic potential frequancy = 8 MeV = 8*a in lattice units
dt = 0.01/w      #time step
E = 1*w*dt       #epsilon, the photon kinetic energy

psi = QuantumCircuit(3)
Unitary_SHO_E1(psi, 0.002, -0.001, 10, 0.002, 2)
psi.draw(output="mpl") 


# In[ ]:




