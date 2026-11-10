from qiskit import QuantumCircuit
import numpy as np


def cnot(qc: QuantumCircuit, qarg_1, qarg_2):
    qc.h(qarg_2)
    qc.cz(qarg_1, qarg_2)
    qc.h(qarg_2)

def triangle_circ(circ,angle,q0,q1,q2,q3,precap=True,endcap=True):
    if precap:
        cnot(circ,q3,q2)
    cnot(circ,q2,q3)
    cnot(circ,q3,q2)
    cnot(circ,q2,q1)
    cnot(circ,q0,q2)
    circ.rzz(angle,q1,q2)
    circ.rz(angle,q1)
    circ.rz(angle,q2)
    cnot(circ,q0,q2)
    cnot(circ,q2,q1)
    cnot(circ,q3,q2)
    cnot(circ,q2,q3)
    if endcap:
        cnot(circ,q3,q2)

def nnn_rzz(circ,angle,q0,q1,q2):
    cnot(circ,q1,q2)
    cnot(circ,q0,q1)
    circ.rzz(angle,q1,q2)
    cnot(circ,q0,q1)
    cnot(circ,q1,q2)

def evolve_ZZ(circ,dt, Lx,Ly):

    if Lx==2 and Ly==2:
        circ.cx(7,15)
        triangle_circ(circ, 2*dt, 0,2,14,3,True,False) # first "up" triangle, but I think the env. qubit is the one out to the left??
        triangle_circ(circ, 2*dt, 1,4,16,5,True,False) 
        triangle_circ(circ, 2*dt, 6,8,21,9,True,False)
        triangle_circ(circ, 2*dt, 7,10,19,11,True,False)
        
        triangle_circ(circ, 2*dt, 3,4,15,7,False,True)
        circ.cx(3,14)
        circ.cx(5,16)
        circ.cx(9,21)
        circ.cx(11,19)
        nnn_rzz(circ,2*dt,2,18,6)
        nnn_rzz(circ,2*dt,9,22,10)
        
        
    if Lx==3 and Ly==3:
                        
        nnn_rzz(circ,2*dt,1,37,2) 
        nnn_rzz(circ,2*dt,3,39,4) 
        nnn_rzz(circ,2*dt,6,45,9) 
        nnn_rzz(circ,2*dt,17,51,23) 
        nnn_rzz(circ,2*dt,24,57,27) 
        
        
        circ.cx(6, 42)
        circ.cx(7, 38)
        circ.cx(8, 40)
        circ.cx(15, 46)
        circ.cx(16, 47)
        circ.cx(17, 48)
        circ.cx(24, 53)
        circ.cx(25, 54)
        circ.cx(26, 56)
        circ.cx(33, 58)
        circ.cx(34, 60)
        circ.cx(35, 62)
        
        triangle_circ(circ, 2*dt, 7,10,43,11, True, False)
        triangle_circ(circ, 2*dt, 8,12,44,13, True, False)
        triangle_circ(circ, 2*dt, 15,19,50,20, True, False)
        triangle_circ(circ, 2*dt, 16,21,55,22, True, False)
        triangle_circ(circ, 2*dt, 25,28,59,29, True, False)
        triangle_circ(circ, 2*dt, 26,30,61,31, True, False)
        
        triangle_circ(circ, 2*dt, 0,1,42,6, False, True)
        triangle_circ(circ, 2*dt, 2,3,38,7, False, True)
        triangle_circ(circ, 2*dt, 4,5,40,8, False, True)
        triangle_circ(circ, 2*dt, 9,10,46,15, False, True)
        triangle_circ(circ, 2*dt, 11,12,47,16, False, True)
        triangle_circ(circ, 2*dt, 13,14,48,17, False, True)
        triangle_circ(circ, 2*dt, 18,19,53,24, False, True)
        triangle_circ(circ, 2*dt, 20,21,54,25, False, True)
        triangle_circ(circ, 2*dt, 22,23,56,26, False, True)
        triangle_circ(circ, 2*dt, 27,28,58,33, False, True)
        triangle_circ(circ, 2*dt, 29,30,60,34, False, True)
        triangle_circ(circ, 2*dt, 31,32,62,35, False, True)
        
        circ.cx(11, 43)
        circ.cx(13, 44)
        circ.cx(20, 50)
        circ.cx(22, 55)
        circ.cx(29, 59)
        circ.cx(31, 61)
            