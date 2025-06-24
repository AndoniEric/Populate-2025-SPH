# Code for: SPH SIMULATION: KOLMOGOROV FLOW
#In the context of Populate 2025
#By Anna Macaluso, Tran Thi Quynh Anh and Eric Andoni
#Guided by Adolfo Vazquez Quesada & Christophe Henry


##############
### IMPORT ###
##############

import numpy as np
import matplotlib.pyplot as plt
#For animations
import matplotlib.animation as animation
import time

#For physical circle sizes plots
from matplotlib.patches import Circle


##################
### PARAMETERS ###
##################






#################
### FUNCTIONS ###
#################

#We shall use object oriented programming
class particle_class:
    def __init__(self, id, position, velocity):
        self.id = id
        self.position = position #postion is the 
        self.velocity = velocity

'''
#NOT PERIODIC BOUNDARIES
def calculate_distance_fun(particle_ref, particle_comp):
  #the inputs are vectors of the positions
  #OUTPUT
  #a float: the euclidian distance of the vectors
  return np.linalg.norm(particle_comp-particle_ref)
'''

#PERIODIC BOUNDARIES
def calculate_distance_fun(particle_ref, particle_comp, boundary_max):
    #INPUT
    #particle_ref and particle_comp are vectors of the positions
    #boundary_max is a list of floats: the values of the boundary (starts at 0)
    #OUTPUT
    #a float: the euclidian distance of the vectors

    #ASSUME xmin = 0  !!
    boundary_min = np.zeros(len(boundary_max))

    square_sum_periodic = 0

    for i in range(len(particle_ref.position)):
        #First reposition so it is symmetric the grid, then take the periodic distance by taking the modulo and account for shift
        distance_i = (particle_comp[i] - particle_ref.position[i]+(boundary_max[i]-boundary_min[i])/2)%(boundary_max[i]-boundary_min[i]) - (boundary_max[i]-boundary_min[i])/2
        square_sum_periodic += distance_i**2

    return np.sqrt(square_sum_periodic)



#Find the cell that the particle is in
def cell_of_particle_fun(particle_ref, cutoff_radius):
    #INPUT
    #particle_i is of particle class
    #r_cutoff is a float: the cutoff distance between particles
    #OUTPUT
    #A list with 2 integers; the cell that the particle is in

    index_x = particle_ref.position[0]//cutoff_radius #This gives the closest lowerbound of the cell (x-direction) with index x and width cutoff_radius
    index_y = particle_ref.position[1]//cutoff_radius #This gives the closest lowerbound of the cell (y-direction) with index y and width cutoff_radius

    return [index_x, index_y]

def close_cells_fun(cell_index, max_index_x, max_index_y):
    #INPUT
    #cell_index is a list with 2 integers: the indeci of the cell
    #max_index_x and max_index_y are integers: the largest possible indeci that the cells can have
    #OUTPUT
    #a list of lists each with 2 integers: a list of the all the cells (given by their indici) that are next to the cell,
    #INCLUDES THE CELL ITSELF
    #Accounts for periodic boundary conditions!

    close_cells = []

    for i in [-1,0,1]:
        for j in [-1,0,1]:
            #if (i == j) and i != 0:
                #This is the cell itself

            #We take modulo as if index is larger than the max_index, then it should loop back (so if one larger than max index it should be zero). 
            #It also works with negative values as -1 will be placed at the highest index
            x_index = (cell_index[0]+i)%(max_index_x+1) 
            y_index = (cell_index[1]+j)%(max_index_y+1)

            close_cells.append([x_index, y_index])
    
    return close_cells



def kernel_W_fun(particle_i, particle_j, r_cutoff):
    #INPUT
    #particle_i and particle_j are particle class
    #r_cutoff is a float: the cutoff distance between particles
    #OUTPUT
    #A float; The kernel for particle_i

    #Calculate the vector FROM j to i and distance
    #r_ij = particle_i.position - particle_j.position
    r_ij_norm = calculate_distance_fun(particle_i.position, particle_j.position)

    C_w = 5/(np.pi*r_cutoff**2) #For 2D

    if r_ij_norm < r_cutoff:
        W = C_w * (1+3*r_ij_norm/r_cutoff)*(1-r_ij_norm/r_cutoff)**(3)
    else:
        W = 0
    
    return W


def kernel_W_derivative_fun(particle_i, particle_j, r_cutoff):
    #INPUT
    #particle_i and particle_j are particle class
    #r_cutoff is a float: the cutoff distance between particles
    #OUTPUT
    #A float; The derivative of the kernel for particle_i

    #Calculate the vector FROM j to i and distance
    #r_ij = particle_i.position - particle_j.position
    r_ij_norm = calculate_distance_fun(particle_i.position, particle_j.position)

    C_w_2 = -60/(np.pi*r_ij_norm**3) #For 2D

    if r_ij_norm < r_cutoff:
        W_der = C_w_2 * r_ij_norm/r_cutoff*(1-r_ij_norm/r_cutoff)**(2)
    else:
        W_der = 0
    
    return W_der

#UNSURE!
def density_fun(number_density, mass_particle, cutoff_radius):
    #INPUT
    #number_density is a float; the number density of the particle (eq.5)
    #mass_particle is a float; the mass of the particle
    #r_cutoff is a float: the cutoff distance between particles
    #OUTPUT
    #the density of the particle
    #ASSUMED 2D!

    volume = 2*np.pi*cutoff_radius**2
    rho = number_density*mass_particle/volume #total mass/volume

    return rho


def pressure_fun(number_density, mass_particle, cutoff_radius, rho_ref, v_sound=343):
    #INPUT
    #number_density is a float; the number density of the particle (eq.5)
    #mass_particle is a float; the mass of the particle
    #r_cutoff is a float: the cutoff distance between particles
    #rho_ref is a float: the reference mass density (typically lower than the expected average mass density)
    #v_sound is a float: the velocity of sound (in the medium). Default is the speed of sound in air [m/s]
    #OUTPUT
    #the pressure of the particle

    rho = density_fun(number_density, mass_particle, cutoff_radius)
    P = v_sound**2 * (rho - rho_ref)
    return P

    






######################
### Initialization ###
######################






########################
### Looping function ###
########################







