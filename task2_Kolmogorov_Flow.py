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
    def __init__(self, id, position, velocity, pressure, numberdensity, cutoff_radius):
        self.id = id
        self.position = position #postion of the particle
        self.velocity = velocity #velocity of the particle
        self.pressure = pressure #the pressure of the particle
        self.numberdensity = numberdensity #The nunber denisty of the particle
        self.cutoff_radius = cutoff_radius #The cutoff radius
    
    #Automatically calculate the index of cell the particle belongs to
    @property
    def cellindex(self):
        return cell_of_particle_fun(self, self.cutoff_radius) #need to know what the cutoff_radius is

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

    for i in range(len(particle_ref)):
        #First reposition so it is symmetric the grid, then take the periodic distance by taking the modulo and account for shift
        distance_i = (particle_ref[i] - particle_comp[i]+(boundary_max[i]-boundary_min[i])/2)%(boundary_max[i]-boundary_min[i]) - (boundary_max[i]-boundary_min[i])/2
        square_sum_periodic += distance_i**2

    return np.sqrt(square_sum_periodic)


#Periodic boundaries
def r_ij_vector_periodic_fun(particle_ref, particle_comp, boundary_max):
    #INPUT
    #particle_ref and particle_comp are vectors of the positions
    #boundary_max is a list of floats: the values of the boundary (starts at 0)
    #OUTPUT
    #a numpy array: the vector that goes from particle j to i

    #ASSUME xmin = 0  !!
    boundary_min = np.zeros(len(boundary_max))

    r_ij_periodic = []

    for i in range(len(particle_ref)):
        #First reposition so it is symmetric the grid, then take the periodic distance by taking the modulo and account for shift
        r_ij_i = (particle_ref[i] - particle_comp[i]+(boundary_max[i]-boundary_min[i])/2)%(boundary_max[i]-boundary_min[i]) - (boundary_max[i]-boundary_min[i])/2
        #Having calculated i'th index of the vector, we add it to the vector
        r_ij_periodic.append(r_ij_i)

    return np.array(r_ij_periodic)


def position_periodic_fun(position, boundary_max):
    #INPUT
    #position is a list of floats; the vector of the positions
    #boundary_max is a list of floats: the values of the boundary (starts at 0)
    #OUTPUT
    #a list of floats: the position of the particles such that periodic boundaries

    #ASSUME THAT BOUNDARY MIN is zero!
    boundary_min = np.zeros(boundary_max)

    position_periodic = []

    for i in range(len(position)):
        #First shift so boundary min is zero, then see if fall outside of te distance/domain then reshift
        position_periodic.append((position[i]-boundary_min[i])%(boundary_max[i]-boundary_min[i]) + boundary_min[i])
    
    #return a numpy array as input was an np.array
    return np.array(position_periodic)


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


def kernel_W_derivative_rij_fun(r_ij_norm, r_cutoff):
    #INPUT
    #r_ij_norm is float; the distance between particle i and particle j
    #r_cutoff is a float: the cutoff distance between particles
    #OUTPUT
    #A float; The derivative of the kernel for particle_i

    #Calculate the vector FROM j to i and distance
    #r_ij = particle_i.position - particle_j.position

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


def pressure_force_fun(particle_i, particle_j, W_ij_der, e_ij):
    #INPUTS
    #particle_i is of the particle class: particle_i is the one the force is being calculated for
    #particle_j is of the particle class: particle_j is the one that is applying the force
    #W_ij_der is a float: the derivative of the kernel function for the particles i and j
    #e_ij is a vector/numpy array: the unit vector that points from particle j to particle i
    #OUTPUTS
    #a vector/numpy array; the pressure force that particle j is applying on particle i

    pressure_force = (particle_i.pressure/(particle_i.numberdensity**2) + particle_j.pressure/(particle_j.numberdensity**2))*(W_ij_der*e_ij)
    
    return pressure_force


#UNSURE: what is v_ij ???
def viscousity_force_fun(particle_i, particle_j, W_ij_der, r_ij_norm, e_ij, eta):
    #INPUTS
    #particle_i is of the particle class: particle_i is the one the force is being calculated for
    #particle_j is of the particle class: particle_j is the one that is applying the force
    #W_ij_der is a float: the derivative of the kernel function for the particles i and j
    #r_ij_norm is a float: the distance between particles i and j
    #e_ij is a vector/numpy array: the unit vector that points from particle j to particle i
    #eta is a float: the viscousity
    #OUTPUTS
    #a vector/numpy array of floats; the viscous force the particle j applies on particle i

    C = 8 #For 2D!

    viscousity_force = C*eta*(W_ij_der/(r_ij_norm*particle_i.numberdensity*particle_j.numberdensity))*e_ij*(np.dot(e_ij, particle_i.velocity))

    return viscousity_force


def external_force_fun(particle_i, boundary_max, F_0):
    #INPUTS
    #particle_i is of the particle class: the particle that the force is applied to
    #boundary_max is a list: the maximum values of the boundary
    #F_0 is a float: the force amplitude
    #OUPUTS
    #A vector/np.array: the external sinusoidal force applied on the particle_i

    #Assume boundary minimum is zero!
    boundary_min = np.zeros

    L = boundary_max-boundary_min
    vector_non_scaled = [np.sin(2*np.pi/L[0]*particle_i.position), 0] #ASSUMED 2D!

    exclusion_force = F_0*np.array(vector_non_scaled)

    return exclusion_force



def total_force_fun(particle_i, particle_j_list, boundary_max, r_cutoff, eta, F_0):
    #INPUTS
    #particle_i is of the particle class: particle_i is the one the force is being calculated for
    #particle_j_list is a list of the particle class: all the particles that are considered close (exhibit a force)
    #boundary_max is a list of floats; the maximum values of the boundary that the simulation is bounded to
    #r_cutoff is a float; the cutoff distance
    #eta is a float; the viscousity
    #F_0 is float; the force amplitude
    #OUTPUT
    #returns a float; the pressure force that particle i experiences
    
    force_pressure = 0
    force_viscousity = 0

    for j_index in range(len(particle_j_list)):
        #Calculate the length
        r_ij_norm = calculate_distance_fun(particle_i, particle_j_list[j_index],  boundary_max)
        #Calculate the vector of the direction of the force of j onto i
        r_ij = r_ij_vector_periodic_fun(particle_i, particle_j_list[j_index], boundary_max)
        #Calculate the corresponding unit vector
        e_ij = r_ij/r_ij_norm

        #Calculate the derivative of the kernel W_ij'
        W_ij_der = kernel_W_derivative_rij_fun(r_ij_norm=r_ij_vector_periodic_fun, r_cutoff=r_cutoff)

        #Calculate the pressure force of part. j on part. i
        force_pressure += pressure_force_fun(particle_i=particle_i, particle_j=particle_j_list[j_index], W_ij_der=W_ij_der, e_ij=e_ij)
        #Calculate the viscous force of part. j on part. i
        force_viscousity += viscousity_force_fun(particle_i=particle_i, particle_j=particle_j_list[j_index], W_ij_der=W_ij_der, r_ij_norm=r_ij_norm, e_ij=e_ij, eta=eta)

    #Calculate the external applied force
    force_external = external_force_fun(particle_i=particle_i, boundary_max=boundary_max, F_0=F_0)

    









######################
### Initialization ###
######################






########################
### Looping function ###
########################







