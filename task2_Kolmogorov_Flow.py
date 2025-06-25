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
import math

#For physical circle sizes plots
from matplotlib.patches import Circle


##################
### PARAMETERS ###
##################

#Reference density (background density)
rho_0 = 0.1

#The force amplitude
F_0 = 0.5

#The viscousity
eta = 10**(-5) #Air

#The distance we consider for the accounting of their force on the particle
cutoff_radius = 0.3

v_sound = 343 #speed of sound in air

#mass of the particles
mass_particle = 10**(-7)


#Starting distance between particles
dx = 0.1
dy = 0.1

#Amount of particles in the x-direction
N_x = 10
#Amount of particles in the y-direction
N_y = 10
#Total amount of particles
N = N_x*N_y

#Size of the grid
xmin = 0 #ASSUMED FOR ALOT OF FUNCTIONS TO BE ZERO!
ymin = 0 #ASSUMED FOR ALOT OF FUNCTIONS TO BE ZERO!
xmax = (N_x + 1) * dx
ymax = (N_y + 1) * dy

boundary_max = np.array([xmax, ymax])

#Initial position of the particles:
    #put them all on a line
x_start_0 = 0+dx/2
x_end_0 = N_x*dx+dx/2
y_start_0 = 0 + dy/2
y_end_0 = N_y*dy + dy/2


#Get the max index
max_index_x = math.ceil(xmax/cutoff_radius)-1 #if the cutoff_radius is a divider of xmax, then the endpoint will be considered as an index with a cell of length zero (as nothing after)
#For seeing why this makes sense, look at the following example
#Suppose cutoff_radius is 5. For xmax=10, then it gives back 2-1= 1 (which is what we want since 2 cells with first index 0 and second 1)
#   For xmax=7, it gives back 2-1 = 1 (which is what we want as we have a cell with width 5 and 2, but in total 2 cells!) 
max_index_y = math.ceil(ymax/cutoff_radius)-1 



#Timestep
dt = 10**(-4)

#Time end of the simulation
t_end = 3*dt*10**(1)

t_end = dt*100

#How often to save the calculated values
saved_iteration = 1

#################
### FUNCTIONS ###
#################

#We shall use object oriented programming
class particle_class:
    def __init__(self, id, position, velocity, numberdensity, mass):
        self.id = id
        self.position = position #postion of the particle
        self.velocity = velocity #velocity of the particle
        self.numberdensity = numberdensity #The nunber denisty of the particle
        self.mass = mass #The mass of the particle
    
    #Automatically calculate the index of cell the particle belongs to
    @property
    def cellindex(self):
        return cell_of_particle_fun(self) #need to know what the cutoff_radius is

    #@property
    #def numberdensity(self):
    #    return numberdensity_fun()

    @property
    def rho(self):
        return density_fun(self.numberdensity, self.mass)

    @property
    def pressure(self):
        return pressure_fun(self.numberdensity, self.mass, self.rho)
'''
#NOT PERIODIC BOUNDARIES
def calculate_distance_fun(particle_ref, particle_comp):
  #the inputs are vectors of the positions
  #OUTPUT
  #a float: the euclidian distance of the vectors
  return np.linalg.norm(particle_comp-particle_ref)
'''

#PERIODIC BOUNDARIES
def calculate_distance_fun(particle_ref, particle_comp, boundary_max = boundary_max):
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
    boundary_min = np.zeros(len(boundary_max))

    position_periodic = []

    for i in range(len(position)):
        #First shift so boundary min is zero, then see if fall outside of te distance/domain then reshift
        position_periodic.append((position[i]-boundary_min[i])%(boundary_max[i]-boundary_min[i]) + boundary_min[i])
    
    #return a numpy array as input was an np.array
    return np.array(position_periodic)


#Find the cell that the particle is in
def cell_of_particle_fun(particle_ref, cutoff_radius=cutoff_radius):
    #INPUT
    #particle_i is of particle class
    #r_cutoff is a float: the cutoff distance between particles
    #OUTPUT
    #A list with 2 integers; the cell that the particle is in

    index_x = particle_ref.position[0]//cutoff_radius #This gives the closest lowerbound of the cell (x-direction) with index x and width cutoff_radius
    index_y = particle_ref.position[1]//cutoff_radius #This gives the closest lowerbound of the cell (y-direction) with index y and width cutoff_radius

    return [index_x, index_y]

def close_cells_fun(cell_index, max_index_x= max_index_x, max_index_y=max_index_y):
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


def particles_close_fun(particle_i, particle_list):
    #INPUTS
    #particle_i is of particle class; the particle for which we are finding the close particles
    #particle_list is a list of particles of the particle class: all of the particles in the simulation
    #OUTPUT
    #A list of particles of the particle class: the particles that are close to your reference particle

    #Get the neighboring cells
    neighboring_cells = close_cells_fun(particle_i.cellindex)

    particles_neighbor_list = []
    #Check for each particle if in the neighboring cells and if they are, append to the list
    for particle_j in particle_list:
        if particle_j.cellindex in neighboring_cells:
            particles_neighbor_list.append(particle_j)
    
    return particles_neighbor_list
        


def kernel_W_fun(particle_i, particle_j, r_cutoff = cutoff_radius):
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


def kernel_W_derivative_fun(particle_i, particle_j, r_cutoff=cutoff_radius):
    #INPUT
    #particle_i and particle_j are particle class
    #r_cutoff is a float: the cutoff distance between particles
    #OUTPUT
    #A float; The derivative of the kernel for particle_i

    #Calculate the vector FROM j to i and distance
    #r_ij = particle_i.position - particle_j.position
    r_ij_norm = calculate_distance_fun(particle_i.position, particle_j.position)

    C_w_2 = -60/(np.pi*r_cutoff**3) #For 2D

    if r_ij_norm < r_cutoff:
        W_der = C_w_2 * r_ij_norm/r_cutoff*(1-r_ij_norm/r_cutoff)**(2)
    else:
        W_der = 0
    
    return W_der


def kernel_W_derivative_rij_fun(r_ij_norm, r_cutoff=cutoff_radius):
    #INPUT
    #r_ij_norm is float; the distance between particle i and particle j
    #r_cutoff is a float: the cutoff distance between particles
    #OUTPUT
    #A float; The derivative of the kernel for particle_i

    #Calculate the vector FROM j to i and distance
    #r_ij = particle_i.position - particle_j.position

    C_w_2 = -60/(np.pi*r_cutoff**3) #For 2D

    if r_ij_norm < r_cutoff:
        W_der = C_w_2 * r_ij_norm/r_cutoff*(1-r_ij_norm/r_cutoff)**(2)
    else:
        W_der = 0
    
    return W_der



def numberdensity_fun(particle_i, particles_list, r_cutoff= cutoff_radius):
    #INPUTS
    #particle_i is of the particle class: the particle for which the number density is being calculated for
    #particle_list is a list of particle class: all of the particles that are being considered
    #r_cutoff is float: the cutoff radius (everything beyond has W_ij = 0)
    #OUTPUTS
    #a float: the number density of particle_i

    #ASSUMED PARTICLES ARE CLOSE!

    numberdensity = 0
    
    for particle_j in particles_list:
        #2. Get the kernel for all the close particles
        W_ij = kernel_W_fun(particle_i=particle_i, particle_j=particle_j, r_cutoff = r_cutoff)
        #3. Get the number density by summing
        numberdensity += W_ij

    return numberdensity



#UNSURE!
def density_fun(number_density, mass_particle, cutoff_radius = cutoff_radius):
    #INPUT
    #number_density is a float; the number density of the particle (eq.5)
    #mass_particle is a float; the mass of the particle
    #r_cutoff is a float: the cutoff distance between particles
    #OUTPUT
    #the density of the particle
    #ASSUMED 2D!

    #volume = np.pi*cutoff_radius**2
    #rho = number_density*mass_particle/volume #total mass/volume
    rho = number_density*mass_particle

    return rho


def pressure_fun(number_density, mass_particle, rho_ref, v_sound=v_sound, cutoff_radius=cutoff_radius):
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

    if r_ij_norm != 0:
        #make sure it isnt the same particle as otherwise divide by zero!
        viscousity_force = C*eta*(W_ij_der/(r_ij_norm*particle_i.numberdensity*particle_j.numberdensity))*e_ij*(np.dot(e_ij, particle_i.velocity))
    else:
        #when it is the same particle, then e_ij is zero so in the limit equals to zero
        viscousity_force = 0
    return viscousity_force


def external_force_fun(particle_i, boundary_max = boundary_max, F_0 = F_0):
    #INPUTS
    #particle_i is of the particle class: the particle that the force is applied to
    #boundary_max is a list: the maximum values of the boundary
    #F_0 is a float: the force amplitude
    #OUPUTS
    #A vector/np.array: the external sinusoidal force applied on the particle_i

    #Assume boundary minimum is zero!
    boundary_min = np.zeros(len(boundary_max))

    L = boundary_max-boundary_min
    vector_non_scaled = [np.sin(2*np.pi/L[0]*particle_i.position[1]), 0] #ASSUMED 2D!

    exclusion_force = F_0*np.array(vector_non_scaled)

    return exclusion_force



def total_force_fun(particle_i, particle_j_list, boundary_max = boundary_max, eta = eta, F_0 = F_0, r_cutoff = cutoff_radius):
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
        r_ij_norm = calculate_distance_fun(particle_i.position, particle_j_list[j_index].position,  boundary_max)
        #Calculate the vector of the direction of the force of j onto i
        r_ij = r_ij_vector_periodic_fun(particle_i.position, particle_j_list[j_index].position, boundary_max)
        #Calculate the corresponding unit vector
        if r_ij_norm != 0:
            #Make sure we arent dividing by zero
            e_ij = r_ij/r_ij_norm
        else:
            e_ij = r_ij #the r_ij will be the zero vector of the right size


        #Calculate the derivative of the kernel W_ij'
        W_ij_der = kernel_W_derivative_rij_fun(r_ij_norm=r_ij_norm, r_cutoff=r_cutoff)

        #Calculate the pressure force of part. j on part. i
        force_pressure += pressure_force_fun(particle_i=particle_i, particle_j=particle_j_list[j_index], W_ij_der=W_ij_der, e_ij=e_ij)
        #Calculate the viscous force of part. j on part. i
        force_viscousity += viscousity_force_fun(particle_i=particle_i, particle_j=particle_j_list[j_index], W_ij_der=W_ij_der, r_ij_norm=r_ij_norm, e_ij=e_ij, eta=eta)

    #Calculate the external applied force
    force_external = external_force_fun(particle_i=particle_i, boundary_max=boundary_max, F_0=F_0)

    F_total = force_pressure + force_viscousity + force_external

    return F_total






######################
### Initialization ###
######################



#This makes an array where the first index is the particle and second is coordinate

particles_position_0_x = np.linspace(x_start_0, x_end_0, N_x)
particles_position_0_y = np.linspace(y_start_0, y_end_0, N_y)


#The initial velocities
v_x_0 = 0
v_y_0 = 0

#This makes an array where the first index is the particle and second is coordinate
particles_velocity_0_x = v_x_0*np.ones(len(particles_position_0_x))
particles_velocity_0_y = v_y_0*np.ones(len(particles_position_0_y))




#Store a list of all the particles (each a class)
particles_list = []

for i in range(N_y):
    for j in range(N_x):
        #The index of the particle is the number of the particle: we first go over all the x particles and then the we go the next y_row
        #So for the particle that is in the second row, first column it would be =#amount of particles in the row (as we start with index zero)
        particle_i = particle_class(id = i*N_x + j,
                                    position = np.array([particles_position_0_x[j], particles_position_0_y[i]]),
                                    velocity = np.array([particles_velocity_0_x[j], particles_velocity_0_y[i]]),
                                    numberdensity = 1, #Not yet calculated as we havent created all the particles yet
                                    mass=mass_particle)
        particles_list.append(particle_i)



#Calculate the correct numberdensities now
for particle_i in particles_list:
    particles_i_close = particles_close_fun(particle_i, particles_list)
    numberdensity_fun(particle_i, particles_i_close, cutoff_radius)

particles_position_0 = []
for particle_i in particles_list:
    particles_position_0.append(particle_i.position)

saved_particles_positions = [particles_position_0]






########################
### Looping function ###
########################



current_t = 0

saved_times = [current_t]
saved_angles = []

current_iteration = 0

iterations_required_given_start_end = (t_end-current_t)/dt
progress_percentage = 0

start_time_py = time.time()




while current_t < t_end:
    particles_new_positions = []
    particles_new_velocity = []
    #Loop for all the particles
    for i in range(len(particles_list)):
        particle_i = particles_list[i]
        #Calculate which particles influence our selected particle i
        #First get all the other particles as dont want to count themselves
        other_particles = particles_list.copy()
        #other_particles.pop(i) #Get the list of all the particles (particles are named by their init number) and then remove your own number
        particles_in_range = particles_close_fun(particle_i, other_particles)

        F_tot = total_force_fun(particle_i=particle_i, particle_j_list=particles_in_range, boundary_max=boundary_max, eta=eta, F_0=F_0, r_cutoff=cutoff_radius)
        #F_tot = F_ev + F_v
       
        #Calculate the new position and velocity
        #position: dx/dt = v hence
        # x^{(n+1)} = x^{(n)} + dt*v^{(n)}
        particle_new_position_i = particle_i.position + dt*particle_i.velocity
        particles_new_positions.append(position_periodic_fun(particle_new_position_i, boundary_max))
        #Velocity: m dv/dt = F_{tot}
        # v^{(n+1)} = dt*F_{tot}/m + v^{(n)}
        particles_new_velocity.append(particle_i.velocity + dt*F_tot/particle_i.mass)

    
    
    


    #After iterated over all the particles, can update the parameters
    current_t += dt
    current_iteration += 1
    for i in range(N):
        #Make sure position remains in grid
        particles_list[i].position = particles_new_positions[i]
        particles_list[i].velocity = particles_new_velocity[i]

        #print(particles_list[i].position)

        if np.isnan(np.sum(particles_list[i].position)):
            print("GOT A NAN at iteration:", current_iteration)
            print(particles_list[i].position)
            print(particles_list[i].velocity)
            print()
        if np.isnan(np.sum(particles_list[i].velocity)):
            print("GOT A NAN at iteration:", current_iteration)
            print(particles_list[i].position)
            print(particles_list[i].velocity)
            print()
        
    #Calculate the correct numberdensities now
    for particle_i in particles_list:
        particles_i_close = particles_close_fun(particle_i, particles_list)
        numberdensity_fun(particle_i, particles_i_close, cutoff_radius)

        

    #Save values
    if current_iteration%saved_iteration == 0:
        #Save the positions
        saved_particles_positions.append(particles_new_positions)
        saved_times.append(current_t)
    
    
    #Print progress
    if (current_iteration*100)//iterations_required_given_start_end >= progress_percentage:
        print("Progress percentage:", progress_percentage, "%")
        print("Elapsed real time:", round(time.time()-start_time_py, 3), "s \n")
        progress_percentage += 10
    



################
### Plotting ###
################

saved_particles_positions_np = np.array(saved_particles_positions)

fig, ax = plt.subplots(figsize=(10,6))

im = ax.scatter(saved_particles_positions_np[0, :, 0], saved_particles_positions_np[0, :, 1], s=2)

ax.set_xlim(xmin, xmax)
ax.set_ylim(ymin, ymax)

ax.set_title(str("Time = "+ str(saved_times[0])))


def animate(i):
    ax.clear()

    im1 = ax.scatter(saved_particles_positions_np[i, :, 0], saved_particles_positions_np[i, :, 1], s=2)

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    ax.set_title(str("Time =" + str(saved_times[i])))

    ax.set_aspect('equal')  # Important to preserve the circular shape



ani = animation.FuncAnimation(fig, animate, len(saved_particles_positions_np), interval=10, blit=False)

fig.tight_layout()
plt.show()

