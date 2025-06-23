# Code for: LANGEVIN DYNAMICS SIMULATION: CHAINS OF MAGNETIC PARTICLES IN A FLUID UNDER ROTATING MAGNETIC FIELDS
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



#The dipole-dipole force amplitude
F_0 = 0.25

#The radius of the particles
a = 0.5


#The radius for which the particles dont account of each other anymore
cutoff_radius = a*6

#The angular velocity
omega = np.pi/3

omega = np.pi/2

omega = 4.689739328080042/2
omega = 4.689739328080042/2*0.95

omega = [np.pi/6, np.pi/4, np.pi/3, np.pi/2, np.pi*3/4, np.pi, np.pi*3/2][4]

omega = np.pi

#The phase
#phi = np.pi/6
phi = 0

#The viscousity
eta = 10**(-5) #Air


#fictional parameters
A = 2
xi = 10

#The mass
m = 10**(-5)

#The timestep
dt = 10**(-4)



#The final time
t_end = dt*10**(4)
#t_end = dt*10**(5)

t_end = 3*dt*10**(4)


#The amount of particles
N = 20
#N = 30


#How often to save the calculated values
saved_iteration = 20
saved_iteration = 100


#################
### FUNCTIONS ###
#################


#We shall use object oriented programming
class particle_class:
    def __init__(self, id, position, velocity):
        self.id = id
        self.position = position #postion is the 
        self.velocity = velocity

#NOT PERIODIC BOUNDARIES
def calculate_distance_fun(particle_ref, particle_comp):
  #the inputs are vectors of the positions
  #OUTPUT
  #a float: the euclidian distance of the vectors
  return np.linalg.norm(particle_comp-particle_ref)

def particles_close_n2_fun(particle_ref, particles, cutoff_radius):
    #particle_ref is a class
    #particles is a list of classes
    #cutoff_radius is a float
    #OUTPUT
    #A list: The particles that are deemed to be closed, i.e., the particles that are within a radius of the cutoff_radius of the reference particle
    particles_close_list = []
    for j in range(len(particles)):
        #Make sure we dont do the calculations on the particle itself
        if particles[j].id != particle_ref.id:
            #calculate the vector FROM j to i
            #r_ij = particle_ref.position - particles[j].position
            r_ij_norm = calculate_distance_fun(particle_ref.position, particles[j].position)

            if r_ij_norm <= cutoff_radius:
                particles_close_list.append(particles[j])
  
    return particles_close_list


def force_magnetic_fun(particle_i, particle_j, b, a=0.5, F_0=2):
    #particle_i and particle_j are particle class
    #b is a vector (list); the magnetic field
    #F_0 and a are floats; the dipole-dipole force amplitude and the radius of the particles respectively
    #OUTPUT
    #A vector (numpy array): the magnetic force on particle i cause by particle j

    #Calculate the vector FROM j to i
    r_ij = particle_i.position - particle_j.position
    r_ij_norm = calculate_distance_fun(particle_i.position, particle_j.position)
    e_ij = r_ij/r_ij_norm

    #Calculate the dimensionless distance between the particles: l ~ 2
    l = r_ij_norm/a

    F_mag = F_0/(l**4)*(2*np.dot(b, e_ij) * b - (5*(np.dot(b, e_ij))**2 - 1)*e_ij)

    return F_mag


def force_viscous_fun(particle, eta=10**(-5), a=0.5):
   #particle is of the particle class
   #eta is a float; the viscousity
   #a is the radius of the particles
   #OUTPUT
   #A vector (numpy array): the viscious force the particle experiences as it moves (background velocity assumed=0)

    F_v = -6*np.pi*eta*a*particle.velocity

    return F_v


def force_excluded_volume_force(particle_i, particle_j, a=0.5, F_0 = 2, A = 2, xi = 10):
    #particle_i and particle_j are variables of the class particle_class
    #a is a float; the radius of the particle
    #F_0 is a float: the dipole-dipole force amplitude
    #A and xi are constants: they dont have physical as this is a fictious force
    #OUTPUT
    #A vector (numpy array): the (fictious) force such that particle i experiences so it doesnt overlap with particle j

    #Calculate the vector FROM j to i
    r_ij = particle_i.position - particle_j.position
    r_ij_norm = calculate_distance_fun(particle_i.position, particle_j.position)
    e_ij = r_ij/r_ij_norm

    #Added a minus to the exponent! (the force needs to decrease the further you go)
    F_ev = A/16*F_0*e_ij*np.exp(-xi*(r_ij_norm/(2*a)-1))

    return F_ev 



######################
### Initialization ###
######################

#Initial position of the particles:
    #put them all on a line
x_start_0 = 0
#x_end_0 = (2+1)*N*a

x_end_0 = (2)*N*a


y_start_0 = 0
y_end_0 = 0

#This makes an array where the first index is the particle and second is coordinate
particles_position_0 = np.linspace((x_start_0, y_start_0), (x_end_0, y_end_0), N)

#The initial velocities
v_x_0 = 0
v_y_0 = 0

#This makes an array where the first index is the particle and second is coordinate
particles_velocity_0 = np.linspace((v_x_0, v_y_0), (v_x_0, v_y_0), N)

#Store a list of all the particles (each a class)
particles_list = []

for i in range(N):
    particle_i = particle_class(id = i,
                                position = particles_position_0[i],
                                velocity = particles_velocity_0[i])
    particles_list.append(particle_i)


saved_particles_positions = [particles_position_0]


################################
### Calculate crit frequency ###
################################

print("The omega chosen:", omega)


l_bar = 2
omega_crit_theory = F_0/(2*np.pi*eta*a**2)*((N-1)/N) * (l_bar)**(-3) * ((1/4)*(N**2 - 1)*l_bar**(2) +4)**(-1)
print("The critical omega according to the theory is:", omega_crit_theory)

mu_0_M_2 = F_0*3/(4*np.pi)*(1/a**2)
omega_crit_36 = 1/12*(mu_0_M_2)/(eta)*(N-1)/(N*(N**2 + 3))
print("The critical omega according to the formula 36 is:", omega_crit_36)




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
        other_particles.pop(i) #Get the list of all the particles (particles are named by their init number) and then remove your own number
        particles_in_range = particles_close_n2_fun(particle_i, other_particles, cutoff_radius=cutoff_radius)

        #Calculate for each particle in range, their force on particle i and sum
        b = np.array([np.cos(omega*current_t + phi), np.sin(omega*current_t + phi)]) #2D formulation
        F_mag = 0
        F_ev = 0
        for particle_j in particles_in_range:
            #Magnetic Force
            F_mag += force_magnetic_fun(particle_i, particle_j, b, a = a, F_0 = F_0)
            #Viscous Force
            F_ev += force_excluded_volume_force(particle_i, particle_j, a = a, F_0 = F_0, A = A, xi = xi)

        #Calculate the viscous force
        F_v = force_viscous_fun(particle_i, eta=eta, a=a)

        F_tot = F_mag + F_ev + F_v
        #F_tot = F_ev + F_v
       
        #Calculate the new position and velocity
        #position: dx/dt = v hence
        # x^{(n+1)} = x^{(n)} + dt*v^{(n)}
        particles_new_positions.append(particle_i.position + dt*particle_i.velocity)
        #Velocity: m dv/dt = F_{tot}
        # v^{(n+1)} = dt*F_{tot}/m + v^{(n)}
        particles_new_velocity.append(particle_i.velocity + dt*F_tot/m)
    
    
    #Calculate the angle between middle of the chain and the magnetic field
    #Select particles in the middle
    particle_left = particles_list[N//2-1] #If even get the middle to left and if uneven then it is the one left of the middle
    particle_right = particles_list[(N+1)//2] #If even get the middle to right and if uneven then it is the one right of the middle

    r_left_right = particle_right.position - particle_left.position
    r_left_right_norm = calculate_distance_fun(particle_left.position, particle_right.position)
    e_left_right = r_left_right/r_left_right_norm

    angle_middle_b = np.arccos(np.dot(e_left_right, b))



    #After iterated over all the particles, can update the parameters
    current_t += dt
    current_iteration += 1
    for i in range(N):
        particles_list[i].position = particles_new_positions[i]
        particles_list[i].velocity = particles_new_velocity[i]
        
        

    #Save values
    if current_iteration%saved_iteration == 0:
        #Save the positions
        saved_particles_positions.append(particles_new_positions)
        saved_times.append(current_t)
        saved_angles.append(angle_middle_b)
    
    
    #Print progress
    if (current_iteration*100)//iterations_required_given_start_end >= progress_percentage:
        print("Progress percentage:", progress_percentage, "%")
        print("Elapsed real time:", round(time.time()-start_time_py, 3), "s \n")
        progress_percentage += 10
    



################
### Plotting ###
################

#xmin = -a*N*2*10
#xmax = a*N*2*10


#ymin = -a*N*2*10
#ymax = a*N*2*10

#Make sure plot the points with the right size #DOESNT WORK
points_whole_ax = 5 * 0.8 * 72    # 1 point = dpi / 72 pixels
points_radius = 2 * a / 1.0 * points_whole_ax



saved_particles_positions_np = np.array(saved_particles_positions)
#Saved_particles has as first index the timeslot, then the second index is the particle and the third the coordinate

xmin = np.min(saved_particles_positions_np[:,:,0])-a
xmax = np.max(saved_particles_positions_np[:,:,0])+a

ymin = np.min(saved_particles_positions_np[:,:,1])-a
ymax = np.max(saved_particles_positions_np[:,:,1])+a





fig, ax = plt.subplots(figsize = (10,6))

#im = ax.scatter(saved_particles_positions_np[0, :, 0], saved_particles_positions_np[0, :, 1], s=points_radius**2)


# Add circles
for xi, yi in zip(saved_particles_positions_np[0, :, 0], saved_particles_positions_np[0, :, 1]):
    circle = Circle((xi, yi), a, edgecolor='red', facecolor='black', alpha=0.5)
    ax.add_patch(circle)

ax.set_xlim(xmin, xmax)
ax.set_ylim(ymin, ymax)

ax.set_title(str("Time = "+ str(saved_times[0])))


ax.set_aspect('equal')  # Important to preserve the circular shape


def animate(i):
    ax.clear()

    #im1 = ax.scatter(saved_particles_positions_np[i, :, 0], saved_particles_positions_np[i, :, 1], s=points_radius**2)
    for xi, yi in zip(saved_particles_positions_np[i, :, 0], saved_particles_positions_np[i, :, 1]):
        circle = Circle((xi, yi), a, edgecolor='red', facecolor='black', alpha=0.5)
        ax.add_patch(circle)

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    ax.set_title(str("Time =" + str(saved_times[i])))

    ax.set_aspect('equal')  # Important to preserve the circular shape



ani = animation.FuncAnimation(fig, animate, len(saved_particles_positions_np), interval=10, blit=False)

fig.tight_layout()
plt.show()


save_animation_boolean = False
if save_animation_boolean:
    # saving to m4 using ffmpeg writer
    writervideo = animation.FFMpegWriter(fps=30)

    #For pi/3 and N=30
    ani.save('animation_langevin_N_'+str(N)+'.mp4', writer=writervideo)


###################
### Plot angles ###
###################

fig, ax = plt.subplots(figsize=(10,6))

ax.plot(saved_times[1:], saved_angles)

ax.set_title('Angle between the middle of the chain and the magnetic field for $\omega$='+str(omega))

ax.set_xlabel('Time')
ax.set_ylabel('Angle')

fig.tight_layout()
plt.show()



#######################################
### Calculate distance consequetive ###
#######################################

mean_distances = []

for time_iter in range(len(saved_particles_positions)):
    sum_time = 0
    for i in range(len(saved_particles_positions[time_iter])-1):
        sum_time += calculate_distance_fun(saved_particles_positions[time_iter][i], saved_particles_positions[time_iter][i+1])
    
    mean_distance = sum_time/len(saved_particles_positions)
    mean_distances.append(mean_distance)

print(mean_distances)


print(np.mean(mean_distances))
    


