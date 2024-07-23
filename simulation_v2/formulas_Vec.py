import matplotlib.pyplot as plt

import math
from typing import List, Tuple, Union


import os
import sys

import numpy as np





# from experiments import k_rep, L0, Dr, sigma, sigma_i, epsilon, Dp, dt, K1, K2, Uc, Umax, omegamax, alpha, beta, gamma, Umax_prey, omegamax_prey, kappa
########## CONSTANTS #############
k_rep    = 2.0                   # Boundary avoidance strength coefficient
L0       = 0.5                   # Avoidance vector relaxation threshold 
Dr       = 0.5                   # Boundary perception radius  
# sigma    = 0.35                  # sigma from tugay 0.7. It works with 0.1 in 10 by 10, it's working with 0.35 in 50 by 50 and N = 100
# sigma    = 1.2                  # sigma from tugay 0.7. It works with 0.1 in 10 by 10
# sigma_i  = np.sqrt(2) * sigma    # Desired distance to neighbors
# sigma_i  = 1.5                   # Desired distance to neighbors
epsilon  = 12.0                  # Strength of the repulsive potential
# Dp       = 5.0                   # Neighbors perception radius val from paper 2.0
dt       = 0.05                  # time step 0.05
K1       = 0.5                   # linear speed gain (table 3)
K2       = 0.05                  # angular speed gain (table 3)
Uc       = 0.05                  # constant speed addition (units/time) (table 2)
Umax     = 0.15                  # maximum linear speed (table 2) val from paper 0.15
omegamax = np.pi/3.0             # maximum angular speed (rad/time) (table 3) in paper is pi/3
alpha    = 2.0                   # for p vector neighbor
beta     = 0.0                   # for h vector heading
gamma    = 1.0                   # for r vector boundary
##################################
Umax_prey = 0.15                 # maximum linear speed for prey (I initially set it to 0.08, also taking into account Tugays code) with equal umax they get caught
alpha_prey = 1.0                 # for p vector neighbor for prey
omegamax_prey = np.pi/3.0        # maximum angular speed for prey (sane angular speed gets caught)
kappa = 3.0                      # for repultion from predator
##################################


##################################
# inside formula functions
# params for non-sensing agents < 99%
Dp       = 3.0                   # 
sigma_i_predator = 0.7
sigma_i_pred_non_sensing = 0.75
# params for sensing agents = 99%
# Dp       = 2.0                   # 
# sigma_i_predator = 0.7
# sigma_i_pred_non_sensing = 0.6
# params for P vector DM
sigma_i_pred_DM = 1.4             # 
sigma_i_pred_non_sensing_DM = 1.7 # 
##################################
Dp_pm    = 3.0                   # Neighbors perception radius val from paper 2.0
Dp_prey  = 3.0                   # Neighbors perception radius val from paper 2.0
# INITIAL SIGMA THIS IS ONLY FOR PLACING THE AGENTS
sigma_i  = 2.5                   # Used for initial placement of agents only once
sigma_i_prey = 0.7               # Used for computing p vector for preys
##################################

##########################################
# NORMAL P VECTOR 99% NON SENSING AGENTS:
# sigma_i = 1.5 INITIAL
# simga_i non sensing = 0.6
# sigma_i sensing = 0.7
# Dp = 2.0
# Else: sigma_i = 0.7 for sensing agents and 0.75 for non sensing agents and Dp = 3.0
# HOWEVER WITH P VECTOR DM:
# sigma_i = 1.5 INITIAL
# simga_i non sensing = 1.7
# sigma_i sensing = 1.4
# Dp = 3.0
# IT WORKS FOR ALL CASES EVEN 99% NON SENSING AGENTS
# sensor_range  = 3 preys
# sensing_range = 6 predators
# min_distance  = 7
##########################################


def get_distance_matrix(agents: np.array) -> np.array:
    positions_x = agents[:, 0]
    positions_y = agents[:, 1]
    N = len(positions_x)
    
    distance_matrix = np.linalg.norm(agents[:, np.newaxis, :2] - agents[np.newaxis, :, :2], axis=-1) # N x N matrix
    np.fill_diagonal(distance_matrix, np.inf) 

    return distance_matrix

def get_angle_matrix(agents: np.array) -> np.array:
    positions_x = agents[:, 0]
    positions_y = agents[:, 1]
    N = len(positions_x)
    
    angles = np.arctan2(positions_y[:, np.newaxis] - positions_y[np.newaxis, :], positions_x[:, np.newaxis] - positions_x[np.newaxis, :]) # N x N matrix
    np.fill_diagonal(angles, np.inf) 

    return angles

def p_vector(agents: np.array, distance_swarm_prey: np.array) -> np.array:
    
    
    positions = agents[:, :2]
    N = len(agents)

    if agents.shape == (N, 5): # predators
        sigma_i = 1.0
        no_distance_sensor_agents = agents[:, 3] == 0
        sigma_i = np.tile(sigma_i, (N,N))
        sigma_i[no_distance_sensor_agents] *= sigma_i_pred_non_sensing
        sigma_i[~no_distance_sensor_agents] *= sigma_i_predator
        # replace distance with 0 if agent is not a distance sensor
        distance_swarm_prey = np.where(agents[:, 3][:, np.newaxis] == 0, 0, distance_swarm_prey)
        # Keeo this in mind!
        # sigma_i += (distance_swarm_prey) * np.diag(sigma_i) 
        sigma_i += (distance_swarm_prey/2) * np.diag(sigma_i) 
        dist_matrix = get_distance_matrix(agents)
        phi = get_angle_matrix(agents)
        mask = dist_matrix <= Dp

        distance = dist_matrix[mask]
        phi_im = phi[mask]
        sigma_i = sigma_i[mask]

        px = -epsilon * ((2.0 * sigma_i**4) / (distance**5) - (sigma_i**2) / (distance**3)) * np.cos(phi_im)
        py = -epsilon * ((2.0 * sigma_i**4) / (distance**5) - (sigma_i**2) / (distance**3)) * np.sin(phi_im)

        p_vectors = np.zeros((N,N,2))
        p_values = np.stack((px, py), axis=-1)
        p_vectors[mask] = p_values
        return np.sum(p_vectors, axis=0) # N x 2

    else:
        sigma    = sigma_i_prey
        sigma_i = sigma
        sigma_i = np.tile(sigma_i, (N,N))
        dist_matrix = get_distance_matrix(agents)
        phi = get_angle_matrix(agents)
        mask = dist_matrix <= Dp_prey

        distance = dist_matrix[mask]
        phi_im = phi[mask]
        sigma_i = sigma_i[mask]
        px = -epsilon * ((2.0 * sigma_i**4) / (distance**5) - (sigma_i**2) / (distance**3)) * np.cos(phi_im)
        py = -epsilon * ((2.0 * sigma_i**4) / (distance**5) - (sigma_i**2) / (distance**3)) * np.sin(phi_im)

        p_vectors = np.zeros((N,N,2))
        p_values = np.stack((px, py), axis=-1)
        p_vectors[mask] = p_values
        return np.sum(p_vectors, axis=0) # N x 2



def p_vector_DM(agents: np.array, distance_swarm_prey: np.array) -> np.array:

    positions = agents[:, :2]
    N = len(agents)
    sigma_i = 1.0
    no_distance_sensor_agents = agents[:, 3] == 0
    # print(no_distance_sensor_agents)
    sigma_i = np.tile(sigma_i, (N,N))
    sigma_i[no_distance_sensor_agents]  *= sigma_i_pred_non_sensing_DM
    sigma_i[~no_distance_sensor_agents] *= sigma_i_pred_DM
    # replace distance with 0 if agent is not a distance sensor
    distance_swarm_prey = np.where(agents[:, 3][:, np.newaxis] == 0, 0, distance_swarm_prey)
    sigma_i += (distance_swarm_prey) * np.diag(sigma_i) #+ 1.e-6

    dist_matrix = get_distance_matrix(agents)
    phi = get_angle_matrix(agents)
    mask = dist_matrix <= Dp_pm 

    distance = dist_matrix[mask]
    phi_im = phi[mask]
    sigma_i = sigma_i[mask]
    # if agents.shape == (N, 5): print(sigma_i)
    # -(epsilon / 2) * ((sigma_j / distance)**(2)-(sigma_j / distance)**(1/2))
    px = -((epsilon /1.0 )* ( (sigma_i / distance ) - np.sqrt( sigma_i / distance ))) * np.cos(phi_im)
    py = -((epsilon /1.0 )* ( (sigma_i / distance ) - np.sqrt( sigma_i / distance ))) * np.sin(phi_im)
    # px = -(epsilon / 5) * ((sigma_i / distance)**3 - (sigma_i / distance)**(1/3)) * np.cos(phi_im)
    # py = -(epsilon / 5) * ((sigma_i / distance)**3 - (sigma_i / distance)**(1/3)) * np.sin(phi_im)

    p_vectors = np.zeros((N,N,2))
    p_values = np.stack((px, py), axis=-1)
    p_vectors[mask] = p_values
    return np.sum(p_vectors, axis=0) # N x 2


def repulsion_predator(preys: np.array, predators: np.array, sensing_range: float) -> np.array:
    positions_preys = preys[:, :2]
    positions_predators = predators[:, :2]
    N = len(preys)
    M = len(predators)

    distance_matrix = np.linalg.norm(positions_preys[:, np.newaxis] - positions_predators[np.newaxis], axis=-1) # N x M matrix
    closest_predator_index = np.argmin(distance_matrix, axis=1)
    closest_predator_position = positions_predators[closest_predator_index]
    closest_predator_distance = distance_matrix[np.arange(N), closest_predator_index]

    dx = positions_preys[:, 0] - closest_predator_position[:, 0]  # Change direction
    dy = positions_preys[:, 1] - closest_predator_position[:, 1]  # Change direction

    mask = closest_predator_distance <= sensing_range
    repulsion_vector = np.zeros((N, 2))
    repulsion_vector[mask, 0] = dx[mask] #/ np.linalg.norm([dx[mask], dy[mask]])
    repulsion_vector[mask, 1] = dy[mask] #/ np.linalg.norm([dx[mask], dy[mask]])

    # average repulsion vector
    repulsion_vector = np.mean(repulsion_vector, axis=0)

    return repulsion_vector



def h_vector(agents: np.array) -> np.array:
    # copy pasted from stefania code, not using it (it does not work for me :(  )

    headings = agents[:, 2]

    # All this is doing is getting the vectorial avg of the headings
    alignment_coss = np.sum(np.cos(headings))
    alignment_sins = np.sum(np.sin(headings))
    alignment_angs = np.arctan2(alignment_sins, alignment_coss)
    alignment_mags = np.sqrt(alignment_coss**2 + alignment_sins**2)

    h_vectors = np.zeros((len(agents), 2))

    h_x = alignment_mags * np.cos(alignment_angs - headings)
    h_y = alignment_mags * np.sin(alignment_angs - headings)

    return np.stack((h_x, h_y), axis=0).T




def r_vector(agent: np.ndarray, boundaries: Tuple[int, int]) -> np.ndarray:

    if boundaries != [0, 0]:
        boundaryX, boundaryY = boundaries
        position_ix = agent[:, 0]
        position_iy = agent[:, 1]
        angle_i = agent[:, 2]
        r_vector = np.zeros((len(agent), 2))

        boundary_x = np.array([0, boundaryX])
        boundary_y = np.array([0, boundaryY])

        # Compute distances to the four boundaries
        dists = np.concatenate([np.abs(position_ix[:, np.newaxis] - boundary_x), np.abs(position_iy[:, np.newaxis] - boundary_y)], axis=-1) # N x 4
        # print("Distances:", dists)

        # compute r vector for each agent and add it to the r_vector matrix
        for i, ag in enumerate(agent):
            for j, dist in enumerate(dists[i]):
                if dist < Dr:
                    # Compute angle based on which boundary we're closest to
                    Lb = dist
                    if Lb == 0:
                        Lb += 0.5
                    angle = [np.pi, 0, np.pi / 2, -np.pi / 2][j]

                    pb_i = np.array([np.cos(angle), np.sin(angle)])

                    # if vertical boundary
                    if j < 2:
                        rb_i = -k_rep * ((1.0 / Lb) - (1.0 / L0)) * (pb_i / (Lb**3))
                    # if horizontal boundary
                    else:
                        rb_i = k_rep * ((1.0 / Lb) - (1.0 / L0)) * (pb_i / (Lb**3))
                    r_vector[i] += rb_i
        return r_vector 
    else:
        return np.zeros((len(agent), 2))