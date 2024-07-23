
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.lines import Line2D
from tqdm import tqdm
import math
import os
import sys
from matplotlib.patches import Wedge
from matplotlib import patches
import matplotlib.transforms as transforms

from typing import List, Tuple, Union

# from agent import planarQuadrator
# from prey import preyPlanarQuadrator
from formulas_vec_gpu import p_vector, h_vector,r_vector, repulsion_predator
from formulas_vec_gpu import k_rep, L0, Dr, sigma_i, epsilon, Dp, dt, K1, K2, Uc, Umax, omegamax, alpha, beta, gamma, Umax_prey, omegamax_prey, kappa
# from matplotlib import collections
from formulas_vec_gpu import p_vector_DM
import random

import cupy as np
# from numba import jit, njit
# from numba import cuda
# np.cuda.set_allocator(np.cuda.MemoryPool().malloc)


class environment_agent_gpu(object):

    def __init__(self,
                 boundaries: Tuple[float, float], 
                 N: int,
                 N_preys: int,
                 no_sensor: float,
                 experiment_id: str,
                 steps: int,
                 save: bool = False,
                 error_range: float = 0.05,
                 sensor_range: float= 5.0,
                 sensing_range: float = 2.0,
                 percentage_no_sensor: float =0.0,
                 draw: bool = False,
                 no_progress_bar: bool = False,
                 filename: str = "",
                 filename_prey: str = "",
                 min_distance: int = 5,
                 blind_spots: bool = False,
                 draw_circles: bool = False,
                 pdm: bool = False) -> None:

        self.boundaries            = boundaries
        self.N                     = N
        self.N_preys               = N_preys
        self.no_sensor             = no_sensor # Percentage of agents with no distance sensor
        self.experiment_id         = experiment_id
        self.steps                 = steps
        self.save                  = save
        self.error_range           = error_range
        self.sensor_range          = sensor_range
        self.sensing_range         = sensing_range
        self.percentage_no_sensor  = percentage_no_sensor
        self.draw                  = draw
        self.no_progress_bar       = no_progress_bar
        self.filename              = f'dataPreyPredator/{self.experiment_id}/'+filename+'.csv'
        self.trajectory            = []
        self.trajectory_prey       = []
        self.min_distance          = min_distance
        self.blind_spots           = blind_spots
        self.draw_circles          = draw_circles
        self.pdm                   = pdm
        self.draw_progress_bar     = tqdm(total=self.steps, desc='Simulation Progress')


        self.predators, self.preys = self.generate_agents_and_preys()
        self.no_sensor_agents(self.no_sensor)

        self.data_to_save          = []

        if self.save:
            self.check_file()

        
    def no_sensor_agents(self, no_sensor: float) -> None:
        # Each row represents an agent: [x, y, theta, sesing_yes/no]
        N = len(self.predators)
        no_sensor_agents = np.ones(N)  # Initially, all sensors are turned on
        
        # Randomly select agents to turn off their sensors
        selected_indices = np.random.choice(N, size=int(no_sensor * N), replace=False)
        no_sensor_agents[selected_indices] = 0
        
        self.predators[:, 3] = no_sensor_agents

    def generate_agents_and_preys(self) -> Tuple[np.array, np.array]:

        # Set the seed for random number generation
        np.random.seed(seed=73)
        boundaryX, boundaryY = self.boundaries
        boundaries = (10, 10) if self.boundaries == [0,0] else (boundaryX, boundaryY) # if boundaries are not defined, then boundless environment
        # Generate 2D agents and preys matrices
        agents = np.ones((self.N, 5))  # Each row represents an agent: [x, y, theta, sesing_yes/no,id]
        preys = np.zeros((self.N_preys, 4))  # Each row represents a prey: [x, y, theta, id]

        # Assign ids to the agents and preys
        agents[:, 4] = np.arange(self.N)
        preys[:, 3] = np.arange(self.N_preys)

        # Generate random x, y for the first agent and prey
        
        # x = np.random.randint(0, boundaries[1]-5)
        # y = np.random.randint(0, boundaries[1]-5)

        x = boundaries[1] // 2
        y = boundaries[1] // 2

        agents[0, :2] = np.array([x, y])

        # generate random x, y for the first prey within a distance of 2 from the first agent
        min_distance = self.min_distance
        x_prey = x + random.choice([-min_distance, min_distance])
        y_prey = y + random.choice([-min_distance, min_distance])
        
        theta = np.random.uniform(-np.pi, np.pi)  # Random theta for the first agent
        theta_prey = np.random.uniform(-np.pi, np.pi)  # Random theta for the first prey
        
        preys[0, :2] = np.array([x_prey, y_prey])
        agents[0, 2] = theta
        preys[0, 2] = theta_prey
        # Generate random relative positions for the rest of the agents and preys
        for i in range(1, self.N):
            relative_x = np.random.uniform(-sigma_i, sigma_i)
            relative_y = np.random.uniform(-sigma_i, sigma_i)

            # Translate relative positions to absolute positions based on the first agent
            x_i = x + relative_x
            y_i = y + relative_y
            agents[i, 0] = x_i
            agents[i, 1] = y_i
            agents[i, 2] = np.random.uniform(-np.pi, np.pi) # Random theta for the rest of the agents

        for i in range(1, self.N_preys):
            relative_x = np.random.uniform(-sigma_i, sigma_i)
            relative_y = np.random.uniform(-sigma_i, sigma_i)


            x_i = x_prey + relative_x
            y_i = y_prey + relative_y
            preys[i, 0] = x_i
            preys[i, 1] = y_i
            preys[i, 2] = np.random.uniform(-np.pi, np.pi) # Random theta for the rest of the preys


        return agents, preys

    def update_states(self, U_i, omega_i, U_i_prey, omega_i_prey, distance_prey, step) -> None:

        

        # Update the states of the agents
        error_x = np.random.uniform(-self.error_range, self.error_range, self.N) * dt
        error_y = np.random.uniform(-self.error_range, self.error_range, self.N) * dt

        dx = U_i * np.cos(self.predators[:, 2]) * dt + error_x
        dy = U_i * np.sin(self.predators[:, 2]) * dt + error_y

        vx = dx / dt
        vy = dy / dt

        self.predators[:, 0] = self.predators[:, 0] + dx
        self.predators[:, 1] = self.predators[:, 1] + dy

        self.predators[:, 2] = self.predators[:, 2] + omega_i * dt

        # Update the states of the preys
        error_x_prey = np.random.uniform(-self.error_range, self.error_range, self.N_preys) * dt
        error_y_prey = np.random.uniform(-self.error_range, self.error_range, self.N_preys) * dt

        dx_prey = U_i_prey * np.cos(self.preys[:, 2]) * dt + error_x_prey
        dy_prey = U_i_prey * np.sin(self.preys[:, 2]) * dt + error_y_prey

        vx_prey = dx_prey / dt
        vy_prey = dy_prey / dt

        self.preys[:, 0] = self.preys[:, 0] + dx_prey
        self.preys[:, 1] = self.preys[:, 1] + dy_prey
        self.preys[:, 2] = self.preys[:, 2] + omega_i_prey * dt


        # Save the data into a csv file
        if self.save:
            # If distance_prey is a 1D array, make it a 2D array with one column

            
            distance_prey = distance_prey[:, np.newaxis]
            U_i = U_i[:, np.newaxis]
            omega_i = omega_i[:, np.newaxis]
            omegamax_ = np.tile(omegamax, (self.N, 1))
            Umax_ = np.tile(Umax, (self.N, 1))
            alpha_pred = np.tile(alpha, (self.N, 1))
            Dp_pred = np.tile(Dp, (self.N, 1))
            vx = vx[:, np.newaxis]
            vy = vy[:, np.newaxis]
            kappa_pred = np.tile(kappa, (self.N, 1))
            step_pred = np.tile(step, (self.N, 1))




            U_i_prey = U_i_prey[:, np.newaxis]
            omega_i_prey = omega_i_prey[:, np.newaxis]
            omegamax_prey_ = np.tile(omegamax_prey, (self.N_preys, 1))
            Umax_prey_ = np.tile(Umax_prey, (self.N_preys, 1))
            alpha_ = np.tile(alpha, (self.N_preys, 1))
            kappa_ = np.tile(kappa, (self.N_preys, 1))
            Dp_ = np.tile(Dp, (self.N_preys, 1))
            vx_prey = vx_prey[:, np.newaxis]
            vy_prey = vy_prey[:, np.newaxis]
            pad_prey = np.zeros((self.N_preys, 1))
            step_prey = np.tile(step, (self.N_preys, 1))



            data_predators = np.hstack((self.predators, distance_prey, U_i, omega_i, omegamax_, Umax_, alpha_pred, kappa_pred, Dp_pred, vx, vy, step_pred))


            data_preys = np.hstack((self.preys,pad_prey, U_i_prey, omega_i_prey, omegamax_prey_, Umax_prey_, alpha_, kappa_, Dp_, vx_prey, vy_prey, step_prey))
            self.saving_data_during_sim(data_predators, data_preys)

        return 


    def get_distance_from_swarm_preys(self, preys: np.array, predators: np.array) -> np.array:
        # Calculate distances between each predator and each prey
        prey_positions = preys[:, :2]  # Extract prey positions
        predator_positions = predators[:, :2]  # Extract predator positions

        # Calculate distances between each predator and each prey using broadcasting
        distances = np.linalg.norm(predator_positions[:, None, :] - prey_positions[None, :, :], axis=2)

        # Filter distances based on sensor range
        distances_filtered = np.where(distances <= self.sensor_range, distances, np.inf)

        # Find the closest prey for each predator
        closest_prey_distances = np.min(distances_filtered, axis=1)

        # Compute the result as 1/distance for non-infinite distances and 0 otherwise
        result = np.where(closest_prey_distances != np.inf, 1 / closest_prey_distances, 0)
        # print(f"result: {result}")

        return result


    def get_distance_from_swarm_preys_blindspot(self, preys: np.array, predators: np.array) -> np.array:
        # Calculate distances between each predator and each prey
        prey_positions = preys[:, :2]  # Extract prey positions
        predator_positions = predators[:, :2]  # Extract predator positions

        # Define the centers of the two circles that make up the "8" shape
        circle1_center = predator_positions + np.array([self.sensor_range/2 * np.cos(predators[:, 2]), self.sensor_range/2 * np.sin(predators[:, 2])]).T
        circle2_center = predator_positions - np.array([self.sensor_range/2 * np.cos(predators[:, 2]), self.sensor_range/2 * np.sin(predators[:, 2])]).T

        # Calculate distances from preys to each circle center
        distances_to_circle1 = np.linalg.norm(prey_positions[None, :, :] - circle1_center[:, None, :], axis=2)
        distances_to_circle2 = np.linalg.norm(prey_positions[None, :, :] - circle2_center[:, None, :], axis=2)

        # Identify preys that are within either circle
        in_sensing_range = (distances_to_circle1 <= self.sensor_range/2) | (distances_to_circle2 <= self.sensor_range/2)

        # Calculate distances between each predator and each prey using broadcasting
        distances = np.linalg.norm(predator_positions[:, None, :] - prey_positions[None, :, :], axis=2)

        # Filter distances based on whether preys are in the sensing range
        distances_filtered = np.where(in_sensing_range, distances, np.inf)

        # Find the closest prey for each predator
        closest_prey_distances = np.min(distances_filtered, axis=1)

        # Compute the result as 1/distance for non-infinite distances and 0 otherwise
        result = np.where(closest_prey_distances != np.inf, 1 / closest_prey_distances, 0)
        # print(f"result: {result}")

        return result



    def check_file(self):
        path = os.path.dirname(self.filename)
        if not os.path.exists(path):
            os.makedirs(path)
        elif os.path.isfile(self.filename):
            overwrite = input("File already exists. Do you want to overwrite? (yes/no): ")
            if overwrite.lower() == 'yes':
                os.remove(self.filename)
            else:
                print("Program stopped.")
                sys.exit()

    def saving_data_during_sim(self, data_predator: np.array, data_prey: np.array) -> None:

        # Determine the number of prey agents in the simulation
        num_prey_agents = max(self.preys.shape[0], data_prey.shape[0])

        # Pad self.preys array if needed
        if self.preys.shape[0] < num_prey_agents:
            padding = np.full((num_prey_agents - self.preys.shape[0], self.preys.shape[1]), np.nan)
            self.preys = np.vstack((self.preys, padding))

        # Pad data_prey array if needed
        if data_prey.shape[0] < num_prey_agents:
            padding = np.full((num_prey_agents - data_prey.shape[0], data_prey.shape[1]), np.nan)
            data_prey = np.vstack((data_prey, padding))

        # Add a column to preys that indicates whether the sensor is turned on (all ones)
        sensor_states = np.ones((num_prey_agents, 1))
        # Insert the sensor state column before the id column (index 3)
        data_prey_with_sensor_state = np.concatenate((data_prey[:, :3], sensor_states, data_prey[:, 3:]), axis=1)

        
        # Combine data for predators and preys
        combined_data = np.hstack((data_predator, np.ones((data_predator.shape[0], 1))))  # Add a column indicating it's predator data
        data_prey_with_sensor_state = np.hstack((data_prey_with_sensor_state, np.zeros((data_prey_with_sensor_state.shape[0], 1))))  # Add a column indicating it's prey data

        # print(f"combined_data: {combined_data.shape}, data_prey_with_sensor_state: {data_prey_with_sensor_state.shape}")
        combined_data = np.vstack((combined_data, data_prey_with_sensor_state))  # Add prey data

        # save the data into self.data_to_save
        self.data_to_save.append(combined_data)

    
    def save_data(self) -> None:
        header = 'x,y,theta,distanceSensor,id,distance,U_i,omega_i,maxAngularVelocity,maxLinearVelocity,alpha,kappa,Dp,vx,vy,step,isPredator\n'
        # combine all matrices into one matrix in self.data_to_save
        flattened_data = [data for data in self.data_to_save]
        self.data_to_save = np.vstack(flattened_data)
        # If file doesn't exist, write the header and the data
        if not os.path.isfile(self.filename):
            with open(self.filename, 'w') as f:
                f.write(header)
                np.savetxt(f, self.data_to_save, delimiter=',', fmt='%f')





    def scm_behavior(self, agents: np.array, preys: np.array, distance:float) -> Tuple[np.array, np.array]:

        # Compute the p vector for the agents
        if self.pdm:
            p = p_vector_DM(agents, distance)
        else:
            p = p_vector(agents, distance)
        # Compute the h vector for the agents
        h = h_vector(agents)
        # compute r vector for the predators
        r = r_vector(agents, self.boundaries)
        # Compute the p vector for the preys
        p_prey = p_vector(preys, 0)
        # Compute the r vector for the preys
        r_prey = r_vector(preys, self.boundaries)
        # Compute the h vector for the preys
        h_prey = h_vector(preys)

        # repulsion predators
        pr = repulsion_predator(preys, agents, self.sensing_range)

        force_agents = alpha * p + beta * h + gamma * r # N x 2 matrix
        force_preys  = alpha * p_prey + beta * h_prey + gamma * r_prey + kappa * pr # N x 2 matrix
        
        # project from global to local reference frame
        fx = np.sqrt(force_agents[:, 0, np.newaxis]**2 + force_agents[:, 1, np.newaxis]**2) * np.cos(np.arctan2(force_agents[:, 1,np.newaxis], force_agents[:, 0,np.newaxis]) - agents[:, 2, np.newaxis])
        fy = np.sqrt(force_agents[:, 0, np.newaxis]**2 + force_agents[:, 1, np.newaxis]**2) * np.sin(np.arctan2(force_agents[:, 1,np.newaxis], force_agents[:, 0,np.newaxis]) - agents[:, 2, np.newaxis])

        fx_prey = np.sqrt(force_preys[:, 0, np.newaxis]**2 + force_preys[:, 1, np.newaxis]**2) * np.cos(np.arctan2(force_preys[:, 1,np.newaxis], force_preys[:, 0,np.newaxis]) - preys[:, 2, np.newaxis])
        fy_prey = np.sqrt(force_preys[:, 0, np.newaxis]**2 + force_preys[:, 1, np.newaxis]**2) * np.sin(np.arctan2(force_preys[:, 1,np.newaxis], force_preys[:, 0,np.newaxis]) - preys[:, 2, np.newaxis])


        fx = np.squeeze(fx)
        fy = np.squeeze(fy)
        fx_prey = np.squeeze(fx_prey)
        fy_prey = np.squeeze(fy_prey)
        # print(f"fx: {fx.shape}, fy: {fy.shape}, fx_prey: {fx_prey.shape}, fy_prey: {fy_prey.shape}")

        # Compute the U_i and omega_i for the agents
        U_i = K1 * fx + Uc
        omega_i = K2 * fy

        # print(f"U_i: {U_i}, omega_i: {omega_i}")

        # Compute the U_i and omega_i for the agents
        U_i_prey = K1 * fx_prey + Uc
        omega_i_prey = K2 * fy_prey

        # RULES FROM PAPER
        U_i = np.where(U_i < 0, 0, U_i)
        U_i = np.where(U_i > Umax, Umax, U_i)

        omega_i = np.where(omega_i < -omegamax, -omegamax, omega_i)
        omega_i = np.where(omega_i > omegamax, omegamax, omega_i)

        U_i_prey = np.where(U_i_prey < 0, 0, U_i_prey)
        U_i_prey = np.where(U_i_prey > Umax, Umax, U_i_prey)

        omega_i_prey = np.where(omega_i_prey < -omegamax, -omegamax, omega_i_prey)
        omega_i_prey = np.where(omega_i_prey > omegamax, omegamax, omega_i_prey)

        return U_i, omega_i, U_i_prey, omega_i_prey
    
    def update(self, step: int):

        # distance = self.get_distance_from_swarm_preys(self.preys, self.predators)
        distance = self.get_distance_from_swarm_preys_blindspot(self.preys, self.predators) if self.blind_spots else self.get_distance_from_swarm_preys(self.preys, self.predators)
        U_i, omega_i, U_i_prey, omega_i_prey = self.scm_behavior(self.predators, self.preys, distance)
        self.update_states(U_i, omega_i, U_i_prey, omega_i_prey, distance, step)

    def run(self):
        if self.draw:
            self.fig, self.ax = plt.subplots(1, 1, figsize=(12, 8))
            self.draw_progress_bar = tqdm(total=self.steps, desc='Simulation Progress')


        for step in range(self.steps):
            self.update(step)



            # if sensor is turned on, then dwaw circle
            # self.circles = [plt.Circle((self.predators[i, 0], self.predators[i, 1]), self.sensor_range, color='r', fill=False) for i in range(len(self.predators)) if self.predators[i, 3] == 1]
            if not self.no_progress_bar:
                self.draw_progress_bar.update(1)
        

        if self.save:
            self.save_data()
        self.draw_progress_bar.close()
        print('Simulation finished')
   