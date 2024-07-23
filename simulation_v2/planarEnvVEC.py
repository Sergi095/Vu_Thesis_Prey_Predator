
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
import matplotlib
matplotlib.use('TkAgg')
# matplotlib.use('Agg')
from typing import List, Tuple, Union

# from agent import planarQuadrator
# from prey import preyPlanarQuadrator
from concurrent.futures import ThreadPoolExecutor


# if 'CUDA_PATH' in os.environ:
#     import cupy as cp

import numpy as np


class environment_agent(object):


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
                #  filename_prey: str = "",
                #  min_distance: int = 5,
                 blind_spots: bool = False,
                 draw_circles: bool = False,
                 pdm: bool = False,
                 folder: str = "tests",
                 # CONSTANTS FOR THE SIMULATION
                 sigma_init: float = 2.5,
                 alpha: float = 2.0,
                 beta: float = 0.0,
                 gamma: float = 0.0, 
                 kappa: float = 0.0,
                 alpha_prey: float = 1.0,
                 Dp: float = 3.0,
                 sigma_i_predator: float = 0.7,
                 sigma_i_pred_non_sensing: float = 0.75,
                 sigma_i_pred_DM: float = 1.4,
                 sigma_i_pred_non_sensing_DM: float = 1.7,
                 Dp_prey: float = 3.0,
                 Dp_pm: float = 3.0,
                 sigma_i_prey: float = 0.7,
                 save_last_step: bool = False,
                 plot_vel: bool = False,
                    ) -> None:




        self.boundaries            = boundaries
        self.folder                = folder
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
        self.filename              = f'dataPreyPredator/{self.folder}/'+filename+'.npy'
        self.trajectory            = []
        self.trajectory_prey       = []
        self.blind_spots           = blind_spots
        self.draw_circles          = draw_circles
        self.pdm                   = pdm
        self.draw_progress_bar     = tqdm(total=self.steps, desc='Simulation Progress')
        self.save_last_step        = save_last_step
        self.plot_vel              = plot_vel

        #################################################################################################
        # CONSTANTS FOR THE SIMULATION
        self.k_rep    = 2.0                   # Boundary avoidance strength coefficient
        self.L0       = 0.5                   # Avoidance vector relaxation threshold 
        self.Dr       = 0.5                   # Boundary perception radius  
        # sigma    = 0.35                  # sigma from tugay 0.7. It works with 0.1 in 10 by 10, it's working with 0.35 in 50 by 50 and N = 100
        # sigma    = 1.2                  # sigma from tugay 0.7. It works with 0.1 in 10 by 10
        # self.sigma_i  = np.sqrt(2) * sigma    # Desired distance to neighbors
        self.sigma_i  = sigma_init            # Desired distance to neighbors
        self.epsilon  = 12.0                  # Strength of the repulsive potential
        self.epsilon_prey = 12.0              # Strength of the repulsive potential for preys
        # self.Dp       = 5.0                   # Neighbors perception radius val from paper 2.0
        self.dt       = 0.05                  # time step 0.05
        self.K1       = 0.5                   # linear speed gain (table 3)
        self.K2       = 0.05                  # angular speed gain (table 3)
        self.Uc       = 0.05                  # constant speed addition (units/time) (table 2)
        self.Umax     = 0.15                  # maximum linear speed (table 2) val from paper 0.15
        self.omegamax = np.pi/3.0             # maximum angular speed (rad/time) (table 3) in paper is pi/3
        self.alpha    = alpha                 # for p vector neighbor
        self.beta     = beta                  # for h vector heading
        self.gamma    = gamma                 # for r vector boundary
        ##################################
        self.Umax_prey = 0.15                 # maximum linear speed for prey (I initially set it to 0.08, also taking into account Tugays code) with equal umax they get caught
        self.alpha_prey = alpha_prey          # for p vector neighbor for prey
        self.omegamax_prey = np.pi/3.0        # maximum angular speed for prey (sane angular speed gets caught)
        self.kappa = kappa                    # for repultion from predator
        ##################################


        ##################################
        # inside formula functions
        # params for non-sensing agents < 99%
        self.Dp       = Dp                   # 
        self.sigma_i_predator = sigma_i_predator
        self.sigma_i_pred_non_sensing = sigma_i_pred_non_sensing
        # params for sensing agents = 99%
        # self.Dp       = 2.0                   # 
        # self.sigma_i_predator = 0.7
        # self.sigma_i_pred_non_sensing = 0.6
        # params for P vector DM
        self.sigma_i_pred_DM = sigma_i_pred_DM         # 
        self.sigma_i_pred_non_sensing_DM = sigma_i_pred_non_sensing_DM 
        ##################################
        self.Dp_pm    = Dp_pm                 # Neighbors perception radius val from paper 2.0
        self.Dp_prey  = Dp_prey               # Neighbors perception radius val from paper 2.0
        # INITIAL SIGMA THIS IS ONLY FOR PLACING THE AGENTS
        self.sigma_i_prey = sigma_i_prey               # Used for computing p vector for preys
        ##################################
        # if self.pdm:
        #     # self.min_distance          = ((self.sigma_i_prey*np.sqrt(2)*self.N_preys)/2*np.pi)# - (self.sensor_range + self.sigma_i_pred_non_sensing_DM) # Minimum distance between predators and preys
        #     # self.min_distance          = (self.sigma_i_prey * np.sqrt(self.N_preys/2)) + ( self.sensor_range/2 + self.sigma_i_prey  - self.no_sensor ) # Minimum distance between predators and preys
        #     # self.min_distance          = ((self.N_preys * self.sigma_i_prey * np.sqrt(2))/(2*np.pi))  
        #     print(np.sqrt(self.N_preys))
        #     self.min_distance = ((np.sqrt(self.N_preys )* self.sigma_i_prey * np.sqrt(2))/(2*np.pi))
        #      #- self.no_sensor - self.sigma_i_pred_DM# - self.sensor_range
        #     print(f"min_distance: {self.min_distance}")
        #     # self.min_distance          = (self.sigma_i_prey * np.sqrt(self.N_preys/2)) + (  self.sensor_range - (self.sigma_i_pred_DM + self.no_sensor + 0.5) ) # Minimum distance between predators and preys
        # else:
        #     self.min_distance          = (self.sigma_i_prey * np.sqrt(self.N_preys/2)) + (  (self.sigma_i_predator * np.sqrt(self.N/2)) - self.sensor_range - self.no_sensor )



        #################################################################################################

        self.predators, self.preys = self.generate_agents_and_preys()
        self.data_to_save          = []
        if self.plot_vel:
            self.velocities_to_plot_predator = []
            self.velocities_to_plot_prey = []
            self.mean_velocities_predator = []
            self.mean_velocities_prey = []
            self.time_steps = []
        if self.save or self.save_last_step:
            self.check_file()
        
    def no_sensor_agents(self, no_sensor: float, predators: np.array, preys: np.array) -> np.array:
        # Each row represents an agent: [x, y, theta, sensing_yes/no]
        N = len(predators)
        no_sensor_agents = np.ones(N)  # Initially, all sensors are turned on
        # Calculate the number of predators without sensors
        no_sensor_count = int(no_sensor * N)
        print(f"no_sensor_count: {no_sensor_count}")
        # if no_sensor_count >= N-5:
        if no_sensor >= 0.10:

            # Calculate distances from each predator to the nearest prey
            distances = [min(np.linalg.norm(predator[:2] - prey[:2]) for prey in preys) for predator in predators]

            # Create a list of tuples, each containing the index of a predator and its distance to the nearest prey
            index_distance_pairs = list(enumerate(distances))

            # Sort the list by distance (from smallest to largest)
            index_distance_pairs.sort(key=lambda x: x[1])

            # Get the indices of the predators
            sorted_indices = [index for index, distance in index_distance_pairs]

            # Reverse the list so that predators closer to preys are at the end
            sorted_indices.reverse()



            # Select the predators that are closest to the preys to have no sensors
            selected_indices = sorted_indices[:no_sensor_count]
            no_sensor_agents[selected_indices] = 0

            predators[:, 3] = no_sensor_agents
        elif no_sensor != 1:
            # Randomly select agents to turn off their sensors
            selected_indices = np.random.choice(N, size=no_sensor_count, replace=False)
            no_sensor_agents[selected_indices] = 0
            predators[:, 3] = no_sensor_agents

        return predators
    
    def generate_agents_and_preys(self) -> Tuple[np.array, np.array]:
        boundaryX, boundaryY = self.boundaries
        boundaries = (10, 10) if self.boundaries == [0,0] else (boundaryX, boundaryY)
        
        agents = np.ones((self.N, 5))  # Each row represents an agent: [x, y, theta, sensing_yes/no, id]
        preys = np.zeros((self.N_preys, 4))  # Each row represents a prey: [x, y, theta, id]
        
        agents[:, 4] = np.arange(self.N)
        preys[:, 3] = np.arange(self.N_preys)
        
        grid_size_agents = int(np.ceil(np.sqrt(self.N)))
        grid_size_preys = int(np.ceil(np.sqrt(self.N_preys)))
        
        agent_spacing_x = 0.5#boundaries[0] / grid_size_agents
        agent_spacing_y = 0.5#boundaries[1] / grid_size_agents
        prey_spacing_x  = 0.5#boundaries[0] / grid_size_preys
        prey_spacing_y  = 0.5#boundaries[1] / grid_size_preys

        min_distance = (np.sqrt(self.N_preys) * self.sigma_i_prey * prey_spacing_x) + self.sensor_range/1.5 if self.N_preys >= 100 else (np.sqrt(self.N_preys) * self.sigma_i_prey * prey_spacing_x) + self.sensor_range/1.5
        print(f"min_distance: {min_distance}")
        
        cons_theta = np.pi/4
        # Place agents on the grid
        agent_idx = 0
        for i in range(grid_size_agents):
            for j in range(grid_size_agents):
                if agent_idx < self.N:
                    x = i * agent_spacing_x + agent_spacing_x / 2
                    y = j * agent_spacing_y + agent_spacing_y / 2
                    theta = np.random.uniform(0, 2*np.pi)
                    theta = np.random.uniform(0, 2*np.pi)
                    agents[agent_idx, :3] = np.array([x, y, theta])
                    agent_idx += 1
        
        # Place preys on the grid
        prey_idx = 0
        signx = np.random.choice([-1, 1])
        signy = np.random.choice([-1, 1])
        signx = np.random.choice([-1, 1])
        signy = np.random.choice([-1, 1])
        for i in range(grid_size_preys):
            for j in range(grid_size_preys):
                if prey_idx < self.N_preys:
                    x = i * prey_spacing_x + prey_spacing_x / 2 + signx* min_distance
                    y = j * prey_spacing_y + prey_spacing_y / 2 + signy* min_distance
                    x = i * prey_spacing_x + prey_spacing_x / 2 + signx* min_distance
                    y = j * prey_spacing_y + prey_spacing_y / 2 + signy* min_distance

                    theta = np.random.uniform(0, 2*np.pi)
                    theta = np.random.uniform(0, 2*np.pi)
                    preys[prey_idx, :3] = np.array([x, y, theta])
                    prey_idx += 1
        
        agents = self.no_sensor_agents(self.no_sensor, agents, preys)
        
        return agents, preys


    

    def update_states(self, U_i, omega_i, U_i_prey, omega_i_prey, distance_prey, step) -> None:

        

        # Update the states of the agents
        error_x = np.random.uniform(-self.error_range, self.error_range, self.N) * self.dt
        error_y = np.random.uniform(-self.error_range, self.error_range, self.N) * self.dt

        dx = U_i * np.cos(self.predators[:, 2]) * self.dt + error_x
        dy = U_i * np.sin(self.predators[:, 2]) * self.dt + error_y

        vx = dx / self.dt
        vy = dy / self.dt

        if self.plot_vel:
            self.velocities_to_plot_predator.append([vx, vy])


        self.predators[:, 0] = self.predators[:, 0] + dx
        self.predators[:, 1] = self.predators[:, 1] + dy

        self.predators[:, 2] = self.predators[:, 2] + omega_i * self.dt

        # Update the states of the preys
        error_x_prey = np.random.uniform(-self.error_range, self.error_range, self.N_preys) * self.dt
        error_y_prey = np.random.uniform(-self.error_range, self.error_range, self.N_preys) * self.dt

        dx_prey = U_i_prey * np.cos(self.preys[:, 2]) * self.dt + error_x_prey
        dy_prey = U_i_prey * np.sin(self.preys[:, 2]) * self.dt + error_y_prey

        vx_prey = dx_prey / self.dt
        vy_prey = dy_prey / self.dt
        if self.plot_vel:
            self.velocities_to_plot_prey.append([vx_prey, vy_prey])


        self.preys[:, 0] = self.preys[:, 0] + dx_prey
        self.preys[:, 1] = self.preys[:, 1] + dy_prey
        self.preys[:, 2] = self.preys[:, 2] + omega_i_prey * self.dt


        # Save the data into a csv file
        if self.save or self.save_last_step:
            # If distance_prey is a 1D array, make it a 2D array with one column

            
            distance_prey = distance_prey[:, np.newaxis]
            U_i = U_i[:, np.newaxis]
            omega_i = omega_i[:, np.newaxis]
            omegamax_ = np.tile(self.omegamax, (self.N, 1))
            Umax_ = np.tile(self.Umax, (self.N, 1))
            alpha_pred = np.tile(self.alpha, (self.N, 1))
            vx = vx[:, np.newaxis]
            vy = vy[:, np.newaxis]
            kappa_pred = np.tile(self.kappa, (self.N, 1))
            step_pred = np.tile(step, (self.N, 1))




            U_i_prey = U_i_prey[:, np.newaxis]
            omega_i_prey = omega_i_prey[:, np.newaxis]
            omegamax_prey_ = np.tile(self.omegamax_prey, (self.N_preys, 1))
            Umax_prey_ = np.tile(self.Umax_prey, (self.N_preys, 1))
            alpha_ = np.tile(self.alpha, (self.N_preys, 1))
            kappa_ = np.tile(self.kappa, (self.N_preys, 1))
            vx_prey = vx_prey[:, np.newaxis]
            vy_prey = vy_prey[:, np.newaxis]
            pad_prey = np.zeros((self.N_preys, 1))
            step_prey = np.tile(step, (self.N_preys, 1))



            data_predators = np.hstack((self.predators, distance_prey, U_i, omega_i, omegamax_, Umax_, alpha_pred, kappa_pred, vx, vy, step_pred))


            data_preys = np.hstack((self.preys,pad_prey, U_i_prey, omega_i_prey, omegamax_prey_, Umax_prey_, alpha_, kappa_, vx_prey, vy_prey, step_prey))
            self.saving_data_during_sim(data_predators, data_preys)

        return 




    def get_distance_from_swarm_preys(self, preys: np.array, predators: np.array) -> np.array:
        # Calculate distances between each predator and each prey
        prey_positions = preys[:, :2]  # Extract prey positions
        predator_positions = predators[:, :2]  # Extract predator positions

        # Calculate distances between each predator and each prey using broadcasting
        distances = np.linalg.norm(predator_positions[:, None, :] - prey_positions[None, :, :], axis=2)

        # Create a mask for prey within the sensor range of each predator
        mask = distances <= self.sensor_range

        result = np.zeros(predators.shape[0])

        for i in range(predators.shape[0]):
            # Get distances of prey within sensor range of the current predator
            within_range_distances = distances[i][mask[i]]

            if within_range_distances.size > 0:
                # Average the distances within the sensor range
                avg_distance = np.mean(within_range_distances)
                
                # Compute the result as 1/distance
                result[i] = 1 / avg_distance

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
        sensor_states = 1
        # Insert the sensor state column before the id column (index 3)
        data_prey_with_sensor_state = np.insert(data_prey, 3, sensor_states, axis=1)

        
        # Combine data for predators and preys
        combined_data = np.hstack((data_predator, np.ones((data_predator.shape[0], 1))))  # Add a column indicating it's predator data
        data_prey_with_sensor_state = np.hstack((data_prey_with_sensor_state, np.zeros((data_prey_with_sensor_state.shape[0], 1))))  # Add a column indicating it's prey data

        # print(f"combined_data: {combined_data.shape}, data_prey_with_sensor_state: {data_prey_with_sensor_state.shape}")
        combined_data = np.vstack((combined_data, data_prey_with_sensor_state))  # Add prey data

        # save the data into self.data_to_save
        self.data_to_save.append(combined_data)

    
    def save_data(self) -> None:
        header = 'x,y,theta,distanceSensor,id,distance,U_i,omega_i,maxAngularVelocity,maxLinearVelocity,alpha,kappa,vx,vy,step,isPredator\n'
        # combine all matrices into one matrix in self.data_to_save
        flattened_data = [data for data in self.data_to_save]
        self.data_to_save = np.vstack(flattened_data)
        # If file doesn't exist, write the header and the data
        # if not os.path.isfile(self.filename):
        #     with open(self.filename, 'w') as f:
        #         f.write(header)
        #         np.savetxt(f, self.data_to_save, delimiter=',', fmt='%f')
        np.save(self.filename, self.data_to_save)



    def scm_behavior(self, agents: np.array, preys: np.array, distance:float) -> Tuple[np.array, np.array]:

        # Compute the p vector for the agents
        if self.pdm:
            p = self.p_vector_DM(agents, distance)
        else:
            p = self.p_vector(agents, distance)
        # Compute the h vector for the agents
        h = self.h_vector(agents)
        # compute r vector for the predators
        r = self.r_vector(agents, self.boundaries)
        # Compute the p vector for the preys
        p_prey = self.p_vector(preys, 0)
        # Compute the r vector for the preys
        r_prey = self.r_vector(preys, self.boundaries)
        # Compute the h vector for the preys
        h_prey = self.h_vector(preys)

        # repulsion predators
        pr = self.repulsion_predator(preys, agents, self.sensing_range)
        # magnitudes = np.sqrt(pr[:, 0]**2 + pr[:, 1]**2)
        # print(f"magnitudes of pr: {magnitudes}")
        # magnitudes = np.sqrt(pr[:, 0]**2 + pr[:, 1]**2)
        # print(f"magnitudes of pr: {magnitudes}")

        force_agents = self.alpha * p + self.beta * h + self.gamma * r # N x 2 matrix
        force_preys  = self.alpha_prey * p_prey + self.beta * h_prey + self.gamma * r_prey + self.kappa * pr # N x 2 matrix
        # print (pr)
        # print (pr)
        
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
        # U_i = self.K1 * fx + 0.15
        U_i = self.K1 * fx + self.Uc
        omega_i = self.K2 * fy

        # print(f"U_i: {U_i}, omega_i: {omega_i}")

        # Compute the U_i and omega_i for the agents
        # U_i_prey = self.K1 * fx_prey + 0.15
        U_i_prey = self.K1 * fx_prey + self.Uc
        omega_i_prey = self.K2 * fy_prey

        # RULES FROM PAPER
        U_i = np.where(U_i < 0, 0, U_i)
        U_i = np.where(U_i > self.Umax, self.Umax, U_i)

        omega_i = np.where(omega_i < -self.omegamax, -self.omegamax, omega_i)
        omega_i = np.where(omega_i > self.omegamax, self.omegamax, omega_i)

        U_i_prey = np.where(U_i_prey < 0, 0, U_i_prey)
        U_i_prey = np.where(U_i_prey > self.Umax, self.Umax, U_i_prey)

        omega_i_prey = np.where(omega_i_prey < -self.omegamax, -self.omegamax, omega_i_prey)
        omega_i_prey = np.where(omega_i_prey > self.omegamax, self.omegamax, omega_i_prey)

        return U_i, omega_i, U_i_prey, omega_i_prey
    
    def update(self, step: int):

        # distance = self.get_distance_from_swarm_preys(self.preys, self.predators)
        distance = self.get_distance_from_swarm_preys_blindspot(self.preys, self.predators) if self.blind_spots else self.get_distance_from_swarm_preys(self.preys, self.predators)
        U_i, omega_i, U_i_prey, omega_i_prey = self.scm_behavior(self.predators, self.preys, distance)
        self.update_states(U_i, omega_i, U_i_prey, omega_i_prey, distance, step)


    def run(self):
        plt.ion()
        if self.draw:
            
            if self.plot_vel:
                self.fig, self.ax = plt.subplots(1, 2, figsize=(12, 8))

            else:
                self.fig, self.ax = plt.subplots(1, 1, figsize=(12, 8))

            self.draw_progress_bar = tqdm(total=self.steps, desc='Simulation Progress')


        for step in range(self.steps):
            self.update(step)
            if self.plot_vel:
                self.time_steps.append(step)

            if self.draw:
                # make step counter in the bottom of the window
                if self.boundaries == [0, 0]:
                    if self.pdm:
                        self.fig.suptitle(f'Swarm Simulation with: \n force={self.alpha}·pADM || force_p = {self.alpha_prey}·p+kappa·rp|| sensor range predator:{self.sensor_range}, sensor range prey:{self.sensing_range} || with {1- round(self.percentage_no_sensor, 2)*100}% sensing agents \n {len(self.predators)} Predators & {len(self.preys)} Preys. \n In boundless environment. \n Step: {step}/{self.steps}')
                    else:
                        self.fig.suptitle(f'Swarm Simulation with: \n  force={self.alpha}·p || force_p = {self.alpha_prey}·p+kappa·rp|| sensor range predator:{self.sensor_range}, sensor range prey:{self.sensing_range} || with {1 -round(self.percentage_no_sensor, 2)*100}% sensing agents \n {len(self.predators)} Predators & {len(self.preys)} Preys. \n In boundless environment. \n Step: {step}/{self.steps}')
                else:
                    if self.pdm:
                        self.fig.suptitle(f'Swarm Simulation with: \n  force={self.alpha}·pDM || force_p = {self.alpha_prey}·p+kappa·rp|| sensor range predator:{self.sensor_range}, sensor range prey:{self.sensing_range} || with { 1- round(self.percentage_no_sensor, 2)*100}% sensing agents \n {len(self.predators)} Predators & {len(self.preys)} Preys. \n In {self.boundaries[0]}X{self.boundaries[1]} environment. \n Step: {step}/{self.steps}')
                    else:
                        self.fig.suptitle(f'Swarm Simulation with: \n  force={self.alpha}·p || force_p = {self.alpha_prey}·p+kappa·rp|| sensor range predator:{self.sensor_range}, sensor range prey:{self.sensing_range} || with {1-round(self.percentage_no_sensor, 2)*100}% sensing agents \n {len(self.predators)} Predators & {len(self.preys)} Preys. \n In {self.boundaries[0]}X{self.boundaries[1]} environment. \n Step: {step}/{self.steps}')

            mean_predatorx = np.mean(self.predators[:, 0])
            mean_predatory = np.mean(self.predators[:, 1])
            mean_preyx = np.mean(self.preys[:, 0])
            mean_preyy = np.mean(self.preys[:, 1])
            self.trajectory.append([mean_predatorx, mean_predatory])
            self.trajectory_prey.append([mean_preyx, mean_preyy])
            if self.plot_vel:
                pred_vel_x = np.array(self.velocities_to_plot_predator)[:, 0]
                pred_vel_y = np.array(self.velocities_to_plot_predator)[:, 1]

                prey_vel_x = np.array(self.velocities_to_plot_prey)[:, 0]
                prey_vel_y = np.array(self.velocities_to_plot_prey)[:, 1]

                self.mean_velocities_predator.append(np.mean((np.sqrt(pred_vel_x**2 + pred_vel_y**2))))
                self.mean_velocities_prey.append(np.mean((np.sqrt(prey_vel_x**2 + prey_vel_y**2))))

            # print('VELOCITIES', len(self.mean_velocities_predator), len(self.mean_velocities_prey))
            # print('STEPS', len(list(range(step+1))))

            # if sensor is turned on, then dwaw circle
            # self.circles = [plt.Circle((self.predators[i, 0], self.predators[i, 1]), self.sensor_range, color='r', fill=False) for i in range(len(self.predators)) if self.predators[i, 3] == 1]
            if not self.no_progress_bar:
                self.draw_progress_bar.update(1)

    # Initialize the plots before starting the simulation loop
    

    # During the simulation loop
            
            if self.pdm:
                text = f' Dp_ADM: {self.Dp_pm}, self.Dp_prey: {self.Dp_prey}, self.sigma_i_prey: {self.sigma_i_prey}, self.sigma_i_pred_ADM: {self.sigma_i_pred_DM}, self.sigma_i_pred_non_sensing_ADM: {self.sigma_i_pred_non_sensing_DM}'
            else:
                text = f' self.Dp: {self.Dp}, self.Dp_prey: {self.Dp_prey}, self.sigma_i_prey: {self.sigma_i_prey}, sigma_i_pred: {self.sigma_i_predator}, self.sigma_i_pred_non_sensing: {self.sigma_i_pred_non_sensing}'
            if self.draw and step % 50 == 0:
                if not plt.fignum_exists(self.fig.number):
                    break
                # self.draw_simulation()
                if self.plot_vel:
                    self.draw_simulation_with_velocities()
                    self.ax[0].text(0.5, -0.1, text, horizontalalignment='center', verticalalignment='center', transform=self.ax[0].transAxes, fontsize=12, rotation=0)
                else:
                    self.draw_simulation()
                    plt.text(0.5, -0.1, text, horizontalalignment='center', verticalalignment='center', transform=self.ax.transAxes, fontsize=12, rotation=0)  # Adjust coordinates and text as needed
                plt.pause(0.00001)  # Pause to allow for interactivity


        plt.show()
        if self.save:
            print('Saving data...')
            self.save_data()
        if self.save_last_step and step == self.steps-1:
            # print(len(self.data_to_save))
            print('Saving data last step...')
            self.data_to_save = self.data_to_save[step]
            self.save_data()
        self.draw_progress_bar.close()
        print('Simulation finished')
        # sys.exit()

    def draw_simulation(self):


        self.ax.clear()
        self.arrow_length = 0.02 # 0.2
        self.arrow_length = 0.02 # 0.2
        self.width = 0.0025 # 0.05
        self.scaled_arrow_length = 1.0 / self.arrow_length
        
        if self.boundaries == [0, 0]:
            
            self.center_of_mass = np.mean(self.predators[:, :2], axis=0)
            self.center_of_mass_prey = np.mean(self.preys[:, :2], axis=0)
            
            self.both_center_of_mass = np.mean([self.center_of_mass, self.center_of_mass_prey], axis=0)
            self.ax.set_xlim(self.both_center_of_mass[0] - 10, self.both_center_of_mass[0] + 10)
            self.ax.set_ylim(self.both_center_of_mass[1] - 10, self.both_center_of_mass[1] + 10)
        else:
            self.ax.set_xlim(0, self.boundaries[0])
            self.ax.set_ylim(0, self.boundaries[1])

        colors = ['g' if sensing == 0 else 'red' for sensing in self.predators[:, 3]]

        # self.simulation = self.ax.scatter(self.predators[:, 0], self.predators[:, 1], c=colors, label='Predators', marker='o', s=20, facecolors='none')
        sensing_predators = [self.predators[i, :2] for i in range(len(self.predators)) if self.predators[i, 3] == 1]
        no_sensing_predators = [self.predators[i, :2] for i in range(len(self.predators)) if self.predators[i, 3] == 0]
        self.simulation_sensing_predators = self.ax.scatter([sensing_predator[0] for sensing_predator in sensing_predators], [sensing_predator[1] for sensing_predator in sensing_predators], c='r', label='Predators with sensor', marker='o', s=20, facecolors='none')
        self.simulation_no_sensing_predators = self.ax.scatter([no_sensing_predator[0] for no_sensing_predator in no_sensing_predators], [no_sensing_predator[1] for no_sensing_predator in no_sensing_predators], c='g', label='Predators without sensor', marker='o', s=20, facecolors='none')
        self.simulation_prey = self.ax.scatter(self.preys[:, 0], self.preys[:, 1], c='b', label='Preys', marker='o', s=20, facecolors='none')

        self.preys_quiver = self.ax.quiver(self.preys[:, 0], self.preys[:, 1], np.cos(self.preys[:, 2]), np.sin(self.preys[:, 2]), color='b', width=self.width, scale=self.scaled_arrow_length)
        self.predators_quiver = self.ax.quiver(self.predators[:, 0], self.predators[:, 1], np.cos(self.predators[:, 2]), np.sin(self.predators[:, 2]), color=colors, width=self.width, scale=self.scaled_arrow_length)

        # Trajectories
        self.ax.plot(*zip(*self.trajectory), color='r', label='Predator Trajectory')
        self.ax.plot(*zip(*self.trajectory_prey), color='b', label='Prey Trajectory')

        # self.Draw circles for the sensors
        # [plt.Circle((self.predators[i, 0], self.predators[i, 1]), self.sensor_range, color='r', fill=False) for i in range(len(self.predators)) if self.predators[i, 3] == 1]
        # self.ax.add_patch(plt.Circle(self.predators[i, 0], self.predators[i, 1], self.sensor_range, color='r', fill=False) for i in range(len(self.predators)) if self.predators[i, 3] == 1)


        # 

        
        # Create circles for each predator's sensing range
        if self.draw_circles:
            if self.blind_spots:
                circles = []
                for i in range(len(self.predators)):
                    if self.predators[i, 3] == 1:
                        # Create two circles for each predator
                        circle1 = patches.Circle((0, self.sensor_range/2), self.sensor_range/2, color=(1, 0.5, 0.5), fill=False)
                        circle2 = patches.Circle((0, self.sensor_range/2), self.sensor_range/2, color=(1, 0.5, 0.5), fill=False)

                        # Create a transform that rotates and translates the circles based on the predator's orientation and position
                        t = transforms.Affine2D().rotate_around(0, 0, self.predators[i, 2] + np.pi/2).translate(self.predators[i, 0], self.predators[i, 1])

                        # Apply the transform to the circles
                        circle1.set_transform(t + self.ax.transData)
                        circle2.set_transform(t + self.ax.transData)

                        # Add the circles to the list
                        circles.append(circle1)
                        circles.append(circle2)
                # # Add all circles to the plot
                for circle in circles:
                    self.ax.add_patch(circle)
                circles_prey = [plt.Circle((self.preys[i, 0], self.preys[i, 1]), self.sensing_range, color=(0.5, 0.5, 1), fill=False) for i in range(len(self.preys))]
                for circle in circles_prey:
                    self.ax.add_artist(circle)

            else:

                circles = [plt.Circle((self.predators[i, 0], self.predators[i, 1]), self.sensor_range, color=(1, 0.5, 0.5), fill=False) for i in range(len(self.predators)) if self.predators[i, 3] == 1]
                circles_prey = [plt.Circle((self.preys[i, 0], self.preys[i, 1]), self.sensing_range, color=(0.5, 0.5, 1), fill=False) for i in range(len(self.preys))]
                for circle in circles:
                    self.ax.add_artist(circle)
                for circle in circles_prey:
                    self.ax.add_artist(circle)


        plt.legend()
        # plt.draw()


    def draw_simulation_with_velocities(self):
        self.ax[0].clear()
        self.ax[1].clear()


        # self.ax.clear()
        self.arrow_length = 0.02 # 0.2
        self.arrow_length = 0.02 # 0.2
        self.width = 0.0025 # 0.05
        self.scaled_arrow_length = 1.0 / self.arrow_length
        
        if self.boundaries == [0, 0]:
            
            self.center_of_mass = np.mean(self.predators[:, :2], axis=0)
            self.center_of_mass_prey = np.mean(self.preys[:, :2], axis=0)
            
            self.both_center_of_mass = np.mean([self.center_of_mass, self.center_of_mass_prey], axis=0)
            self.ax[0].set_xlim(self.both_center_of_mass[0] - 10, self.both_center_of_mass[0] + 10)
            self.ax[0].set_ylim(self.both_center_of_mass[1] - 10, self.both_center_of_mass[1] + 10)

            
        else:
            self.ax[0].set_xlim(0, self.boundaries[0])
            self.ax[0].set_ylim(0, self.boundaries[1])
# ######################################################
        self.pred_vel_line, = self.ax[1].plot(self.time_steps, self.mean_velocities_predator, 'ro', label='Mean velocity predator')
        self.prey_vel_line, = self.ax[1].plot(self.time_steps, self.mean_velocities_prey, 'bo', label='Mean velocity prey')
        # self.ax[1].set_xlim(0, self.steps)
# ######################################################
        

        colors = ['g' if sensing == 0 else 'red' for sensing in self.predators[:, 3]]

        # self.simulation = self.ax.scatter(self.predators[:, 0], self.predators[:, 1], c=colors, label='Predators', marker='o', s=20, facecolors='none')
        sensing_predators = [self.predators[i, :2] for i in range(len(self.predators)) if self.predators[i, 3] == 1]
        no_sensing_predators = [self.predators[i, :2] for i in range(len(self.predators)) if self.predators[i, 3] == 0]
        self.simulation_sensing_predators = self.ax[0].scatter([sensing_predator[0] for sensing_predator in sensing_predators], [sensing_predator[1] for sensing_predator in sensing_predators], c='r', label='Predators with sensor', marker='o', s=20, facecolors='none')
        self.simulation_no_sensing_predators = self.ax[0].scatter([no_sensing_predator[0] for no_sensing_predator in no_sensing_predators], [no_sensing_predator[1] for no_sensing_predator in no_sensing_predators], c='g', label='Predators without sensor', marker='o', s=20, facecolors='none')
        self.simulation_prey = self.ax[0].scatter(self.preys[:, 0], self.preys[:, 1], c='b', label='Preys', marker='o', s=20, facecolors='none')

        self.preys_quiver = self.ax[0].quiver(self.preys[:, 0], self.preys[:, 1], np.cos(self.preys[:, 2]), np.sin(self.preys[:, 2]), color='b', width=self.width, scale=self.scaled_arrow_length)
        self.predators_quiver = self.ax[0].quiver(self.predators[:, 0], self.predators[:, 1], np.cos(self.predators[:, 2]), np.sin(self.predators[:, 2]), color=colors, width=self.width, scale=self.scaled_arrow_length)

        # Trajectories
        self.ax[0].plot(*zip(*self.trajectory), color='r', label='Predator Trajectory')
        self.ax[0].plot(*zip(*self.trajectory_prey), color='b', label='Prey Trajectory')

        # self.Draw circles for the sensors
        # [plt.Circle((self.predators[i, 0], self.predators[i, 1]), self.sensor_range, color='r', fill=False) for i in range(len(self.predators)) if self.predators[i, 3] == 1]
        # self.ax.add_patch(plt.Circle(self.predators[i, 0], self.predators[i, 1], self.sensor_range, color='r', fill=False) for i in range(len(self.predators)) if self.predators[i, 3] == 1)


        # 

        
        # Create circles for each predator's sensing range
        if self.draw_circles:
            if self.blind_spots:
                circles = []
                for i in range(len(self.predators)):
                    if self.predators[i, 3] == 1:
                        # Create two circles for each predator
                        circle1 = patches.Circle((0, self.sensor_range/2), self.sensor_range/2, color=(1, 0.5, 0.5), fill=False)
                        circle2 = patches.Circle((0, -self.sensor_range/2), self.sensor_range/2, color=(1, 0.5, 0.5), fill=False)

                        # Create a transform that rotates and translates the circles based on the predator's orientation and position
                        t = transforms.Affine2D().rotate_around(0, 0, self.predators[i, 2] + np.pi/2).translate(self.predators[i, 0], self.predators[i, 1])

                        # Apply the transform to the circles
                        circle1.set_transform(t + self.ax[0].transData)
                        circle2.set_transform(t + self.ax[0].transData)

                        # Add the circles to the list
                        circles.append(circle1)
                        circles.append(circle2)
                # # Add all circles to the plot
                for circle in circles:
                    self.ax[1].add_patch(circle)
                circles_prey = [plt.Circle((self.preys[i, 0], self.preys[i, 1]), self.sensing_range, color=(0.5, 0.5, 1), fill=False) for i in range(len(self.preys))]
                for circle in circles_prey:
                    self.ax[1].add_artist(circle)

            else:

                circles = [plt.Circle((self.predators[i, 0], self.predators[i, 1]), self.sensor_range, color=(1, 0.5, 0.5), fill=False) for i in range(len(self.predators)) if self.predators[i, 3] == 1]
                circles_prey = [plt.Circle((self.preys[i, 0], self.preys[i, 1]), self.sensing_range, color=(0.5, 0.5, 1), fill=False) for i in range(len(self.preys))]
                for circle in circles:
                    self.ax[1].add_artist(circle)
                for circle in circles_prey:
                    self.ax[1].add_artist(circle)


        plt.legend()
###############################################################################################
    # FORMULAS

    def get_distance_matrix(self, agents: np.array) -> np.array:
        positions_x = agents[:, 0]
        positions_y = agents[:, 1]
        N = len(positions_x)
        
        distance_matrix = np.linalg.norm(agents[:, np.newaxis, :2] - agents[np.newaxis, :, :2], axis=-1) # N x N matrix
        np.fill_diagonal(distance_matrix, np.inf) 

        return distance_matrix

    def get_angle_matrix(self, agents: np.array) -> np.array:
        positions_x = agents[:, 0]
        positions_y = agents[:, 1]
        N = len(positions_x)
        
        angles = np.arctan2(positions_y[:, np.newaxis] - positions_y[np.newaxis, :], positions_x[:, np.newaxis] - positions_x[np.newaxis, :]) # N x N matrix
        np.fill_diagonal(angles, np.inf) 

        return angles

    def p_vector(self, agents: np.array, distance_swarm_prey: np.array) -> np.array:
        
        
        positions = agents[:, :2]
        N = len(agents)

        if agents.shape == (N, 5): # predators
            self.sigma_i = 1.0
            no_distance_sensor_agents = agents[:, 3] == 0
            self.sigma_i = np.tile(self.sigma_i, (N,N))
            self.sigma_i[no_distance_sensor_agents] *= self.sigma_i_pred_non_sensing
            self.sigma_i[~no_distance_sensor_agents] *= self.sigma_i_predator
            # replace distance with 0 if agent is not a distance sensor
            distance_swarm_prey = np.where(agents[:, 3][:, np.newaxis] == 0, 0, distance_swarm_prey)
            # Keeo this in mind!
            # self.sigma_i += (distance_swarm_prey) * np.diag(self.sigma_i) 
            self.sigma_i += (distance_swarm_prey) * np.diag(self.sigma_i) 
            dist_matrix = self.get_distance_matrix(agents)
            phi = self.get_angle_matrix(agents)
            mask = dist_matrix <= self.Dp

            distance = dist_matrix[mask]
            phi_im = phi[mask]
            self.sigma_i = self.sigma_i[mask]

            px = -self.epsilon * ((2.0 * self.sigma_i**4) / (distance**5) - (self.sigma_i**2) / (distance**3)) * np.cos(phi_im)
            py = -self.epsilon * ((2.0 * self.sigma_i**4) / (distance**5) - (self.sigma_i**2) / (distance**3)) * np.sin(phi_im)

            p_vectors = np.zeros((N,N,2))
            p_values = np.stack((px, py), axis=-1)
            p_vectors[mask] = p_values
            return np.sum(p_vectors, axis=0) # N x 2

        else:
            sigma    = self.sigma_i_prey
            self.sigma_i = sigma
            self.sigma_i = np.tile(self.sigma_i, (N,N))
            dist_matrix = self.get_distance_matrix(agents)
            phi = self.get_angle_matrix(agents)
            mask = dist_matrix <= self.Dp_prey

            distance = dist_matrix[mask]
            phi_im = phi[mask]
            self.sigma_i = self.sigma_i[mask]
            px = -self.epsilon_prey * ((2.0 * self.sigma_i**4) / (distance**5) - (self.sigma_i**2) / (distance**3)) * np.cos(phi_im)
            py = -self.epsilon_prey * ((2.0 * self.sigma_i**4) / (distance**5) - (self.sigma_i**2) / (distance**3)) * np.sin(phi_im)

            p_vectors = np.zeros((N,N,2))
            p_values = np.stack((px, py), axis=-1)
            p_vectors[mask] = p_values
            return np.sum(p_vectors, axis=0) # N x 2


    def p_vector_DM(self, agents: np.array, distance_swarm_prey: np.array) -> np.array:

        positions = agents[:, :2]
        N = len(agents)
        self.sigma_i = 1.0
        no_distance_sensor_agents = agents[:, 3] == 0
        # print(no_distance_sensor_agents)
        self.sigma_i = np.tile(self.sigma_i, (N,N))
        self.sigma_i[no_distance_sensor_agents]  *= self.sigma_i_pred_non_sensing_DM
        self.sigma_i[~no_distance_sensor_agents] *= self.sigma_i_pred_DM
        # replace distance with 0 if agent is not a distance sensor
        distance_swarm_prey = np.where(agents[:, 3][:, np.newaxis] == 0, 0, distance_swarm_prey)
        self.sigma_i += (distance_swarm_prey) * np.diag(self.sigma_i) #+ 1.e-6
        # print(distance_swarm_prey)
        # sigma_i_sum = np.sum(self.sigma_i, axis=1)
        # self.sigma_i /= sigma_i_sum
        # print(self.sigma_i)
        # print(self.sigma_i)
        dist_matrix = self.get_distance_matrix(agents)
        phi = self.get_angle_matrix(agents)
        mask = dist_matrix <= self.Dp_pm 

        distance = dist_matrix[mask]
        phi_im = phi[mask]
        self.sigma_i = self.sigma_i[mask]
        # if agents.shape == (N, 5): print(self.sigma_i)
        # -(self.epsilon / 2) * ((sigma_j / distance)**(2)-(sigma_j / distance)**(1/2))
        px = -((self.epsilon /1.0 )* ( (self.sigma_i / distance ) - np.sqrt( self.sigma_i / distance ))) * np.cos(phi_im)
        py = -((self.epsilon /1.0 )* ( (self.sigma_i / distance ) - np.sqrt( self.sigma_i / distance ))) * np.sin(phi_im)
        # px = -(self.epsilon / 5) * ((self.sigma_i / distance)**3 - (self.sigma_i / distance)**(1/3)) * np.cos(phi_im)
        # py = -(self.epsilon / 5) * ((self.sigma_i / distance)**3 - (self.sigma_i / distance)**(1/3)) * np.sin(phi_im)

        p_vectors = np.zeros((N,N,2))
        p_values = np.stack((px, py), axis=-1)
        p_vectors[mask] = p_values
        return np.sum(p_vectors, axis=0) # N x 2






    def repulsion_predator(self, preys: np.array, predators: np.array, sensing_range: float) -> np.array:
        positions_preys = preys[:, :2]
        positions_predators = predators[:, :2]
        N = len(preys)
        M = len(predators)

        # Calculate the distance matrix (N x M)
        distance_matrix = np.linalg.norm(positions_preys[:, np.newaxis] - positions_predators[np.newaxis], axis=-1)
        
        # Create a mask for predators within the sensing range of each prey
        mask = distance_matrix <= sensing_range

        # Initialize the repulsion vector
        repulsion_vector = np.zeros((N, 2))

        for i in range(N):
            # Get indices of predators within sensing range of the current prey
            within_range_indices = np.where(mask[i])[0]
            
            if within_range_indices.size > 0:
                average_preds = np.mean(positions_predators[within_range_indices], axis=0)

                # Calculate the direction vector from prey to average position of nearby predators
                direction_x = positions_preys[i, 0] - average_preds[0]
                direction_y = positions_preys[i, 1] - average_preds[1]
                distances = np.linalg.norm([direction_x, direction_y])
                weight = 2.0 / (1+distances)

                # Normalize direction vector
                # norm = np.sqrt(direction_x**2 + direction_y**2)
                # if norm > 0:
                #     direction_x /= norm
                #     direction_y /= norm

                # Update the repulsion vector using the direction and weight
                repulsion_vector[i, 0] = weight * direction_x
                repulsion_vector[i, 1] = weight * direction_y

                
                # dist_x = positions_preys[i, 0] - average_preds[0]
                # dist_y = positions_preys[i, 1] - average_preds[1]
                # distances = np.sqrt(dist_x**2 + dist_y**2)
                # # print(distances)  
                # relative_angle = np.arctan2(dist_y, dist_x)

                # weight = 1 / (1+distances)

                # repulsion_vector[i, 0] = weight * (dist_x + np.cos(relative_angle))
                # repulsion_vector[i, 1] = weight * (dist_y + np.sin(relative_angle)) 


                # Calculate the direction vector from prey to average position of nearby predators
                direction_x = positions_preys[i, 0] - average_preds[0]
                direction_y = positions_preys[i, 1] - average_preds[1]
                distances = np.linalg.norm([direction_x, direction_y])
                weight = 2.0 / (1+distances)

                # Normalize direction vector
                # norm = np.sqrt(direction_x**2 + direction_y**2)
                # if norm > 0:
                #     direction_x /= norm
                #     direction_y /= norm

                # Update the repulsion vector using the direction and weight
                repulsion_vector[i, 0] = weight * direction_x
                repulsion_vector[i, 1] = weight * direction_y

                
                # dist_x = positions_preys[i, 0] - average_preds[0]
                # dist_y = positions_preys[i, 1] - average_preds[1]
                # distances = np.sqrt(dist_x**2 + dist_y**2)
                # # print(distances)  
                # relative_angle = np.arctan2(dist_y, dist_x)

                # weight = 1 / (1+distances)

                # repulsion_vector[i, 0] = weight * (dist_x + np.cos(relative_angle))
                # repulsion_vector[i, 1] = weight * (dist_y + np.sin(relative_angle)) 

        
        return repulsion_vector


    def h_vector(self, agents: np.array) -> np.array:
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




    def r_vector(self, agent: np.ndarray, boundaries: Tuple[int, int]) -> np.ndarray:

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
                    if dist < self.Dr:
                        # Compute angle based on which boundary we're closest to
                        Lb = dist
                        if Lb == 0:
                            Lb += 0.5
                        angle = [np.pi, 0, np.pi / 2, -np.pi / 2][j]

                        pb_i = np.array([np.cos(angle), np.sin(angle)])

                        # if vertical boundary
                        if j < 2:
                            rb_i = -self.k_rep * ((1.0 / Lb) - (1.0 / self.L0)) * (pb_i / (Lb**3))
                        # if horizontal boundary
                        else:
                            rb_i = self.k_rep * ((1.0 / Lb) - (1.0 / self.L0)) * (pb_i / (Lb**3))
                        r_vector[i] += rb_i
            return r_vector 
        else:
            return np.zeros((len(agent), 2))
###############################################################################################