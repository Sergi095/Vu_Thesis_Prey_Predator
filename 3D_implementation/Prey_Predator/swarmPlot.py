import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib
matplotlib.use('TkAgg')

class SwarmPlotter:
    def __init__(self, n_agents, n_preys, x_lim, y_lim, z_lim, no_sensor_percentage, boundless=False):
        title = f"Swarm of {n_agents} agents and {n_preys} preys with {no_sensor_percentage * 100}% no distance sensors predators"
        self.n_agents = n_agents
        self.n_preys = n_preys
        self.boundless = boundless
        self.fig = plt.figure(figsize=(7,5))
        self.ax = self.fig.add_subplot(projection='3d')
        self.trajectory = []
        self.trajectory_preys = []
        self.scat = self.ax.scatter([], [], [], color="red", label="Predator")
        self.scat2 = self.ax.scatter([], [], [], color="blue", label="Prey")
        self.scat3 = self.ax.scatter([], [], [], color="green", label="No sensors")
        self.quiv = self.ax.quiver([], [], [], [], [], [], color="red")
        self.quiv2 = self.ax.quiver([], [], [], [], [], [], color="blue")
        self.quiv3 = self.ax.quiver([], [], [], [], [], [], color="green")
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        if not self.boundless:
            # print('plot here')
            self.ax.set_xlim(0, x_lim + 0)
            self.ax.set_ylim(0, y_lim + 0)
            self.ax.set_zlim(0, z_lim + 0)
        plt.title(title)
        plt.legend()
        self.sphere_plots = []

    def no_sensor_preds(self, drones_no_sensors_list):
        self.drones_no_sensors = np.array(drones_no_sensors_list, dtype=int)

    def add_horizontal_plane(self):
        # Define the plane at z=0
        x = np.linspace(self.ax.get_xlim()[0], self.ax.get_xlim()[1], 100)
        y = np.linspace(self.ax.get_ylim()[0], self.ax.get_ylim()[1], 100)
        X, Y = np.meshgrid(x, y)
        Z = np.ones_like(X)

        # Create the polygon for the plane
        vertices = np.array([X.flatten(), Y.flatten(), Z.flatten()]).T
        faces = [[i, i + 1, i + 1 + 100, i + 100] for i in range(0, 100 * (100 - 1)) if (i + 1) % 100 != 0]
        plane = Poly3DCollection([vertices[face] for face in faces], color='gray', alpha=1)
        self.ax.add_collection3d(plane)

    def update_plot(self, pos_xs, pos_ys, pos_zs, pos_hxs, pos_hys, pos_hzs, pos_preys_xs, pos_preys_ys, pos_preys_zs, pos_preys_hxs, pos_preys_hys, pos_preys_hzs):
        mean_pos = (np.mean(pos_xs), np.mean(pos_ys), np.mean(pos_zs))
        self.trajectory.append(mean_pos)
        self.ax.plot(*zip(*self.trajectory), color='purple', label="Trajectory Predators")

        mean_pos_preys = (np.mean(pos_preys_xs), np.mean(pos_preys_ys), np.mean(pos_preys_zs))
        self.trajectory_preys.append(mean_pos_preys)
        self.ax.plot(*zip(*self.trajectory_preys), color='cyan', label="Trajectory Preys")

        if self.boundless:
            padding = 5
            all_xs = np.concatenate([pos_xs, pos_preys_xs])
            all_ys = np.concatenate([pos_ys, pos_preys_ys])
            all_zs = np.concatenate([pos_zs, pos_preys_zs])
            self.ax.set_xlim(all_xs.min() - padding, all_xs.max() + padding)
            self.ax.set_ylim(all_ys.min() - padding, all_ys.max() + padding)
            self.ax.set_zlim(all_zs.min() - padding, all_zs.max() + padding)

        boolean_mask = np.zeros(len(pos_xs), dtype=bool)
        boolean_mask[self.drones_no_sensors] = True

        pos_xs_no_sensors = pos_xs[boolean_mask]
        pos_ys_no_sensors = pos_ys[boolean_mask]
        pos_zs_no_sensors = pos_zs[boolean_mask]
        pos_hxs_no_sensors = pos_hxs[boolean_mask]
        pos_hys_no_sensors = pos_hys[boolean_mask]
        pos_hzs_no_sensors = pos_hzs[boolean_mask]

        pos_xs = pos_xs[~boolean_mask]
        pos_ys = pos_ys[~boolean_mask]
        pos_zs = pos_zs[~boolean_mask]
        pos_hxs = pos_hxs[~boolean_mask]
        pos_hys = pos_hys[~boolean_mask]
        pos_hzs = pos_hzs[~boolean_mask]

        self.scat3._offsets3d = (pos_xs_no_sensors, pos_ys_no_sensors, pos_zs_no_sensors)
        self.quiv3.remove()

        self.scat._offsets3d = (pos_xs, pos_ys, pos_zs)
        self.scat2._offsets3d = (pos_preys_xs, pos_preys_ys, pos_preys_zs)

        self.quiv.remove()
        self.quiv2.remove()
        if self.boundless:
            self.quiv = self.ax.quiver(pos_xs, pos_ys, pos_zs, np.cos(pos_hxs), np.cos(pos_hys), np.cos(pos_hzs), color="red", length=0.8)
            self.quiv2 = self.ax.quiver(pos_preys_xs, pos_preys_ys, pos_preys_zs, np.cos(pos_preys_hxs), np.cos(pos_preys_hys), np.cos(pos_preys_hzs), color="blue", length=0.8)
            self.quiv3 = self.ax.quiver(pos_xs_no_sensors, pos_ys_no_sensors, pos_zs_no_sensors, np.cos(pos_hxs_no_sensors), np.cos(pos_hys_no_sensors), np.cos(pos_hzs_no_sensors), color="green", length=0.8)
        else:
            self.quiv = self.ax.quiver(pos_xs, pos_ys, pos_zs, np.cos(pos_hxs), np.cos(pos_hys), np.cos(pos_hzs), color="red", length=0.2)
            self.quiv2 = self.ax.quiver(pos_preys_xs, pos_preys_ys, pos_preys_zs, np.cos(pos_preys_hxs), np.cos(pos_preys_hys), np.cos(pos_preys_hzs), color="blue", length=0.2)
            self.quiv3 = self.ax.quiver(pos_xs_no_sensors, pos_ys_no_sensors, pos_zs_no_sensors, np.cos(pos_hxs_no_sensors), np.cos(pos_hys_no_sensors), np.cos(pos_hzs_no_sensors), color="green", length=0.2)

        # # Remove existing spheres
        # for sphere in self.sphere_plots:
        #     sphere.remove()
        # self.sphere_plots = []

        # # Add new spheres for predators with sensors
        # for x, y, z in zip(pos_xs, pos_ys, pos_zs):
        #     u = np.linspace(0, 2 * np.pi, 100)
        #     v = np.linspace(0, np.pi, 100)
        #     x_sphere = 6 * np.outer(np.cos(u), np.sin(v)) + x 
        #     y_sphere = 6 * np.outer(np.sin(u), np.sin(v)) + y
        #     z_sphere = 6 * np.outer(np.ones(np.size(u)), np.cos(v)) + z
        #     self.sphere_plots.append(self.ax.plot_surface(x_sphere, y_sphere, z_sphere, color='r', alpha=0.05))

        plt.pause(0.0001)  # Adjust this value as needed for your visualization needs
