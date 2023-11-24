# General dependencies
from pathlib import Path
import numpy as np
import os
import imageio
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import warnings
from cfgs.config import PROJECT_PATH, get_scenario_dict, set_display_window

# Nocturne dependencies
from examples.imitation_learning import waymo_data_loader
from examples.imitation_learning.filters import MeanStdFilter
from torch.distributions.categorical import Categorical
from nocturne import Simulation

# Configuration for training and the scenario
train_config = {
    'batch_size': 512,
    'num_workers': 0, # Use a single worker
    'hidden_layers': [1025, 256, 128], # Model used in paper
    'action_discretizations': [15, 43], # Number of discretizations (acc, steering)
    'action_bounds': [[-6, 6], [-.7, .7]], # Bounds for (acc, steering)
    'lr': 1e-4,
    'num_epochs': 5,
    'samples_per_epoch': 50_000,
}

scenario_config = {
    'start_time': 0,
    'allow_non_vehicles': False,
    'max_visible_road_points': 500,
    'sample_every_n': 1,
    'road_edge_first': False
}

# Class for Behavioral Cloning Agent
class BehavioralCloningAgent(nn.Module):
    """ Simple Behavioral Cloning class. """
    def __init__(self, num_inputs, config):
        super(BehavioralCloningAgent, self).__init__()
        self.num_states = num_inputs
        self.hidden_layers = config['hidden_layers']
        self.action_discretizations = config['action_discretizations']
        self.action_bounds = config['action_bounds']

        # Create an action space
        self.action_grids = [
            torch.linspace(a_min, a_max, a_count, requires_grad=False)
                for (a_min, a_max), a_count in zip(
                    self.action_bounds, self.action_discretizations)
        ]
        self._build_model()

    def _build_model(self):
        """Build agent MLP"""

        # Create neural network model
        self.neural_net = nn.Sequential(
            MeanStdFilter(self.num_states), # Pass states through filter
            nn.Linear(self.num_states, self.hidden_layers[0]),
            nn.Tanh(),
            *[
                nn.Sequential(
                    nn.Linear(self.hidden_layers[i],
                                self.hidden_layers[i + 1]),
                    nn.Tanh(),
                ) for i in range(len(self.hidden_layers) - 1)
            ],
        )

        # Map model representation to discrete action distributions
        pre_head_size = self.hidden_layers[-1]
        self.heads = nn.ModuleList([
            nn.Linear(pre_head_size, discretization)
            for discretization in self.action_discretizations
        ])

    def forward(self, state):
        """Forward pass through the BC model.

            Args:
                state (Tensor): Input tensor representing the state of the environment.

            Returns:
                Tuple[List[Tensor], List[Tensor], List[Categorical]]: A tuple of three lists:
                1. A list of tensors representing the actions to take in response to the input state.
                2. A list of tensors representing the indices of the actions in their corresponding action grids.
                3. A list of Categorical distributions over the actions.
            """

        # Feed state to nn model
        outputs = self.neural_net(state)

        # Get distribution over every action in action types (acc, steering, head tilt)
        action_dists_in_state = [Categorical(logits=head(outputs)) for head in self.heads]

        # Get action indices (here deterministic)
        # Find indexes in actions grids whose values are the closest to the ground truth actions
        actions_idx = [dist.logits.argmax(axis=-1) for dist in action_dists_in_state]

        # Get action in action grids
        actions = [
            action_grid[action_idx] for action_grid, action_idx in zip(
                self.action_grids, actions_idx)
        ]

        return actions, actions_idx, action_dists_in_state

def save_image(img, output_path='./img.png'):
    """
    Make a single image from the scenario.
    """
    dpi = 100
    height, width, depth = img.shape
    figsize = width / float(dpi), height / float(dpi)
    plt.figure(figsize=figsize, dpi=dpi)
    plt.axis('off')
    plt.imshow(img)
    plt.savefig(output_path)
    plt.close()
    print('>', output_path)

def construct_state(scenario, vehicle, view_dist=80, view_angle=np.radians(180)):
    """Construct the full state for a vehicle.
    Args:
        scenario (nocturne_cpp.Scenario): Simulation at a particular timepoint.
        vehicle (nocturne_cpp.Vehicle): A vehicle object in the simulation.
        view_dist (int): Viewing distance of the vehicle.
        view_angle (int): The view cone angle in radians.
    Returns:
        state (ndarray): The vehicle state.
    """
    ego_state = scenario.ego_state(
        vehicle
    )
    visible_state = scenario.flattened_visible_state(
        vehicle,
        view_dist=view_dist,
        view_angle=view_angle
    )
    return np.concatenate((ego_state, visible_state))

def main():
    """
    Function to load in model, evaluate, and save images
    """
    set_display_window()

    # First, load in model
    model = BehavioralCloningAgent(
        num_inputs=35110,
        config=train_config
    )
    model.load_state_dict(torch.load("nocturne/docs/tutorials/weights/model_weights.pth"))
    print("Model loaded!")

    # Set up the simulation and save initial image
    stacked_state = defaultdict(lambda: None)
    sim = Simulation("nocturne/docs/tutorials/data/example_scenario.json", scenario_config)
    scenario = sim.getScenario()
    img = scenario.getImage(
        img_width=2000,
        img_height=2000,
        padding=50.0,
        draw_target_positions=True,
    )
    save_image(img, "examples/rollout/imgs/0.png")

    # Get all vehicles and vehicles that moved
    vehicles = scenario.getVehicles()
    objects_that_moved = scenario.getObjectsThatMoved()

    # Set all vehicles to expert control mode
    for obj in scenario.getVehicles():
        obj.expert_control = True

    # If a model is given, model will control vehicles that moved
    controlled_vehicles = [obj for obj in vehicles if obj in objects_that_moved]
    for veh in controlled_vehicles: veh.expert_control = False

    # Step through the simulation
    for time in range(1, 91):
        for veh in controlled_vehicles:

            # Get the state for vehicle at timepoint
            state = construct_state(scenario, veh)

            # Stack state
            if stacked_state[veh.getID()] is None:
                stacked_state[veh.getID()] = np.zeros(len(state) * 5, dtype=state.dtype)
            # Add state to the end and convert to tensor
            stacked_state[veh.getID()] = np.roll(stacked_state[veh.getID()], len(state))
            stacked_state[veh.getID()][:len(state)] = state
            state_tensor = torch.Tensor(stacked_state[veh.getID()]).unsqueeze(0)

            # Pred actions
            actions, _ , _ = model(state_tensor)

            # Set vehicle actions (assuming we don't have head tilt)
            veh.acceleration = actions[0]
            veh.steering = actions[1]

        # Step the simulator and save as image
        sim.step(0.1)
        scenario = sim.getScenario()
        img = scenario.getImage(
            img_width=2000,
            img_height=2000,
            padding=50.0,
            draw_target_positions=True,
        )
        save_image(img, 'examples/rollout/imgs/' + str(time) + '.png')


if __name__ == "__main__":
    main()
    # Save all files to a movie
    ims = []
    for i in range(91):
        ims.append(imageio.imread('examples/rollout/imgs/' + str(i) + '.png'))
    imageio.mimwrite('examples/rollout/video/test.mp4', ims, fps=10)
