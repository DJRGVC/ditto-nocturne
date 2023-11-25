"""
Create clean roll-out data for DITTO World Model.
"""
from cfgs.config import PROJECT_PATH, get_scenario_dict, set_display_window
from matplotlib import pyplot as plt
from nocturne import Simulation, Action
import numpy as np
import os
from scipy import stats

# Constants for dealing with expert actions
ACC_RANGE = (-6, 6)
STEERING_RANGE = (-1, 1)
TIME_RANGE = 90

# Constants for dealing with scenarios
FILE_PATH = "/Users/cpondoc/Desktop/ditto-nocturne/nocturne/nocturne_mini/formatted_json_v2_no_tl_valid/"
scenario_config = {
    'start_time': 0, # When to start the simulation
    'allow_non_vehicles': True, # Whether to include cyclists and pedestrians
    'max_visible_road_points': 10, # Maximum number of road points for a vehicle
    'max_visible_objects': 10, # Maximum number of road objects for a vehicle
    'max_visible_traffic_lights': 10, # Maximum number of traffic lights in constructed view
    'max_visible_stop_signs': 10, # Maximum number of stop signs in constructed view
}

def find_full_timestep_vehicle(moving_vehicles, scenario):
    """
    Find a vehicle that takes actions across the entire episode.
    """
    # Iterate through all moving vehicles
    for vehicle in moving_vehicles:
        satisfies_all = True

        # Break out if there is a non-existent action at a time step
        for time in range(TIME_RANGE):
            if (scenario.expert_action(vehicle, time) is None):
                satisfied_all = False
                break

        # Return if it actually went through each time step
        if (satisfies_all):
            return vehicle

def display_img(img):
    """
    Helper function that can be used ad-hoc to display image.
    """
    plt.imshow(img, interpolation='nearest')
    plt.show()

def collect_full_images(sim):
    """
    Iterate and collect images of timesteps for fully observable case.
    """
    # Create an array of images, set step size and number of steps
    imgs = []
    steps, dt = 90, 0.1

    # Training loop: take a step, get the image, add to total array
    for i in range(steps):
        sim.step(dt)
        scenario = sim.getScenario()
        img = scenario.getImage(
            img_width=2000, # 500?
            img_height=2000, # 500?
            padding=50.0, # 10?
            draw_target_positions=False,
        )
        imgs.append(img)

    # Stack them all together into an individual array
    imgs = np.stack(imgs)
    return imgs

def save_to_npz(imgs):
    """
    Save all the data to an NPZ file, which is used for episodes.
    """
    with open("examples/episodes/sample.npz", "wb") as f:
        np.savez(f, images=imgs)

def main():
    """
    Run through each simulation, generate data
    """
    # Iterate through all trajectories
    files = os.listdir(FILE_PATH)
    print("Total number of files: " + str(len(files)))
    for file in files:

        # Create simulation
        if (file != "valid_files.json"):
            data_path = FILE_PATH + file
            sim = Simulation(data_path, scenario_config)
            print("> Loaded in simulation " + str(file))

            # Get traffic scenario at timepoint and set all vehicles to expert control
            scenario = sim.getScenario()
            all_vehicles = scenario.getVehicles()
            for i in range(len(all_vehicles)):
                all_vehicles[i].expert_control = True
            print("> Set all vehicles to expert control")

            # Get all moving vehicles and find a full timestep vehicle
            objects_that_moved = scenario.getObjectsThatMoved()
            moving_vehicles = [obj for obj in scenario.getVehicles() if obj in objects_that_moved]
            ft_vehicle = find_full_timestep_vehicle(moving_vehicles, scenario)
            print("> Found a vehicle with actions at all timesteps")

            # Get discretized set of actions
            actions = discretized_expert_actions(ft_vehicle)
            print("> Got discretized expert actions")

            # Formatting
            print("")

    # Create simulation
    '''

    # Collect all images from the simulation
    imgs = collect_full_images(sim)
    print("Collected all of the images from the simulation!")

    # Get a random vehicle that moved
    objects_that_moved = scenario.getObjectsThatMoved()
    moving_vehicles = [obj for obj in scenario.getVehicles() if obj in objects_that_moved]
    ego_vehicle = moving_vehicles[1]

    # Get their action trajectory
    for time in range(90):
        print("Action " + str(time) + ":", scenario.expert_action(ego_vehicle, time))

    # Save images
    save_to_npz(imgs)
    print("Saved to a file!")'''

if __name__ == '__main__':
    main()
