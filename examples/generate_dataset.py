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
ACC_RANGE, ACC_BUCKETS = (-6, 6), 13
STEERING_RANGE, STEERING_BUCKETS = (-1, 1), 21
TIME_RANGE = 90

# Constants for dealing with scenarios (CHANGE TO YOUR FILE PATH!)
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
            expert_action = scenario.expert_action(vehicle, time)
            if (expert_action is None):
                satisfies_all = False
                break
            else:
                if (np.isnan(expert_action.acceleration) or np.isnan(expert_action.steering)):
                    satisfies_all = False
                    break

        # Return if it actually went through each time step
        if (satisfies_all):
            return vehicle

    # If no match found, return None
    return None

def discretized_expert_action(ft_vehicle, scenario):
    """
    Given a set of expert action dictionaries, return discrete action.
    """
    # For each time step, grab expert action
    action_indices = []
    for time in range(TIME_RANGE):
        expert_action = scenario.expert_action(ft_vehicle, time)

        # Calculate index of acceleration
        acc_index = round(expert_action.acceleration) - ACC_RANGE[0]
        acc_index = min(max(acc_index, 0), ACC_BUCKETS)

        # Calculate index of steering
        steering_index = int((round(expert_action.steering, 1) - STEERING_RANGE[0]) / .1)
        steering_index = min(max(steering_index, 0), STEERING_BUCKETS)

        # Put together to calculate final index
        final_index = acc_index * STEERING_BUCKETS + steering_index
        action_indices.append(final_index)

    return action_indices

def display_img(img):
    """
    Helper function that can be used ad-hoc to display image.
    """
    plt.imshow(img, cmap='gray')
    plt.show()

def collect_full_images(sim, vehicle):
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

        # Specifically, get a cone image to emulate with just one vehicle
        img = scenario.getConeImage(
            source=vehicle,
            view_dist=80,
            view_angle=np.pi * (120 / 180),
            head_angle=0.0,
            img_width=1600, # 500?
            img_height=1600, # 500?
            padding=50.0, # 10?
            draw_target_position=False,
        )

        # [TO-DO] Check if we need to change to grayscale?
        img = np.mean(img, axis=2, dtype=np.uint8)
        img = np.expand_dims(img, axis=2)
        display_img(img)
        imgs.append(img)

    # Stack them all together into an individual array
    imgs = np.stack(imgs)
    return imgs

def save_to_npz(file, imgs, actions, resets):
    """
    Save all the data to an NPZ file, which is used for episodes.
    """
    with open("examples/episodes/" + file[:-5] + ".npz", "wb") as f:
        np.savez(f, images=imgs, actions=actions, resets=resets)

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

            # Only run if we know we have a full timestep vehicle
            if (ft_vehicle != None):
                print("> Found a vehicle with actions at all timesteps")

                # Get discretized set of actions
                actions = discretized_expert_action(ft_vehicle, scenario)
                resets = [False] * len(actions)
                resets[-1] = True
                print("> Got discretized expert actions")

                # [TO-DO] Change to cone images?
                imgs = collect_full_images(sim, ft_vehicle)
                print("> Got all images")

                # Save file
                save_to_npz(file, imgs, np.array(actions), np.array(resets))
                print("> Saved to a file!")
                print("")

if __name__ == '__main__':
    main()
