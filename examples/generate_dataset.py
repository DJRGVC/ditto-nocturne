"""
Create clean roll-out data for DITTO World Model.
"""
from cfgs.config import PROJECT_PATH, get_scenario_dict, set_display_window
from matplotlib import pyplot as plt
from nocturne import Simulation, Action
import numpy as np

# Constants, specifically defining the scenario configuration
scenario_config = {
    'start_time': 0, # When to start the simulation
    'allow_non_vehicles': True, # Whether to include cyclists and pedestrians
    'max_visible_road_points': 10, # Maximum number of road points for a vehicle
    'max_visible_objects': 10, # Maximum number of road objects for a vehicle
    'max_visible_traffic_lights': 10, # Maximum number of traffic lights in constructed view
    'max_visible_stop_signs': 10, # Maximum number of stop signs in constructed view
}

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
    # Create simulation
    data_path = "/Users/cpondoc/Desktop/ditto-nocturne/nocturne/nocturne_mini/formatted_json_v2_no_tl_train/tfrecord-00003-of-01000_109.json"
    sim = Simulation(data_path, scenario_config)
    print("Loaded in simulation!")

    # Get traffic scenario at timepoint and set all vehicles to expert control
    scenario = sim.getScenario()
    all_vehicles = scenario.getVehicles()
    for i in range(len(all_vehicles)):
        all_vehicles[i].expert_control = True

    # Collect all images from the simulation
    imgs = collect_full_images(sim)
    print("Collected all of the images from the simulation!")

    # Save images
    save_to_npz(imgs)
    print("Saved to a file!")

if __name__ == '__main__':
    main()
