# Setting up a World Model for Nocturne

For the tutorial on November 27, 2023, I tried to set up a World Model (WM) for Nocturne. For this session and future sessions, I'll be keeping track of all my notes for accountability as well as to go over my work.

## Gathering Data

### Primer on DITTO and Nocturne Episode Data

Given the [DITTO](https://github.com/brantondemoss/DITTO) codebase, you can train a WM based on episodes of a strong PPO agent playing Breakout.
Each of these files were `.npz` files and contained information containing images of the state, actions taken by the player, the total rewards gained, whether the game was reset or not, and if the state was a terminal state. Thus, I decided to try to collect Nocturne data that was formatted in much the same way.

As part of the [Nocturne](https://github.com/facebookresearch/nocturne) codebase, you can download a dataset from Waymo that contains traffic scenes. Each traffic scene consists of a name, road objects and vehicles in the scene, roads, and the states of traffic lights. We can then wrap each of these traffic scenes as a Nocturne simulation, which consists of discretezed traffic scenarios that are snapshots of a traffic situation at a paricular timepoint. Each simulation lasts for 9 seconds, and they are discretized into step sizes of 0.1 seconds, meaning there are 90 total timesteps.

### Creating the dataset

For each simulation, I then followed the below procedure to generate episodes:
1. First, I loaded in the traffic simulation, and set all vehicles to be expertly controlled, which corresponds to how the cars actually moved in the real-life dataset.
2. For all of the vehicles that moved in the traffic scene, I found a vehicle that was moving (i.e., took an action) at each timestep across the entire episode. I labeled this vehicle the *ego vehicle*.
3. After finding an ego vehicle, I then retrieved its action at each timestep (Nocturne has functionality that allows you to retrieve the expert action at each timestep), and also obtained an image of the scene *from the perspective of the ego vehicle*.
4. Finally, I saved this data into a `.npz` file to use for training.

### A Note on the Nocturne Action Space

The action set for a vehicle consists of three components: acceleration, steering, and the head angle. Actions are discretized based on an upper and lower bound.

For the data I was looking through (the mini dataset), all of the head angles were set to 0. Thus, I only had to decide on the discretization for the acceleration and steering. In both instances, I took the following steps, I collected all of the corresponding metrics across all vehicles in all traffic scenes at each time step. Then, I generated a histogram and visualized where appropriate upper and louwer bounds would be.

After the above, I found that the acceleration could be bounded by [-6, 6], with 13 discrete buckets, and that the steering could be bounded by [-1, 1] with 21 discrete buckets.

Finally, when reporting the final action, I effectively mapped the two actions into a single action. I did this by finding an index for both the acceleration and steering index based on the bounds and discrete buckets. Since this effectively like indexing into a 2D-array, I then imagined "flattening" the 2D-array and finding the corresponding index in the now 1D-array.
