# Finishing the Validation Function

For my final (!) tutorial on December 8, 2023, I was able to fix the issues with the `validate` function.

## The Existing Issue
The biggest issue that I had with previously running the code was generating the image observations from the Nocturne environment. More specifically, I kept getting the error below:

```
libGL error: MESA-LOADER: failed to open swrast: /usr/lib/dri/swrast_dri.so: cannot open shared object file: No such file or directory (search paths /usr/lib/x86_64-linux-gnu/dri:\$${ORIGIN}/dri:/usr/lib/dri, suffix _dri)
libGL error: failed to load driver: swrast
Failed to create an OpenGL context for this window
sfml-graphics requires support for OpenGL 1.1 or greater
Ensure that hardware acceleration is enabled if available
Failed to create texture, its internal size is too high (2048x2048, maximum is 0x0)
Impossible to create render texture (failed to create the target texture)
Trying to access the pixels of an empty image
Segmentation fault (core dumped)
```

While I thought of this initially as a drivers issue, I realized that I had all of the correct drivers installed by using the `find` command on Linux. However, it seemed as though the program wasn't looking in the same location as where they were installed (i.e., the program was looking in `/usr/lib/dri`). Thus, what I did was simply copy over the appropriate static files into this drivers folder.

I also changed some OS environment variables. For instance:

- I ran `export LIB_GL_ALWAYS_INDIRECT=1`. This variable is related to the OpenGL library, and setting it to 1 typically enables indirect rendering, which is when direct rendering is causing issues or when compatibility with certain graphics configurations is needed.

Overall, I am still trying to recreate how I was able to get it running -- it did run -- but this was the general process I had to undertake.

## Finishing up Validation

Finally, now that I could access the image, I completed the `validate` function. In particular, I was able to generate the image and then turn it into a latent state using the `prep_obs` function, which feeds the image into the world model. This can be found in the `get_nocturne_features` function:

```python
def get_nocturne_features(self, env):
    """
    Abstract the latent features retrieval to a new function.
    """
    # Reset the environment
    init_obs = env.reset()

    # Get a vehicle to focus on
    objects_that_moved = env.get_objects_that_moved()
    moving_vehicles = [obj for obj in env.env.get_vehicles() if obj in objects_that_moved]
    vehicle = self.find_full_timestep_vehicle(moving_vehicles, env.env)

    # Get the observation -- also alter dimensions to make sure it works
    init_obs = env.scenario.getConeImage(
        vehicle,
        view_dist=80,
        view_angle=np.pi * (120 / 180),
        head_angle=0.0,
        draw_target_position=False
    )

    # Convert to grayscale and 128
    init_obs = np.mean(init_obs, axis=2, dtype=np.uint8)
    init_obs = np.array(cv2.resize(init_obs, dsize=(128, 128), interpolation=cv2.INTER_AREA))

    # Now going to stack dimensions
    init_obs = np.expand_dims(init_obs, axis=2)
    init_obs = np.expand_dims(init_obs, axis=0)

    # Get features and vehicle ID
    features = env.prep_obs(init_obs)
    return features, vehicle
```

From there, I fed in the latent state observation into the model, which was then returned a singular discrete action.

```python
# Get action recommendation and take step in environment
s_action = self.get_action2(features, policy=policy)
actions.extend(s_action.tolist())
features, r, done, info, timestep = env.step(s_action, vehicle=vehicle, save_path=folder_name, game=n_complete)
```

Since the action space of Noturne takes values for both the acceleration as well as the steering angle, I then wrote a function to turn the singular action back into the acceleration and steering angle pairs:

```python
def reverse_discretized_action(self, action_index):
    """
    Function to convert from discretized index to acceleration and steering angle
    """
    # Calculate steering index
    steering_index = action_index % STEERING_BUCKETS
    steering_angle = (steering_index * 0.1) + STEERING_RANGE[0]

    # Calculate acceleration index
    acc_index = action_index // STEERING_BUCKETS
    acceleration = acc_index + ACC_RANGE[0]

    return acceleration, steering_angle
```

Finally, I repeated for `n_games`, and recorded validation statistics!

```python
# Keep iterating and log rewards
for i in range(n_envs):
    if (vehicle.id in done):
        if (done[vehicle.id]):
            n_complete += 1
            if (vehicle.id not in r):
                rewards.append(0)
            else:
                rewards.append(r[vehicle.id])

            # Update progress and reset
            pbar.update(1)
            features, vehicle = self.get_nocturne_features(env)
    else:
        features, vehicle = self.get_nocturne_features(env)
```

Overall, I was able to get some images from the trajectory, but can debug to make it more consistent.
