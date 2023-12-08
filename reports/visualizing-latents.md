# Visualizing Steps in the Latent Space

For the tutorial on December 5, 2023, I was able to visualize the steps my Behavior Cloning (BC) agent was able to take in the latent space.

## The Existing Issue

From the tutorial the previous week, I was able to train a BC agent on the latents from the world model of Nocturne. However, I had a hard time visualizing the rollout, since I was able to retrieve the latent states, but i wasn't able to see any features outputted.

What I failed to realize was that the `state` being returned from taking a step in the world model was exactly what I needed to pass in to recreate the image. Specifically, all I had to do was index into the first dimension of the state (since the function still returned batched observation) so I could retrieve the properly sized latent that could be passed into the decoder. From there, I was able to write a function for the decoder that was able to recover an image just from the latents (no existing function with this sole job), and I was able to generate what the world model would look like after taking an action in the latent space:

<p align="center">
  <img src="imgs/wm-latent-step.tiff" />
  <br />
  Figure 1: A reconstruction of the visualization of the Nocturne environment from a step in the latent space of the world model.
</p>

## Future Work

Given the noise of the image, it's most clear that the bulk of the work is in training a better world model with larger image sizes and more data points.
