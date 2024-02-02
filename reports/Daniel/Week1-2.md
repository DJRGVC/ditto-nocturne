# Week One-Two: Getting up to speed with Nocturne and Ditto

## General Work Outline

Following my first meeting with Branton regarding our work together on this project, we came up with the following list of tasks to complete while I set up my ditto-nocturne working environment:

1. First set up SSH to work remotely on his system, and fork a new repo for this project
2. Make sure both DITTO and Nocturne were running smoothly, and go through some of their code examples
3. Rerun some of Chris's work from last term
4. And, as the main task, compare *all* expert rollouts to current step of training rather than simply the next step of the current nearest expert rollout.

## Setting up SSH 

I am using VSCode's SSH extension to connect to Branton's system. This is a very convenient way to work remotely, as it allows me to use my local machine's resources to run the code, while still being able to access the files and run the code on Branton's system.

## Initiating Nocturne and DITTO Environments

This required downloading some extranous packages, and altering some of the base configuration that comes with the Nocturne version used as a subpackage for this project. More information about the general changes I made are available [here](https://github.com/DJRGVC/ditto-nocturne/blob/main/reports/Installation/tips.md).

### Brief Introduction to Relevant Frameworks

[DITTO](https://github.com/brantondemoss/DITTO) is a codebase that allows training a World Model (WM) based on episodes of a strong PPO agent playing Breakout. The episodes are stored as `.npz` files and contain information such as state images, actions taken, rewards, and terminal states.

[Nocturne](https://github.com/facebookresearch/nocturne) is a codebase that provides a dataset of traffic scenes from Waymo. Each scene consists of road objects, vehicles, roads, and traffic light states. These scenes can be wrapped as Nocturne simulations, which are discrete traffic scenarios captured at specific time points.

To generate Nocturne episodes for training a WM, the following steps were taken:
1. The traffic simulation was loaded, and all vehicles were set to be expertly controlled, mimicking real-life movements.
2. An "ego vehicle" was identified for each timestep, which is a vehicle that took an action.
3. The action and an image of the scene from the ego vehicle's perspective were retrieved.
4. The data was saved as `.npz` files for training.

By collecting Nocturne data in a format similar to DITTO, it was possible to train a WM using the Nocturne episodes. The WM is trained using the [DreamerV2](https://arxiv.org/abs/2010.02193) algorithm, which is a combination of a Recurrent State Space Model (RSSM) and a VAE. The RSSM is trained to predict the next state and reward given the current state and action, while the VAE is trained to reconstruct the input image.

## Training the World Model & Altering Expert Rollouts

*This is a work in progress, and will carry on to a weekend task*

## Next Steps + Questions

Completed steps for this project include:

1. Complete the setup of SSH to work remotely on Branton's system and fork a new repo for the project.
2. Ensure that both DITTO and Nocturne are running smoothly by going through their code examples.
3. Rerun some of Chris's work from last term to familiarize myself with the project.

However, the main task of comparing *all* expert rollouts to the current step of training rather than simply the next step of the current nearest expert rollout is still in progress. This will be the main focus of my work in the coming days.

Some questions to I will be considering as I continue to work on this project include:

1. How can I compare all expert rollouts to the current step of training? What exactly does this entail from a technical standpoint?
2. Will it even perform better than the current method of comparing the next step of the current nearest expert rollout? What are the potential benefits and drawbacks of this approach?
3. After making this change, what are the next steps in the project? What are obvious next tasks that I should be considering?

##*Cheers*
