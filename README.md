# DITTO-Nocturne

This is work done as part of my tutorial at Oxford during the 2023 Michaelmas term. Specifically, I worked on applying the [DITTO](https://arxiv.org/abs/2302.03086) framework to the [Nocturne](https://github.com/facebookresearch/nocturne) RL environment.

## Outline of Work

Throughout the four-ish weeks that I worked on the project, I was able to accomplish the following milestones. For all reports, check out the `reports/` folder:

1. Train a simple behavior cloning agent (BC) that was able to learn to drive within the Nocturne environment. I was also able to generate [rollouts](https://github.com/cpondoc/ditto-nocturne/blob/main/examples/test_rollout.py) using this agent.

2. Train a [world model](https://worldmodels.github.io/) on Nocturne, using expert state action pairs from Waymo. See `nocturne-wm.md`.

3. Train a BC agent on the latents of the Nocturne world model. See `bc-nocturne.md` and `visualizing-latents.md`.

4. Train a actor-critic (AC) policy using the Nocturne world model. See `finishing-validation.md`.

## How to Set-Up

First, set up [Nocturne](https://github.com/facebookresearch/nocturne). Then, the submodule should already be set up to point to [Daphne's branch](https://github.com/facebookresearch/nocturne/tree/ca024c35baf0aecb41fe928d025efe5808c3f0c8), which contains tutorials about how to use Nocturne.

### Setting up the Imitation Learning Agent

From the tutorials, run the [Imitation Learning tutorial](https://github.com/facebookresearch/nocturne/blob/ca024c35baf0aecb41fe928d025efe5808c3f0c8/docs/tutorials/03_imitation_learning.md) (`03_imitation_learning.md`) and train a basic Imitation Learning agent. Finally, save the weights, adjust the line the calls `model.load_state_dict` to load in the correct path. From there, you should be able to run `test_rollout.py`.

## Acknowledgments

Special thanks to [Branton Demoss](https://ori.ox.ac.uk/people/branton-demoss/), who advised my work over the term.
