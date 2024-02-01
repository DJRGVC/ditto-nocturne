# DITTO-Nocturne

This work is done collaboratively with Branton DeMoss (https://brantondemoss.com), a mentor and creator of the [DITTO](https://arxiv.org/abs/2302.03086) framework (https://arxiv.org/abs/2302.03086) central to this project. This repo serves as a continuation to work done by [Chris Pondoc](https://www.chrispondoc.com/tutorial-oxford/) during the Oxford University Michealmas term of 2023. We are applying the DITTO framework to the [Nocturne](https://github.com/facebookresearch/nocturne) RL environment, alongside a few other benchmark environments. 

## Outline of Work

Here is a brief summary of the work Chris Pondoc did before I began working on this project: (For all reports, including Chris's, check out the `reports/` folder)

1. Train a simple behavior cloning agent (BC) that was able to learn to drive within the Nocturne environment. I was also able to generate [rollouts](https://github.com/DJRGVC/ditto-nocturne/blob/main/examples/test_rollout.py) using this agent.

2. Train a [world model](https://worldmodels.github.io/) on Nocturne, using expert state action pairs from Waymo. See `nocturne-wm.md`.

3. Train a BC agent on the latents of the Nocturne world model. See `bc-nocturne.md` and `visualizing-latents.md`.

4. Train an actor-critic (AC) policy using the Nocturne world model. See `finishing-validation.md`.

Below are my major updates to the program, chronologically ordered by work period:

1. Comparing *all* expert rollouts to current step of training rather than simply the next step of the current nearest expert rollout. [Week 1-2 (Jan 22 - Feb 3, 2024)](https://github.com/DJRGVC/ditto-nocturne/blob/main/reports/Daniel/Week1-2.md)

## Acknowledgments

Thanks to [Branton Demoss](https://ori.ox.ac.uk/people/branton-demoss/) who graciously allowed me to participate in tutorials alongside Chris Pondoc during the Michealmas Term. Also, thanks to Chris for spending time getting me up to speed on the progress he had made over the Christmas break preceding my work on this project.
