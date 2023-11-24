# DITTO-Nocturne

Applying the DITTO model to the Nocturne RL environment.

## How to Set-Up

### Basic Example (IL Agent)

First, set up [Nocturne](https://github.com/facebookresearch/nocturne). Then, the submodule should already be set up to point to [Daphne's branch](https://github.com/facebookresearch/nocturne/tree/ca024c35baf0aecb41fe928d025efe5808c3f0c8), which contains tutorials about how to use Nocturne.

From the tutorials, run the [Imitation Learning tutorial](https://github.com/facebookresearch/nocturne/blob/ca024c35baf0aecb41fe928d025efe5808c3f0c8/docs/tutorials/03_imitation_learning.md) (`03_imitation_learning.md`) and train a basic Imitation Learning agent. Finally, save the weights, adjust the line the calls `model.load_state_dict` to load in the correct path. From there, you should be able to run `test_rollout.py`.
