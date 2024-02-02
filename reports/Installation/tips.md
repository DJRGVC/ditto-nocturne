# Installation Adjustments for Running on Ubuntu 22.04.2 LTS

This is a collection of general fixes to known errors when running nocturne-ditto on 22.04.2.

## For Nocturne

- use Miniconda as a package intallation system, and create a different environment for Nocturne and DITTO.
- In the imitation learning tutorial and for `run_rllib`, you need to change `/checkpoint` to go away from absolute because we don’t know if we have access to the top folder (also b/c it doesn’t make sense to set checkpoint)
- Install `jupytext` to read in the tutorials
- Update tornado by reinstalling notebook
- Change version of OpenAI Gym
- Also have to change code in `get_helptext` within `pyvirtualdisplay/util.py` to run Python 3.7+ code.
- Have to install XQuartz and scipy to run.
- Make sure you check the config folder, as there are a few places where paths need to be changed.

## For DITTO:

- Set `WANDB_EXECUTABLE` to be equal to where Python is installed in conda
- Reinstall `gettext`
- Change version of OpenAI Gym
- Install atari-py==0.2.5
