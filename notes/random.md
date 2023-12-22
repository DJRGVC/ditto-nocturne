# Random Fixes

These are some notes that I took in order to get Nocturne and DITTO set up. These are very rough -- definitely proceed with caution.

## Notes on Nocturne

- Change version of OpenAI Gym
- Have to use an absolute path versus a relative path
- Also have to change code in `get_helptext` within `pyvirtualdisplay/util.py` to run Python 3.7+ code.
- Have to install XQuartz and scipy to run?
- In the imitation learning tutorial and for `run_rllib`, you need to change `/checkpoint` to go away from absolute because we don’t know if we have access to the top folder (also b/c it doesn’t make sense to set checkpoint)
- Install `jupytext` to read in the tutorials
- Update tornado by reinstalling notebook

## Notes on DITTO:

- Change version of OpenAI Gym
- Set `WANDB_EXECUTABLE` to be equal to where Python is installed in conda
- Reinstall `gettext`
- Install atari-py==0.2.5
