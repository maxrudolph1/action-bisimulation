# Repo for Action-bisim work
## File Structure
- `habitat_representation` is the code for training and evaluating bisim in habitat
- `nav2d_representation` is the code that houses nav2d gym environment and autoencoder modules
- `scripts` houses code for actually collecting data and training in the nav2d task

## dependencies:
- conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
- conda install -c conda-forge gym
- conda install -c conda-forge matplotlib
- conda install -c conda-forge tqdm
- conda install -c conda-forge h5py
- conda install -c anaconda pyyaml
- pip install -e .
- pip install tianshou
- cd nav2d_representation/nav2d_gymnasium
- pip install -e .


# Experiments Outline


| ExpName | Runs Trained | Frozen Encoder | Offline Goal Information | offline method | offline path |
|-----------|-----|----------------------|-------| --| ---|
| vanilla<#> | 3| no| n/a| none | none |
|    vae_fe_ng<#>  | 3 | No   | No   | VAE|ae_0.0001_lr_fine_model_annealing_changed/autoencoder_20000_clean.pt|
|  vae_ue_ng<#>     | 3 | Yes  | No   | VAE |ae_0.0001_lr_fine_model_annealing_changed/autoencoder_20000_clean.pt|
|ms_fe_ng<#>| 3|No| No| ControlBisim|ae_0.0001_lr_fine_model_annealing_changed/multi_step_final_encoder_clean.pt|
|ms_ue_ng<#>| 3|Yes| No| ControlBisim|ae_0.0001_lr_fine_model_annealing_changed/multi_step_final_encoder_clean.pt|
|ss_ue_ng<#>| 3|No| No| SingleStep|ae_0.0001_lr_fine_model_annealing_changed/single_step_final_encoder_clean.pt|
|ss_ue_ng<#>| 3|Yes| No| SingleStep|ae_0.0001_lr_fine_model_annealing_changed/single_step_final_encoder_clean.pt|

# Install 

For running experiments in habitat, you will need 3 key libraries: habitat-sim, habitat-lab, and habitat-baselines. I have made significant changes to habitat-lab and habitat-baselines so we will just use the versions in this repo.

1. Install habitat-sim and conda environment habiat-sim repo (info at this [link](https://github.com/facebookresearch/habitat-sim)). 
2. Install habitat-lab
```
cd habitat_representation/habitat-lab
pip install -e habitat-lab
```
3. install habitat-baselines
```
pip install -e habitat-baselines 
```
4. All other dependencies should be install automatically with the three installs from above
5. Now you must download the data. Just talk to max for this one. It depends on which experiments youre running.



# 11/07/2023
- Run RL experiments
- get rid of dynamics model
- run more verifications
	- change activations
	- run more ground truth models with diff hyper parameters
	- add more obstacles
- run corridor experiments

# 12/13/2023
### Running RL in habitat
- The goal is to use the pretrained encoders for RL training. 
- Collect data from random policies in the environment 
- Logging the meta data of the interaction data collection (scenes, random actions, )
- Can increase the number of environments to train on but this leads to other bugs ssomewhere else....
	- added in an unsqueeze somewhere in there...that was a maxadd
	- Increasing # of environmebts speeds up things significantly but the learning curve is slightly different
### Scenes
- The episodes are held in `data/datasets/pointnav` 
	- the episodes are made from json files that contain dictionaries with starting pose, goal pose, and the scene_id. 
	- The `train` folder contains all the training episodes and the `val` folder contains all the validation episodes. It is NOT guaranteed that these are the same thing. The validation episodes (and scenes) are not found anywhere in the training set. So there is no possibility of memorizing the home. 

## Current Experiments
- Training RL policies in habitat. Now that the pretrained encoders are actually being used. 
	- `ss_bisim_gibson_64_d`
	- `ss_bisim_gibson_64_d_1`
	- `ms_bisim_gibson_64_d`

# 12/14/2023
## Why is pre-trained encoder not performing better than vanilla
- Currently, with pretrained MS and SS encoders, the policies are not learning any fast than without them. 
- *QUESTION*: is the problem coming from the RL training or the pretraining?
	- we can easily figure this out by pretraining a bad encoder and seeing how RL fairs
	- Two experiments: completely zero'd out encoder and l1 penalty pretrained encoder
	- *Zeroed Out Encoder* (`completly_zero_encoder`): literally just multiply the visual encoder by 0. This produced policies that could still solve habitat environments with ~65%
	- unfrozen l1_penalty  (`l1ss_bisim_gibson_partial_64_d`) pretraining had little affect on performance
	- frozen l1_pentalty  (`frozen_l1ss_bisim_gibson_partial_64_d`) performed the same as zeroed out encoder
- Because the inputs to the policy are depth image + goal sensor, this leads me to believe that the tasks are too easy and don't require the visual goal

- What are the differences between old results and now?
	- Slightly different scene datasets
	- Using 64x64 instead of 256x256
	- 

# 12/15/2023
- Looking at ImageGoal task
- Documented the config structure in [[Habitat File Structure]] 
- Tried to build new episodes that were longer
- generated plenty of episodes but am not able to control the Geodesic/Euclidean ratio very precisely. It is more sampling based

# 12/18/2023
- Discovered that on the partial gibson dataset (that I filtered out) and using 64x64 images, the pointnav policies could be learned with the goal sensor alone
- Want to validate the necessity for the visual policies, performed ablations on differently sized observation and full gibson dataset. 
### Trained

- Attempted to train RL policies on entire gibson dataset
##### results can be found in the rl_experiments folder 
| Image Size | Image Type | num policies trained | 
| -------- | -------- | -------- | 
| 64x64| rgb | 2 | 
| 64x64| rgbd | 1 |
| 64x64| depth | 1 |
| 128x128| rgb | 2 |
| 128x128| rgbd | 2 |
| 128x128| depth | 2 |
| 256x256 | rgb | 2 |
| 256x256 | rgbd | 1 |
| 256x256 | depth | 1 |
| in 64x64 sim | blind | 2 |

It seems that as we increase the resolution of the simulator, the images become more important. there is not much gained by going from 128 -> 256 but there is a big jump between 64 -> 128 in terms of success and SPL![[Screenshot 2023-12-19 at 12.19.49 AM.png]]
![[Screenshot 2023-12-19 at 12.21.49 AM.png]]