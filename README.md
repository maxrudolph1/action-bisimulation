## Introduction (CODE IS DEPRECATED. New repo coming soon Summer 2025)
This repo contains the code to run the

## Install Instructions:

	# navigate to folder
	cd action-bisimulation/

	# create conda environment
	conda create -n actbisim python=3.9

	# install dependencies listed in requirements.txt
	pip install -r requirements.txt

	# install our code base in editale mode
	pip install -e .


<!-- - conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
- conda install -c conda-forge gym
- conda install -c conda-forge matplotlib
- conda install -c conda-forge tqdm
- conda install -c conda-forge h5py
- conda install -c anaconda pyyaml
- pip install -e .
- pip install tianshou
- cd nav2d_representation/nav2d_gymnasium
- pip install -e . -->

## Current state of what has been tried in this repo (compared to branched master and other backup dm_control branch)
- So the original issue was that the dataset was heavily imbalanced in terms of actions. The original 30% accuracy (even after stacking) was because the inverse dynamics was just learning to predict the same action for everything (southeast action).
- With this in mind I played around with adding class weights to the cross entropy loss. This didn't seem to do much though.
- I also tried doing a class-prior logit adjustment and played around with a parameter that controled how strong the correction for this was.
- No significant results from those (ended up being around 40% accuracy).
- I shifted from here and messed with the actual dataset itself.
- I created a script that grabbed the indices of a bunch of samples such that the resulting list of sample indices contained the same number of samples per action (this ended up beign somewhere around a dataset size of 300K). This effectively cretaed a new dataset that had completely balanced action labels.
- This didn't help too much either.
- Eventually I tried a "stall" threshold (as I've been calling it) and turned off the class prior adjustment from earlier.
	- The problem that I realized was happening was the there were a number of samples where there was not much change between the frames because of things such as dynamics in the environment or being against a wall (not super common).
	- I filtered these samples out by using a threshold on the displacement between t and t+1 in physical space.
- This got me to about 70% accuracy. I'm kinda out of ideas for this though, so I have been working on the point mass maze environment that was in your paper today.

