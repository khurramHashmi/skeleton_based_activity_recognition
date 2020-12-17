# skeleton_based_activity_recognition
Skeleton based activity recognition using Transformers

Download the dataset from the following link:
https://cloud.dfki.de/owncloud/index.php/s/E8WMyo6bLa24btq

## Docker Build and Usage

Docker file usage and run the following 3 commands.

- docker build -t skel_image .

- docker run -it skel_image  (-it to run in an interactive container)
- docker run --runtime=nvidia -u $(id -u):$(id -g) -it -v $HOME:/home/ image_name ( to mount docker with your home directory)
- If this command does not work, you need to install nvidia-docker

## Running the network
  	cd src
	cd ucla_github_pytorch
	python main.py --datase_path



