# skeleton_based_activity_recognition
Skeleton based activity recognition using Transformers

Download the dataset from the following link:
https://cloud.dfki.de/owncloud/index.php/s/E8WMyo6bLa24btq

## Docker Build and Usage

Docker file usage and run the following 3 commands.

	docker build -t skel_image .
	docker run -it skel_image  (-it to run in an interactive container)
 	docker run --runtime=nvidia -u $(id -u):$(id -g) -it -v $HOME:/home/ image_name ( to mount docker with your home directory)
- If this command does not work, you need to install nvidia-docker

## Start Training
  	cd src
	cd ucla_github_pytorch
	python main.py --help
- This will help you to see what are the required arguments and how you can give them into the training script.
## Command Template
```
python main.py
  --train_datapath <location of train dataset numpy file>
  --train_labelpath <location of train label pickle file>
  --val_datapath <location of Validation dataset numpy file>
  --val_labelpath <location of Validation label pickle file>
  [--lr <base learning rate>]
  [--batch-size <batch size>]
  [--transformed_data <needs to be set to True if transformed data is used instead of standard>]
  [--max_epochs <number of epochs for training>]
  [--num_class <number of classes for the dataset>]
```

- An example command for running the main.py is as follows :
### Example Command
	python main.py --train_datapath ./xsub/train_data_joint.npy --train_labelpath ./xsub/train_label.pkl --val_datapath ./xsub/val_data_joint.npy --val_labelpath ./xsub/val_label.pkl
	


