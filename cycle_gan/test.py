import dataset_loader

train_data = dataset_loader.custom_data_loader('/home/Desktop/dataset_skeleton/SGN_Data/train.h5',
                                                   '/home/Desktop/dataset_skeleton/SGN_Data/test.h5')

train_data.read_train()