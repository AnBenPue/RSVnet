import os 

from configuration import file_paths, var_RSVnet
from RSVnet_model_global import rsvnet_global_model

# Configuration parameters
f = file_paths
v = var_RSVnet

# Load the path to the train and evaluation files
files_training = [f for f in os.listdir(f.RSVNET_DATA_TRAIN_G)]   
files_evaluating  = [f for f in os.listdir(f.RSVNET_DATA_EVAL_G)]   

# Run RSVnet
RSV_net = rsvnet_global_model(model_name='myRSVnet')
for epoch in range(v.NUM_OF_EPOCHS):
    print('Info: Current epoch: ' + str(epoch))
    for f_train in files_training:
        # Load the training data
        file_path = os.path.join(f.RSVNET_DATA_TRAIN_G, f_train)
        [X_train, Y_train, points, normals, seeds, seeds_normals] = RSV_net.loadData(file_path)
        # If a pretrained model exist, load it
        if len(os.listdir(f.RSVNET_PT_G)) != 0:
            RSV_net.loadModel(f.RSVNET_PT_G) 
        # Train and evaluate the model
        RSV_net.fit(X_train, Y_train, num_of_epochs=1)
        RSV_net.evaluateBatch(base_path = f.RSVNET_DATA_EVAL_G, 
                              file_list = files_evaluating)
        RSV_net.saveModel(f.RSVNET_PT_G)        