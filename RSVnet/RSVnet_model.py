import copy
import json
import os
import random
import sys
import time

# Set tensorflow debugging level (Should be placed after 'import os')
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'

import h5py
import numpy as np
import open3d as o3d
import sklearn
import tensorflow as tf

from configuration import var_RSVnet
from pointcloud import fromNumpyArray
from rotational_subgroup_voting import rsv
from utilities import getScaledArray, getUnscaledArray
from visualizer import visualizer

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

# Configuration parameters
v = var_RSVnet

class RSVnet_model:
    """
    This class the RSVnet model used to regress the necessary scalar projection 
    and radial vector length values See:
        -   Rotational Subgroup Voting (RSV) and Pose Clustering forRobust 3D 
        Object Recognition 
        [http://openaccess.thecvf.com/content_ICCV_2017/papers/Buch_Rotational_Subgroup_Voting_ICCV_2017_paper.pdf]
    """
    def __init__(self, model_name = 'myRSVmodel'):
        # Initialize containers to save the model metrics
        self.test_metrics_accuracy = np.array([])
        self.test_metrics_loss = np.array([])
        self.training_metrics_loss = np.array([])
        self.training_metrics_accuracy = np.array([])
        self.model_name = model_name
        # Initialize the model
        self._build()
        self._compile()

    def _build(self):
        """
        Define the structure of the model.
        """ 

        '''
        self.model.add(tf.keras.layers.BatchNormalization())
        self.model.add(tf.keras.layers.Conv1D(256, 1, activation='relu'))
        self.model.add(tf.keras.layers.BatchNormalization())
        self.model.add(tf.keras.layers.Conv1D(128, 1, activation='relu'))
        self.model.add(tf.keras.layers.BatchNormalization())
        self.model.add(tf.keras.layers.Conv1D(128, 1, activation='relu'))
        self.model.add(tf.keras.layers.MaxPooling1D())   
        self.model.add(tf.keras.layers.Dense(units=512, 
                                    activation='relu'))
        self.model.add(tf.keras.layers.BatchNormalization())
        self.model.add(tf.keras.layers.Dense(units=256, 
                                    activation='relu'))
        self.model.add(tf.keras.layers.BatchNormalization())
        self.model.add(tf.keras.layers.Dense(units=128, 
                                    activation='relu'))
        self.model.add(tf.keras.layers.BatchNormalization())
        self.model.add(tf.keras.layers.Dense(units=64, 
                                    activation='relu'))
        self.model.add(tf.keras.layers.BatchNormalization())
        self.model.add(tf.keras.layers.Dense(units=2))  
        '''
        
        self.model = tf.keras.models.Sequential()
        self.model.add(tf.keras.layers.Dense(units=1024, 
                                             activation='relu', 
                                             input_shape=(v.NUM_OF_FEATURES,)))
        self.model.add(tf.keras.layers.BatchNormalization())
        self.model.add(tf.keras.layers.Dense(units=256, 
                                             activation='relu'))
        self.model.add(tf.keras.layers.BatchNormalization())
        self.model.add(tf.keras.layers.Dense(units=128, 
                                             activation='relu'))
        self.model.add(tf.keras.layers.BatchNormalization())
        self.model.add(tf.keras.layers.Dense(units=32, 
                                             activation='relu'))
        self.model.add(tf.keras.layers.BatchNormalization())
        self.model.add(tf.keras.layers.Dense(units=2))  
        
        #self.model.summary()
    
    def _compile(self):
        """
        Compile the model.
        """  
        opt = tf.keras.optimizers.SGD(learning_rate=0.3, momentum=0.00, nesterov=False, name="SGD")
        #opt = 'SGD'
        self.model.compile(optimizer=opt, 
                           loss='mean_squared_error', 
                           metrics=['accuracy'])

    def fit(self, X_train, Y_train, num_of_epochs):
        """
        Train the model.
                
        Parameters
        ----------
        X_train: N x NUM_OF_FEATURES array
            Local feature vector data that will be used to train the model.
        Y_train: N x 2 array
            Values for the scalar projection ad radial vector length.
        num_of_epochs: int
            Number of epochs that the model will be trained
        """ 
        history = self.model.fit(x=X_train, 
                                 y=Y_train, 
                                 epochs=num_of_epochs)
        # Update the training metrics
        self.training_metrics_accuracy = np.append(self.training_metrics_accuracy, 
                                                   history.history['acc'][0])
        self.training_metrics_loss = np.append(self.training_metrics_loss,
                                               history.history['loss'][0])
                
    def evaluate(self, X_test, Y_test):
        """
        Evaluate the model. 
                
        Parameters
        ----------
        X_test: N x NUM_OF_FEATURES array
            Local feature vector data that will be used to evaluate the model.
        Y_test: N x 2 array
            Values for the scalar projection ad radial vector length.
        
        Returns
        ----------
        test_loss: float

        test_accuracy: float
        """       
        test_loss, test_accuracy = self.model.evaluate(X_test, Y_test)
        return test_loss, test_accuracy
    
    def evaluateBatch(self, base_path: str, file_list: list()):
        """
        Evaluate the model with a batch of files. 
                
        Parameters
        ----------
        base_path: str
            Path to the folder containing the files.
        file_list: list()
            List with the names of the files to be used for the evaluation.
        """   
        # Initialize container for the batch metrics
        batch_metrics = None
        # Loop through all teh evaluation files
        for f_test in file_list:
            # Open file and load the testing data
            file_path = os.path.join(base_path, f_test)
            [X_test, Y_test, _, _, _, _] = self.loadData(file_path)
            # Evaluate current data
            eval_metrics = self.evaluate(X_test, Y_test)
            # Save current data evaluation metrics
            if batch_metrics is None:
                batch_metrics = eval_metrics
            else:
                batch_metrics = np.row_stack([batch_metrics, eval_metrics])

        # Get the average value of the metrics for the batch
        batch_metrics_avg = np.mean(batch_metrics, axis=0)
        self.test_metrics_loss = np.append(self.test_metrics_loss,
                                           batch_metrics_avg[0])
        self.test_metrics_accuracy = np.append(self.test_metrics_accuracy,
                                               batch_metrics_avg[1])

    def predict(self, X_test):  
        """
        Predict the values of the scalar projection and radial vector length 
        for a sample. 
                
        Parameters
        ----------
        X_test: N x NUM_OF_FEATURES array
            Local feature vector data that will be used to test the model.

        Returns
        ----------
        scores: N x 2 array
            Prediction for the sample.
        """          
        scores = self.model.predict(X_test)
        return scores

    def loadData(self, file_path: str):
        """
        Load the input and output data used to train or test the model from an
        .h5 file. Each sample also contains its pointcloud data (points and 
        normals).

        Parameters
        ----------
        file_path: str
            Path to the .h5 file containing the model.

        Returns
        ----------
        X: (num_of_samples x NUM_OF_POINTS) x NUM_OF_FEATURES array
            LocalFeatureVectors for all the points in the file.
        Y: (num_of_samples x NUM_OF_POINTS) x 2 array
            RadialVectorLength and Scalarprojection for all the points in the
            file
        points: num_of_samples x NUM_OF_POINTS x NUM_OF_DIMENSIONS array
            Point data for each sample in the file.
        normals: num_of_samples x NUM_OF_POINTS x 3 array
            Normal data for each sample in the file.
        """ 
        # Open the file, load the data and reshape it in order to be usable by 
        # the model        
        with h5py.File(file_path, 'r') as f:
            data_f = f['GlobalFeatureVectors'][:]
            number_of_samples = data_f.__len__()
            data_f = np.reshape(data_f, (number_of_samples, v.NUM_OF_FEATURES))
            data_d = f['Projection'][:]
            #data_d = np.reshape(data_d, (number_of_samples, 1))
            data_l = f['RadialVectorLength'][:]
            #data_l = np.reshape(data_l, (number_of_samples, 1))
            points = f['Cloud'][:]
            normals = f['Normals'][:]
            seeds = f['Seed'][:]
            seeds_normals = f['Seed_Normal'][:]
        # Set the input (X) and output (Y) data
        X = data_f 
        Y = np.column_stack([data_d, data_l])
        # Scale the output data to be within the interval [0,1]
        Y , mins, maxs = getScaledArray(Y, high=1.0, low=0.0, mins=np.asarray([-103.75,0.07]), maxs=np.asarray([117.24,110.8]), bycolumn=True)
        self.updateNormalizationData(mins, maxs, high=1.0, low=0.0)
        return X, Y, points, normals, seeds, seeds_normals

    def loadModel(self, path_to_model: str):
        """
        Load the previous weights, metrics and normalization data for the model.

        Parameters
        ----------
        path_to_model: str
            Path to the folder containing the data for the model.
        """
        # All the files have the same base name which is given to the model at 
        # its declaration.
        base_filename = path_to_model + self.model_name 
        # Load model weights
        weights_path = base_filename + '_weights.h5'
        self.model.load_weights(weights_path)
        # Compile the model to apply the loaded weights
        self._compile()
        # Load model metrics
        metrics_path = base_filename + '_metrics.json'
        self.loadModelMetrics(metrics_path)
        # Load normalization data
        norm_val_path = base_filename + '_normalization_values.json'
        self.loadNormalizationData(norm_val_path)

    def saveModel(self, save_path: str):
        """
        Save the weights, metrics and normalization data for the model.

        Parameters
        ----------
        path_to_model: str
            Path to the folder in which to save the data of the model.
        """
        base_filename = save_path + self.model_name 
        # Save model weights
        weights_path = base_filename + '_weights.h5'  
        self.model.save_weights(weights_path)
        # Save model metrics
        metrics_path = base_filename + '_metrics.json'
        self.saveModelMetrics(metrics_path)
        # Save values used for normalization
        norm_val_path = base_filename + '_normalization_values.json'
        self.saveNormalizationData(norm_val_path)

    def loadModelMetrics(self, file_path: str):
        """
        Load the metrics data for the model.

        Parameters
        ----------
        file_path: str
            Path to the .json file containing the metrics data for the model.
        """
        print('INFO: Loading model metrics from: ' + file_path)
        with open(file_path, "r") as read_file:
            metrics_data = json.load(read_file)        
        # Update the metrics containers
        self.test_metrics_accuracy =  metrics_data['test']['accuracy']
        self.test_metrics_loss =  metrics_data['test']['loss']
        self.training_metrics_accuracy =  metrics_data['training']['accuracy']
        self.training_metrics_loss =  metrics_data['training']['loss']

    def saveModelMetrics(self, file_path:str):
        """
        Save the metrics data for the model.

        Parameters
        ----------
        file_path: str
            Path to the .json file in which to save the mertics data.
        """
        print('INFO: Saving model metrics to: ' + file_path)
        metrics = {
            "training":
                {
                    "accuracy": np.asarray(self.training_metrics_accuracy),
                    "loss": np.asarray(self.training_metrics_loss),
                },
                "test":
                {
                    "accuracy": np.asarray(self.test_metrics_accuracy),
                    "loss": np.asarray(self.test_metrics_loss),
                },
        }


        with open(file_path, "w") as write_file:
            json.dump(metrics, write_file,cls=NumpyEncoder, indent=4)

    def loadNormalizationData(self, file_path: str):  
        """
        Load the metrics data for the model.

        Parameters
        ----------
        file_path: str
            Path to the .json file containing the normalization data for the 
            model.
        """
        print('INFO: Loading normalization data from: ' + file_path)
        with open(file_path, "r") as read_file:
            normalization_data = json.load(read_file)
        # Load the old normalization values from the loaded model
        mins = [normalization_data['d']['min'], normalization_data['l']['min']]
        maxs = [normalization_data['d']['max'], normalization_data['l']['max']]
        low = normalization_data['range']['min']
        high = normalization_data['range']['max']
        # Update the normalization data
        self.updateNormalizationData(mins, maxs, low, high)

    def saveNormalizationData(self, file_path:str):
        """
        Save the normalization data of the model.

        Parameters
        ----------
        file_path: str
            Path to the .json file in which to save the normalization data.
        """
        print('INFO: Saving normalizing_values to: ' + file_path)
        normalization_data = {
            "d": 
                {
                    "min":self.mins[0], 
                    "max":self.maxs[0]
                },
            "l": 
                {
                    "min":self.mins[1], 
                    "max":self.maxs[1]
                },
            "range": 
                {
                    "min":self.low,
                    "max":self.high
                },
        }

        with open(file_path, "w") as write_file:
            json.dump(normalization_data, write_file, indent=4)

    def updateNormalizationData(self, mins, maxs, low, high):
        """
        Update the normalization values used in the model when un-scaling the
        predictions. If old values already exist, the average between old and
        new will be used to update.

        Parameters
        ----------
        min
            Lower bound of the unscaled range.
        max
            Upper bound of the unscaled range.
        high: float
            Upper bound of the scaled range.
        low: float
            Lower bound of the scaled range.
        """
        if hasattr(self, 'mins'):
            self.mins = (self.mins+mins)/2
        else:
            self.mins = mins
            
        if hasattr(self, 'maxs'):
            self.maxs = (self.maxs+maxs)/2
        else:
            self.maxs = maxs
        
        if hasattr(self, 'low'):
            self.low = (self.low+low)/2
        else:
            self.low = low           

        if hasattr(self, 'high'):
            self.high = (self.high+high)/2
        else:
            self.high = high
    
    def getVotes_i(self, points, normals, scores, t,  vis=False):
        # Unscale the predictions:
        scores_unscaled = getUnscaledArray(scaled_points=scores, 
                                           mins=np.asarray(self.mins), 
                                           maxs=np.asarray(self.maxs),
                                           high=1.0,
                                           low=0.0)
        sample = fromNumpyArray(points, normals)
        # Initialize the rotational subgroup voting module
        RSV = rsv(sample, 'scene')
        # Pass segment to RSV module and compute q, r_prima and t 
        q = RSV.computeRadialPoint(d=scores_unscaled[:,0])
        r_prima = RSV.computeRadialVectorPrima(l=scores_unscaled[:,1])
        r_prima_tesallation = RSV.computeTesallation(r_prima, t)
        votes = RSV.computeVotes(r_prima_tesallation, q)
        R = RSV.computeRotationFrameScene(r_prima_tesallation)

        if vis is True:
            RSV.visualizeVotes(votes)
            RSV.visualizeVote(votes, R, q, num_of_votes = 2)
            RSV.visualizeRadialVector(q, r_prima, num_of_samples=20)

        return votes, R, q

    def getDensity_i(self, points, normals, model, votes, R, s_t, s_R, vis=False):
        sample = fromNumpyArray(points=points, normals=normals)
        RSV = rsv(sample,'scene')
        density, votes_r, R_r = RSV.computeVoteDensity(votes, R, s_t, s_R)
        t_candidates, R_candidates = RSV.computeBestCandidates(votes_r, R_r, density, v.NUM_OF_CANDIDATES)
        if vis is True:
            # Get the number of votes
            [num_of_points, tesallation_level, _] = votes.shape
            number_of_votes = tesallation_level*num_of_points
            # Reshape the votes and rotational frame containers
            votes_r0 = np.reshape(votes,(number_of_votes, 3))
            RSV.visualizeDensity(votes_r0, density, t_candidates, R_candidates, model)
            RSV.visualizeDensity(votes_r, density, t_candidates, R_candidates, model)
        return density, t_candidates, R_candidates

    def getVotes(self, points, normals, scores, t: int, vis=False):
        # Unscale the predictions:
        scores_unscaled = getUnscaledArray(scaled_points=scores, 
                                           mins=np.asarray(self.mins), 
                                           maxs=np.asarray(self.maxs),
                                           high=1.0,
                                           low=0.0)
        #new_shape = (num_of_samples, v.NUM_OF_POINTS, 2)
        #scores_unscaled = np.reshape(scores_unscaled, new_shape)
        # Containers for the votes and q data
        test_q = list()
        test_votes = list()
        test_R = list()

        sample = fromNumpyArray(points[it], normals[it])
        # Initialize the rotational subgroup voting module
        RSV = rsv(sample,'scene')
        # Pass segment to RSV module and compute q, r_prima and t 
        q = RSV.computeRadialPoint(d=scores_unscaled[:,0])
        r_prima = RSV.computeRadialVectorPrima(l=scores_unscaled[:,1])
        r_prima_tesallation = RSV.computeTesallation(r_prima, t)
        votes = RSV.computeVotes(r_prima_tesallation, q)
        R = RSV.computeRotationFrame(r_prima_tesallation)
        # Add samples the test batch
        test_q.append(q)
        test_votes.append(votes)
        test_R.append(R)

        if vis is True:
            RSV.visualizeVotes(votes)
            RSV.visualizeVote(votes, R, q, num_of_votes = 1)
            RSV.visualizeRadialVector(q, r_prima, num_of_samples=20)

        return test_votes, test_R, test_q

    def getDensity(self, 
                   cloud_points, 
                   cloud_normals, 
                   model: o3d.geometry.PointCloud, 
                   votes, 
                   R, 
                   s_t, 
                   s_R, 
                   vis=False
                   ):
        num_of_samples = cloud_points.__len__()
        # Containers for the density data
        test_density = list()
        for it in range(num_of_samples):
            print('INFO: Get density: '+ str(it) + ' / ' + str(num_of_samples))
            sample = fromNumpyArray(cloud_points[it], cloud_normals[it])
            RSV = rsv(sample,'scene')
            density = RSV.computeVoteDensity(votes[it], R[it], s_t, s_R)
            test_density.append(density)
            if vis is True:
                RSV.visualizeDensity(votes[it], R[it], density, model, v.NUM_OF_CANDIDATES)

    def visualizeResults(self, points, normals, seeds, seeds_normals, scores, Y_real, num_of_points):
        num_of_samples = points.__len__()
        # Unscale the predictions:
        scores_unscaled = getUnscaledArray(scaled_points=scores, 
                                           mins=np.asarray(self.mins), 
                                           maxs=np.asarray(self.maxs),
                                           high=1.0,
                                           low=0.0)
        #new_shape = (num_of_samples, v.NUM_OF_POINTS, 2)
        #scores_unscaled = np.reshape(scores_unscaled, new_shape)
        Y_real_unscaled = getUnscaledArray(scaled_points=Y_real, 
                                    mins=np.asarray(self.mins), 
                                    maxs=np.asarray(self.maxs),
                                    high=1.0,
                                    low=0.0)
        #Y_real_unscaled = np.reshape(Y_real_unscaled, new_shape)



        for it in range(num_of_samples):
            score = scores_unscaled[it]
            score_real = Y_real_unscaled[it]
            
            sample_points = points[it]            
            sample_normals = normals[it]
            seed_point = seeds[it]
            seed_normal = seeds_normals[it]
            sample_points = [p + seed_point for p in sample_points]
            sample = fromNumpyArray(sample_points, sample_normals)
            rsv_predicted = rsv(sample)
            q = rsv_predicted.computeRadialPoint_i(seed_point, seed_normal, delta=score[0]) 
            r_prima = rsv_predicted.computeRadialVectorPrima_i(seed_normal, l=score[1]) 

            rsv_real = rsv(sample)
            q_real = rsv_real.computeRadialPoint_i(seed_point, seed_normal, delta=score_real[0]) 
            r_prima_real = rsv_real.computeRadialVectorPrima_i(seed_normal, l=score_real[1]) 

            vis = visualizer()
            vis.addPointcloud(sample)    
            vis.addSphere(center = seed_point, color=[0,0,0])
            vis.addSphere(center = q, color=[0,1,0])
            vis.addSphere(center = q_real, color=[0,0,1])
            vis.addLine(q, q_real)
            vis.addLine(seed_point, q_real, color=[1,0,0])
            vis.show()

            vis = visualizer()
            vis.addPointcloud(sample)
            vis.addSphere(center = seed_point, color=[0,0,0])
            vis.addSphere(center = q, color=[0,0,0])
            vis.addSphere(center = q + r_prima, color=[0,1,0])
            vis.addSphere(center = q + r_prima_real, color=[0,0,1])
            vis.addLine(q, q + r_prima_real)
            vis.addLine(q, q + r_prima)
            vis.addLine(seed_point, q, color=[1,0,0])
            vis.show()

    def visualizeBestCandidates(self, scene, object, t_candidates, R_candidates):
        vis = visualizer()
        vis.addPointcloud(scene)
        for i in range(len(t_candidates)):
            vis.addPointcloud(copy.deepcopy(object), t=t_candidates[i], R=R_candidates[i])
        vis.show()