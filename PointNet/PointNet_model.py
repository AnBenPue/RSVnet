import os

# Set tensorflow debugging level (Should be placed after 'import os')
#os.environ['TF_CPP_MIN_LOG_LEVEL']='3'

import h5py
import keras
import numpy as np
import tensorflow as tf
from keras import backend as K
from keras import optimizers
from keras.layers import (BatchNormalization, Convolution1D, Dense, Dropout,
                          Flatten, Input, Lambda, MaxPooling1D, Reshape)
from keras.models import Model
from keras.utils import np_utils
from sklearn.model_selection import train_test_split

from pointcloud import fromNumpyArray
from visualizer import visualizer

# Configuration parameters
# Visualization
BACKGROUND_COLOR = [1.0, 1.0, 0.0]
CANDIDATE_COLOR = [0.0, 1.0, 0.0]
CANDIDATE_RADIUS = 2
CENTER_COLOR = [0.0,0.0, 1.0]
CENTER_RADIUS = 9
CLOUD_COLOR = [0.8,0.8,0.8]

class pointnet_model:
    """
    This class implements a PointNet model:
        -   PointNet: Deep Learning on Point Sets for 3D Classification and 
        Segmentation. Created by Charles R. Qi, Hao Su, Kaichun Mo, 
        Leonidas J. Guibas from Stanford University.
            [https://github.com/charlesq34/pointnet]
        - Based on the implementation of garyli1019.
            [https://github.com/garyli1019/pointnet-keras]
    """
    def __init__(self, input_shape, num_of_categories):
        """
        Constructor of the class.
        
        Parameters
        ----------
        input_shape: 1 x 2 array
            Shape for the PointNet input.
        num_of_categories: int
            Number of output categories for the model.
        """  
        [self.num_of_points, self.num_of_dimensions] = input_shape 
        self.num_of_categories = num_of_categories 
        self._build() 
        self._compile()   

        self.metrics_loss = list()    
        self.metrics_acc = list()    

        self.metrics_loss_test = list()    
        self.metrics_acc_test = list()   

    def _build(self):
        """
        Define the structure of the model.
        """ 
        adam = optimizers.Adam(lr=0.001, decay=0.7)
        # ------------------------------------ Pointnet Architecture
        # input_Transformation_net
        input_shape = (self.num_of_points, self.num_of_dimensions) 
        input_points = Input(shape = input_shape)
        x = Convolution1D(filters = 64,
                          kernel_size = 1, 
                          activation ='relu', 
                          input_shape = input_shape)(input_points)
        x = BatchNormalization()(x)
        x = Convolution1D(filters = 128, 
                          kernel_size = 1, 
                          activation = 'relu')(x)
        x = BatchNormalization()(x)
        x = Convolution1D(filters = 1024,                           
                          kernel_size = 1, 
                          activation = 'relu')(x)
        x = BatchNormalization()(x)
        x = MaxPooling1D(pool_size = self.num_of_points)(x)
        x = Dense(units = 512, activation = 'relu')(x)
        x = BatchNormalization()(x)
        x = Dense(units = 256, activation = 'relu')(x)
        x = BatchNormalization()(x)
        x = Dense(units = 9, 
                  weights = [np.zeros([256, 9]), 
                             np.eye(3).flatten().astype(np.float32)])(x)
        input_T = Reshape((3, 3))(x)

        # forward net
        g = Lambda(self.mat_mul, arguments={'B': input_T})(input_points)
        g = Convolution1D(filters = 64, 
                          kernel_size = 1, 
                          input_shape = input_shape, 
                          activation = 'relu')(g)
        g = BatchNormalization()(g)
        g = Convolution1D(filters = 64, 
                          kernel_size = 1, 
                          input_shape = input_shape, 
                          activation = 'relu')(g)
        g = BatchNormalization()(g)
    
        # feature transform net
        f = Convolution1D(filters = 64, 
                          kernel_size = 1, 
                          activation = 'relu')(g)
        f = BatchNormalization()(f)
        f = Convolution1D(filters = 128, 
                          kernel_size = 1, 
                          activation = 'relu')(f)
        f = BatchNormalization()(f)
        f = Convolution1D(filters = 1024, 
                          kernel_size = 1, 
                          activation = 'relu')(f)
        f = BatchNormalization()(f)
        f = MaxPooling1D(pool_size = self.num_of_points)(f)
        f = Dense(units = 512, activation = 'relu')(f)
        f = BatchNormalization()(f)
        f = Dense(units = 256, activation = 'relu')(f)
        f = BatchNormalization()(f)
        f = Dense(units = 64 * 64, 
                  weights = [np.zeros([256, 64 * 64]),
                             np.eye(64).flatten().astype(np.float32)])(f)
        feature_T = Reshape((64, 64))(f)

        # forward net
        g = Lambda(self.mat_mul, arguments = {'B': feature_T})(g)
        g = Convolution1D(filters = 64, 
                          kernel_size = 1, 
                          activation = 'relu')(g)
        g = BatchNormalization()(g)
        g = Convolution1D(filters = 128,
                          kernel_size = 1, 
                          activation = 'relu')(g)
        g = BatchNormalization()(g)
        g = Convolution1D(filters = 1024,
                          kernel_size = 1, 
                          activation = 'relu')(g)
        g = BatchNormalization()(g)

        # global_feature
        global_feature = MaxPooling1D(pool_size = self.num_of_points)(g)

        # point_net_cls
        c = Dense(units = 512, activation = 'relu')(global_feature)
        c = BatchNormalization()(c)
        c = Dropout(rate = 0.7)(c)
        c = Dense(units = 256, activation = 'relu')(c)
        c = BatchNormalization()(c)
        c = Dropout(rate = 0.7)(c)
        c = Dense(units = self.num_of_categories, activation = 'softmax')(c)
        prediction = Flatten()(c)
        # --------------------------------------------------end of pointnet
        self.model = Model(inputs=input_points, outputs=prediction)
        #print(self.model.summary())
    
    def _compile(self):
        """
        Compile the model.
        """    
        self.model.compile(optimizer = 'adam', 
                           loss = 'categorical_crossentropy', 
                           metrics = ['accuracy'])

    def fit(self, X_train, Y_train, X_test, Y_test, num_of_epochs=1):
        """
        Train the model. First the X_train data is used to train the model for
        one epoch. After this, the input points are modified by applying a 
        rotation and jittering in order to increase the training data. After 
        the modifications, the model is trained for another epoch. This process 
        is repeated as many time as requested. Every fifth epoch the model will
        be evaluated using the test data.
                
        Parameters
        ----------
        X_train: N x num_of_points x num_of_dimensions array
            Point cloud data that will be used to train the model.
        Y_train: N x num_of_categories array
            Labels corresponding to which category the training sample belongs.
        X_test: N x num_of_points x num_of_dimensions array
            Point cloud data that will be used to evaluate the model.
        Y_test: N x  num_of_categories array
            Labels corresponding to which category the testing sample belongs.
        num_of_epochs: int
            Number of epochs that the model will be trained
        """   
        for i in range(num_of_epochs):
            metrics = self.model.fit(x = X_train, 
                           y = Y_train, 
                           batch_size = 32, 
                           epochs = 1, 
                           shuffle = True, 
                           verbose = 1)
            self.metrics_acc.append(metrics.history['accuracy'][0])
            self.metrics_loss.append(metrics.history['loss'][0])
            # rotate and jitter the points
            X_train_rotate = self.rotate_point_cloud(X_train)
            X_train_jitter = self.jitter_point_cloud(X_train_rotate)
            metrics = self.model.fit(x = X_train_jitter, 
                           y = Y_train, 
                           batch_size = 32, 
                           epochs = 1, 
                           shuffle = True, 
                           verbose = 1)
            self.metrics_acc.append(metrics.history['accuracy'][0])
            self.metrics_loss.append(metrics.history['loss'][0])

            if i % 1 == 0:
                score = self.evaluate(X_test, Y_test)
                self.metrics_acc_test.append(score[1])
                self.metrics_loss_test.append(score[0])

    def evaluate(self, X_test, Y_test):
        """
        Evaluate the model. 
                
        Parameters
        ----------
        X_test: N x num_of_points x num_of_dimensions array
            Point cloud data that will be used to evaluate the model.
        Y_test: N x num_of_categories array
            Labels corresponding to which category the testing sample belongs.
                
        Returns
        ----------
        scores: N x num_of_categories array
            Predictions for the evaluation data.
        """       
        scores = self.model.evaluate(X_test, Y_test, verbose=1)
        print('Test loss: ', scores[0])
        print('Test accuracy: ', scores[1])
        return scores

    def predict(self, X_test): 
        """
        Predict the category for a given set of point clouds. 
                
        Parameters
        ----------
        X_test: N x num_of_points x num_of_dimensions array
            Point cloud data that will be used to test the model.

        Returns
        ----------
        scores: N x num_of_categories array
            Predictions for the test data.
        """         
        scores = self.model.predict(X_test, verbose=1)
        return scores

    def predictLocalFeatureVectors(self, X_test):
        """
        Predict the local feature vector for a given set of point clouds. 
        This is, for each point in the cloud obtain the feature vector with 
        length NUM_OF_FEATURES.
                
        Parameters
        ----------
        X_test: N x num_of_points x NUM_OF_DIMENSION array
            Point cloud data that will be used to test the model.
        
        Returns
        ----------
        local_feature_vectors: N x num_of_points x NUM_OF_FEATURES array
            Feature vector for all the points in the set.
        """        
        # Build an intermediated model that outputs the local feature vector 
        # for each point
        layer_name = 'batch_normalization_15'
        sub_model = Model(inputs = self.model.input, 
                          outputs = self.model.get_layer(layer_name).output)
        local_feature_vectors = sub_model.predict(X_test, verbose=1)
        return local_feature_vectors

    def predictGlobalFeatureVectors(self, X_test):    
        layer_name = 'max_pooling1d_3'
        sub_model = Model(inputs = self.model.input, 
                          outputs = self.model.get_layer(layer_name).output)
        global_feature_vectors = sub_model.predict(X_test, verbose=1)
        return global_feature_vectors

    def saveModel(self, model_path: str):
        """
        Save the model weights into a .h5 file,
                
        Parameters
        ----------
        model_path: str
            Path where to save the model.
        """  
        print('Saving model to:' + model_path)
        self.model.save_weights(model_path)
        print(self.metrics_acc)
        print(self.metrics_loss)
        print(self.metrics_acc_test)
        print(self.metrics_loss_test)

    def loadModel(self, model_path: str):
        """
        Load the model weights from a .h5 file,
                
        Parameters
        ----------
        model_path: str
            Path where to save the model.
        """  
        print('INFO: Loading model from:' + model_path)
        self.model.load_weights(model_path)
        # Compile the model to apply the loaded weights
        self._compile()
    
    def mat_mul(self, A, B):
        """
        Multiplication of two matrices,
        """  
        return tf.matmul(A, B)

    def rotate_point_cloud(self, batch_data):
        """ 
        Randomly rotate the point clouds to augment the dataset rotation is
        per shape based along up direction

        Parameters
        ----------
        batch_data: B x N x 3 array
            Original batch of point clouds

        Returns
        ----------
        rotated_data: B x N x 3 array
            Rotated batch of point clouds
        """
        rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
        for k in range(batch_data.shape[0]):
            rotation_angle = np.random.uniform() * 2 * np.pi
            cosval = np.cos(rotation_angle)
            sinval = np.sin(rotation_angle)
            rotation_matrix = np.array([[cosval, 0, sinval],
                                        [0, 1, 0],
                                        [-sinval, 0, cosval]])
            shape_pc = batch_data[k, ...]
            rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)),
                                          rotation_matrix)
        return rotated_data

    def jitter_point_cloud(self, batch_data, sigma=0.01, clip=0.05):
        """ 
        Randomly rotate the point clouds to augment the dataset
        rotation is per shape based along up direction

        Parameters
        ----------
        batch_data: B x N x 3 array
            Original batch of point clouds

        Returns
        ----------
        jittered_data: B x N x 3 array
            Jittered batch of point clouds
        """
        B, N, C = batch_data.shape
        #assert(clip > 0)
        jittered_data = np.clip(sigma * np.random.randn(B, N, C), -1*clip, clip)
        jittered_data += batch_data
        return jittered_data
        
    def loadDataset(self, data_path: str):
        """
        Load the dataset in order to train and evaluate the model.
                
        Parameters
        ----------
        data_path: str
            Path where the dataset is saved.

        Returns
        ----------
        data_points: N x num_of_points x num_of_dimensions array
            Data for the samples.
        data_labels: N x num_of_points x 1 array
            Label for the samples.
        """ 
        # Get all the files in the file path
        filenames = [d for d in os.listdir(data_path)]
        # containers for the data
        data_points = None
        data_labels = None
        # Loop through all the data
        for d in filenames:
            # Open the current .h5 file
            cur_points, cur_labels = self.load_h5(os.path.join(data_path, d))
            if data_labels is None or data_points is None:
                data_labels = cur_labels
                data_points = cur_points
            else:
                data_labels = np.row_stack([data_labels, cur_labels])
                data_points = np.row_stack([data_points, cur_points])
        return data_points, data_labels

    def splitDataset(self, data_points, 
                           data_labels,
                           test_size: float):   
        """
        Split the data into the test and training test.
                
        Parameters
        ----------
        data_points: numN x num_of_points x num_of_dimensions array
            Data for the samples.
        data_labels: N x num_of_points x 1 array
            Label for the samples.
        test_size: float
            Percentage of data for the test set.

        Returns
        ----------
        X_train: N x num_of_points x num_of_dimensions array
            Point cloud data that will be used to train the model.
        y_train: N x  num_of_categories array
            Labels corresponding to which category the training sample belongs.
        X_test: N x num_of_points x num_of_dimensions array
            Point cloud data that will be used to evaluate the model.
        y_test: N x  num_of_categories array
            Labels corresponding to which category the testing sample belongs.   
        """  
        # Select only XYZ variables
        data_xyz = data_points[:, :, 0:self.num_of_dimensions]
        data_split = train_test_split(data_xyz, data_labels, test_size=test_size, random_state=42)
        # Unpack the splitted data
        [X_train, X_test, y_train, y_test] = data_split
        # label to categorical
        y_train = np_utils.to_categorical(y_train, self.num_of_categories)
        y_test = np_utils.to_categorical(y_test, self.num_of_categories)

        return X_train, y_train, X_test, y_test

    def load_h5(self, h5_filename):
        """
        Load and .h5 data containing the data samples.

        Parameters
        ----------
        h5_filename: str
            File path for the .h5 file.

        Returns
        ----------
        data: N x num_of_points x num_of_dimensions array
            Data for the sample.
        label: N x 1 array 
            Label for the sample.
        """
        f = h5py.File(h5_filename)
        data = f['data'][:]
        label = f['label'][:]
        return (data, label)
        
    def visualizeData(self, data_points, num_of_samples: int):
        """
        Visualization for the PointNet data.
            Visualize an specified number of samples.
                
        Parameters
        ----------            
        data_points: N x num_of_points x num_of_dimensions array
            Point data for the set.
        num_of_samples: int
            Number of samples to be visualized.    
        """ 
        # Create visualizer object.
        vis = visualizer()  
        for d in range(num_of_samples):  
            offset = d*2
            sample = data_points[d,:,0:num_of_dimensions]/100
            sample[:,0] = sample[:,0]+ offset
            vis.addPointcloud(fromNumpyArray(sample))
        vis.show() 

class scene_classifier(pointnet_model):

    def getPredictedClass(self, score):
        """
        Given the score for a sample, get the class to which it belongs.
                
        Parameters
        ----------
        score: float
            Score for the sample

        Returns
        ----------
        :str
            Predicted class for the sample
        """  
        if score[0] > score[1]:
            return "object"
        elif score[0] <= score[1]:
            return "background"
            
    def getCandidates(self, scores, points, num_of_candidates: int):
        """
        For a set of samples, get the ones that are more likely to be the 
        model.
                
        Parameters
        ----------
        scores: N x NUM_OF_CATEGORIES array
            Scores for the set.            
        points: N x num_of_points x num_of_dimensions array
            Point data for the set.
        num_of_candidates: int
            Number of candidates to be selected.

        Returns
        ----------
        c_indices: N x 1
            Indices of the selected samples.
        """  
        # Containers for the candidates data
        c_indices = np.array([], dtype=int)
        for i in range(num_of_candidates):
            # Select the first column of the scores, Since we are classifying 
            # between object and background, the score for a given segment will
            # be:
            # [p_of_segment_being_object, p_of_segment_being_background]
            score_object = scores[:,0]
            # Get the maximum score value, this is, the score of the segment 
            # which the model believes most likely to be the object.
            score_max = np.amax(score_object)
            # Get the index of the element containing the max value
            idx_max = np.where(score_object == score_max)
            # Get the points belonging to the candidate, we take the first 
            # element of idx_max just in case there are two or more scores 
            # with the maximum value
            points_i = points[idx_max[0]] 
            c_indices =  np.append(c_indices, idx_max[0])           
            # Delete the maximum value
            scores[idx_max[0]] = np.asarray([0.0,0.0])

        return c_indices
            
    def visualizeBestCandidates(self, points): 
        """
        Visualization for the selected candidates.
                
        Parameters
        ----------            
        points: N x num_of_points x num_of_dimensions array
            Point data for the candidates.
        """   
        # Create visualizer object.
        vis = visualizer()                  
        # Loop through all the candidates
        for c in points:            
            # Add cloud with the candidate points
            candidate = fromNumpyArray(c)            
            vis.addPointcloud(candidate, color=CANDIDATE_COLOR)
            # Add sphere to visualize the cloud center
            vis.addSphere(candidate.get_center(), CENTER_COLOR, CENTER_RADIUS)
        vis.show()

    def visualizePrediction(self, points, scores):
        """
        Visualization for the segments classified as object.
                
        Parameters
        ----------            
        points: N x num_of_points x num_of_dimensions array
            Point data for the set.
        scores: N x NUM_OF_CATEGORIES array
            Scores for the set.    
        """  
        # Create visualizer object.
        vis = visualizer()  
        # Loop through all the samples in order to assign the correct color 
        # depending on their score
        num_of_samples=points.__len__()

        for i in range(num_of_samples):
            # Get the predicted class for the segment
            predicted_class = self.getPredictedClass(scores[i])
            # Get the segment as a point cloud
            segment = fromNumpyArray(points[i])
            if predicted_class == "object":
                vis.addPointcloud(segment, color=CANDIDATE_COLOR)

        vis.show()

class object_classifier(pointnet_model):

    def getPredictedCategory(self, scores, min_confidence):
        #TODO: add description

        categories = [self.getPredictedCategory_i(s, min_confidence=min_confidence) for s in scores]    
        return categories

    def getPredictedCategory_i(self, s, min_confidence=0.05):
        #TODO: add description

        # Get the maximum score value, this is, among all the categories, 
        # the one with the highest probability which the segment belongs to
        score_max = np.amax(s)
        if score_max > min_confidence:
            # Get the index of the maximun value, 
            idx_max = np.where(s == score_max)
            # idx_max idx_max contains an array with all the indices where the 
            # condition was met. 
            category = idx_max[0]
            # Since it is possible that a tie between two categories exists, we 
            # take the first element
            return category[0]
        else:
            return -1

    def visualizePrediction(self, seeds, scene, scene_indices, predicted_categories, num_of_segments = 10):
        """
        Visualize the predicted categories for different points. 
                
        Parameters
        ----------
        seeds: 
        
        scene:

        scene_indices: 
            Indices of the points from the original scene point cloud which 
            were used to generate each segment when splitting the scene.

        predicted_categories:
        
        num_of_segments:

        """
        # Visualize the predicted results
        vis = visualizer()
        for s in np.asarray(seeds.points):
            vis.addSphere(s, color=[0,0,0], radius=2)
        # Add point clouds to the visualization
        vis.addPointcloud(cloud=seeds)
        vis.addPointcloud(cloud=scene)
        # Get the points of both point clouds as numpy array
        seeds_points = np.asarray(seeds.points)
        scene_points = np.asarray(scene.points)
        # Add lines to visualize the correspondences        
        for i in np.random.uniform(low=0, high=(len(scene_indices)-1), size=num_of_segments):
            p1 = scene_points[scene_indices[int(i)]]            
            c = predicted_categories[int(i)] 
            p2 = seeds_points[c]
            if c is not -1: 
                color = [0,0,1]                
            else:
                color = [1,0,0]
            vis.addSphere(center=p1, color=color, radius=2)
            vis.addSphere(center=p2, color=color, radius=2)
            vis.addLine(p1, p2, color=color)
        vis.show()