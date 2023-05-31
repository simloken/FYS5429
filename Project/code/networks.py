from data import read_file, strip_time
import numpy as np
import tensorflow as tf
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV
import joblib
import os

class Tensorflow:
    """
    A class for creating and training TensorFlow neural network models for solving double pendulum problems.

    Parameters:
    NNType (str): The type of neural network to be used ('NN', 'CNN', 'RNN'). Defaults to NN or None.
    initialsFile (str): The file path to the initial conditions data. Defaults to '../data/initials.txt'.
    solutionFile (str): The file path to the double pendulum solution data. Defaults to '../data/double_pendulum.txt'.
    ratio (float): The ratio of data to be used for training. Defaults to 0.8.
    replace (bool): Whether to replace an existing cached model. Defaults to False.
    verbose (bool): Whether to print progress and information during training. Defaults to True.
    alwaysSave (bool): Whether to save the trained model regardless of the NNType. Defaults to False.

    Attributes:
    initial_scaler (sklearn.preprocessing.MinMaxScaler): A scaler object for normalizing the initial conditions data.
    result_scaler (sklearn.preprocessing.MinMaxScaler): A scaler object for normalizing the solution data.
    lossList (list): A list to store the training loss values.
    is_trained (bool): Indicates if the model has been trained.
    t (tf.Variable): A TensorFlow variable used for the custom loss function. UNUSED
    model (tf.keras.Model): The TensorFlow neural network model.

    Methods:
    Model(): Define and compile the neural network model.
    splitData(ratio): Splits the data into training and testing sets based on the specified ratio.
    train(epochs, batch_size): Train the neural network model.
    predict(verbose): Make predictions using the trained model.
    
    Unused methods:
    optimizeModel(): Optimize the neural network model using grid search.
    physical_constraints_loss(): Compute the physical constraints loss for the double pendulum system.
    find_initial_energy(): Compute the initial energy of the double pendulum system.
    
    """
    def __init__(self,
                 NNType=None,
                 initialsFile='../data/initials.txt',
                 solutionFile='../data/double_pendulum.txt',
                 ratio=0.8,
                 replace=False,
                 verbose=True,
                 alwaysSave=False):
        """
        Initialize the Tensorflow class with the specified parameters.
        
        Parameters:
        NNType (str): The type of neural network to be used ('NN', 'CNN', 'RNN'). Defaults to NN or None.
        initialsFile (str): The file path to the initial conditions data. Defaults to '../data/initials.txt'.
        solutionFile (str): The file path to the double pendulum solution data. Defaults to '../data/double_pendulum.txt'.
        ratio (float): The ratio of data to be used for training. Defaults to 0.8.
        replace (bool): Whether to replace an existing cached model. Defaults to False.
        verbose (bool): Whether to print progress and information during training. Defaults to True.
        alwaysSave (bool): Whether to save the trained model regardless of the NNType. Defaults to False.
        """

        self.NNType = NNType
        self.initialsFile = initialsFile
        self.solutionFile = solutionFile
        self.ratio = ratio
        self.replace = replace
        self.verbose = verbose
        self.initial_scaler = MinMaxScaler()
        self.result_scaler = MinMaxScaler()
        self.lossList = []
        self.is_trained = False
        self.t = tf.Variable(0.0, trainable=False)
        self.alwaysSave = alwaysSave
        
        self.splitData(self.ratio)
        self.model = self.Model()
        
        
        
        if replace==False:
            if os.path.exists("../cached/%s_model.h5" %(self.typeStr)):
                self.model = tf.keras.models.load_model("../cached/%s_model.h5" %(self.typeStr))
                self.initial_scaler = joblib.load('../cached/%s_scalerx1.pk1' %(self.typeStr))
                self.result_scaler = joblib.load('../cached/%s_scalerx2.pk1' %(self.typeStr))
                file_path = "../cached/%s_losses.txt" %(self.typeStr)
                with open(file_path, 'r') as file:
                    line = file.readline()
                    losses = line.split()
                    self.lossList = [float(loss) for loss in losses]
                
                self.is_trained = True
                if self.verbose:
                    print("Existing %s model found! Loading model\nIf you want a new model, create a Tensorflow object with the boolean 'replace' set to True\n" %(self.typeStr))
        
    def Model(self):
        """
       Define and compile the neural network model.
    
       Parameters:
       None
    
       Returns:
       model (tf.keras.Model): The compiled neural network model.
    
       Note:
       This method constructs a neural network model based on the specified neural network type (NNType) and returns the compiled model.
       """
        input1 = tf.keras.layers.Input(shape=(8,))
        input2 = tf.keras.layers.Input(shape=(1000, 2))
        
        if self.NNType == 'RNN':
            self.typeStr = 'RNN'
            rnn_output = tf.keras.layers.LSTM(units=64, return_sequences=True)(input2)
            rnn_output = tf.keras.layers.Flatten()(rnn_output)
            rnn_output = tf.keras.layers.Dropout(0.3)(rnn_output)
            concat_output = tf.keras.layers.concatenate([input1, rnn_output])
            optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)

        elif self.NNType == 'CNN':
            self.typeStr = 'CNN'
            reshape_input2 = tf.keras.layers.Reshape((1000, 2, 1))(input2)
            conv_output = tf.keras.layers.Conv2D(filters=64, kernel_size=(2, 1), activation='relu')(reshape_input2)
            flatten_output = tf.keras.layers.Flatten()(conv_output)
            flatten_output = tf.keras.layers.Dropout(0.3)(flatten_output)
            concat_output = tf.keras.layers.concatenate([input1, flatten_output])
            optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)
        else:
            self.typeStr = 'NN'
            flatten_input = tf.keras.layers.Flatten()(input2)
            flatten_input = tf.keras.layers.Dropout(0.3)(flatten_input)
            concat_output = tf.keras.layers.concatenate([input1, flatten_input])
            optimizer = 'adam'
        
        output = tf.keras.layers.Dense(units=1000*2, kernel_regularizer=tf.keras.regularizers.l2(0.0005))(concat_output)  # Adjusted output shape
        
        model = tf.keras.Model(inputs=[input1, input2], outputs=output)
                   
        
        model.compile(optimizer=optimizer, loss='mse', loss_weights=[1.0, 1.0])
        
        return model
    
    def optimizeModel(self): #unimplemented
        """
       Optimize the neural network model using grid search.
    
       Parameters:
       None
    
       Note:
       This method performs a grid search over different hyperparameter combinations to find the best model configuration based on cross-validated performance.
       UNUSED
       """
        param_grid = {
            'optimizer': ['adam', 'sgd'],
            'activation': ['relu', 'sigmoid'],
            'hidden_units': [64, 128, 256],
            'regularization': [None, tf.keras.regularizers.l2(0.01)],
            'learning_rate': [0.001, 0.01, 0.1]
        }
    
        model = KerasClassifier(build_fn=self.Model, verbose=0)
    
        grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=3)
        
        x1 = self.initial_scaler.fit_transform(self.initial_train)
        x2 = self.result_scaler.fit_transform(self.result_train.reshape(-1, 2)).reshape(self.result_train.shape)
        x2_reshaped = x2.reshape(-1, 1000 * 2)

        grid_result = grid.fit([x1, x2], x2_reshaped)
    
        print("Best score: %f" % grid_result.best_score_)
        print("Best parameters: ", grid_result.best_params_)
    
        best_model = grid_result.best_estimator_.model
    
        best_model.fit([x1, x2], x2_reshaped, epochs=10, batch_size=32)
        
        self.model = best_model
    
    def splitData(self, ratio):
        """
        Split the data into training and testing sets based on the specified ratio.

        Parameters:
        ratio (float): The ratio of data to be used for training.

        Note:
        This method reads the initial conditions and solution data files, splits them into training and testing sets, and assigns them to the respective attributes of the class.
        """
        initial = read_file(self.initialsFile)
        result = strip_time(read_file(self.solutionFile, collapse=True))
        
        train_size = int(len(initial) * ratio)
        initial_train, initial_test = initial[:train_size], initial[train_size:]
        result_train, result_test = result[:train_size], result[train_size:]
        self.initial_train = initial_train
        self.initial_test = initial_test
        self.result_train = result_train
        self.result_test = result_test
    
    def train(self, epochs=10, batch_size=32):
        """
        Train the neural network model.
    
        Parameters:
        - epochs (int): Number of training epochs. Default is 10.
        - batch_size (int): Batch size for training. Default is 32.
    
        Note:
        This method fits the model using the provided training data and saves the trained model, scalers, and loss history.
        """
        if self.is_trained:
            if self.verbose:
                print('Model is already trained thus the .train() method is unnecessary.\nSkipping training\n')
            return
        
        x1 = self.initial_scaler.fit_transform(self.initial_train)
        x2 = self.result_scaler.fit_transform(self.result_train.reshape(-1, 2)).reshape(self.result_train.shape)
        
        loss = tf.keras.callbacks.LambdaCallback(on_epoch_end=lambda epoch,
                                                          logs: self.lossList.append(logs['loss']))
        
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss',
                                                        patience=2,
                                                        restore_best_weights=True)
        
        self.model.fit([x1, x2], x2.reshape(-1, 1000*2), epochs=epochs, batch_size=batch_size, callbacks=[loss, early_stopping])
        
        if not self.alwaysSave:
            if self.typeStr == 'CNN':
                confirm = str(input("Warning!\nConvolutional Neural Networks tend to be quite big, and in this case approaches 3Gb.\n\nDo you still wish to cache this model? [y/n]\n"))
                if confirm.lower() == 'n':
                    self.is_trained = True
                    return
            elif self.typeStr == 'RNN':
                confirm = str(input("Warning!\nRecurrent Neural Networks tend to be quite big, and in this case approaches 1.5Gb.\n\nDo you still wish to cache this model? [y/n]\n"))
                if confirm.lower() == 'n':
                    self.is_trained = True
                    return
            
        self.model.save('../cached/%s_model.h5'%(self.typeStr))
        joblib.dump(self.initial_scaler, '../cached/%s_scalerx1.pk1' %(self.typeStr))
        joblib.dump(self.result_scaler, '../cached/%s_scalerx2.pk1' %(self.typeStr))
        
        file_path = "../cached/%s_losses.txt" %(self.typeStr)
        with open(file_path, 'w') as file:
            line = ' '.join(str(loss) for loss in self.lossList)
            file.write(line)
        
        
        self.is_trained = True
        

    def predict(self, verbose=True):
        """
        Generate predictions using the trained model.
    
        Parameters:
        - verbose (bool): Whether to print the Mean Squared Error (MSE) or not. Default is True.
    
        Returns:
        - np.ndarray: Predicted values.
    
        Note:
        This method uses the trained model to generate predictions based on the provided test data.
        """
        if self.verbose != True:
            verbose=False
        x1 = self.initial_scaler.transform(self.initial_test)
        x2 = self.result_scaler.transform(self.result_test.reshape(-1, 2)).reshape(self.result_test.shape)
        
        predictions = self.model.predict([x1, x2])
        predictions = predictions.reshape(-1, 1000, 2)
        
        predictions = self.result_scaler.inverse_transform(predictions.reshape(-1, 2)).reshape(predictions.shape)
        
        mse = np.mean(np.square(predictions - x2))
        if verbose:
            print("Mean Squared Error (MSE):", mse)
        
        return predictions
    
    def physical_constraints_loss(self, y_true, y_pred): #unimplemented
        """
        Compute the physical constraints loss for the double pendulum system.
    
        Parameters:
        - y_true (tf.Tensor): True values of the double pendulum angles.
        - y_pred (tf.Tensor): Predicted values of the double pendulum angles.
    
        Returns:
        - tf.Tensor: The computed physical constraints loss.
    
        Note:
        This method is intended to be used as a loss function in a Tensorflow model
        UNUSED
        """
        
        theta1_pred, theta2_pred = y_pred[:, 0], y_pred[:, 1]
        
        batch_indices = tf.range(tf.shape(theta1_pred)[0])
        M1 = tf.cast(tf.gather(self.initial_train[:, 2], batch_indices), dtype=tf.float32)
        M2 = tf.cast(tf.gather(self.initial_train[:, 3], batch_indices), dtype=tf.float32)
        L1 = tf.cast(tf.gather(self.initial_train[:, 0], batch_indices), dtype=tf.float32)
        L2 = tf.cast(tf.gather(self.initial_train[:, 1], batch_indices), dtype=tf.float32)
            
        
        with tf.GradientTape() as tape:
            tape.watch([theta1_pred, theta2_pred])
            angles_pred = tf.stack([theta1_pred, theta2_pred], axis=1)
            time_derivative = tape.gradient(angles_pred, [theta1_pred, theta2_pred])
        omega1_pred, omega2_pred = time_derivative


        kinetic_energy = 0.5 * M1 * L1**2 * omega1_pred**2 + \
                        0.5 * M2 * (L1**2 * omega1_pred**2 +
                                    L2**2 * omega2_pred**2 +
                                    2 * L1 * L2 * omega1_pred * omega2_pred * tf.cos(theta1_pred - theta2_pred))
                        
                        

        g = -9.81
        potential_energy = M1 * L1 * g * (1 - tf.cos(theta1_pred)) + \
                           M2 * g * ((L1 * (1 - tf.cos(theta1_pred))) +
                                                         (L2 * (1 - tf.cos(theta2_pred))))
        
        total_energy = kinetic_energy + potential_energy
        
        target_energy = self.find_initial_energy(M1, M2, L1, L2, y_true)
        constraints_loss = tf.reduce_mean(tf.square(total_energy - 0.001*target_energy))
        
        return constraints_loss
    
    def find_initial_energy(self, M1, M2, L1, L2, y_true):
        """
        Compute the initial energy of the double pendulum system.
    
        Parameters:
        - M1 (tf.Tensor): Mass of the first pendulum.
        - M2 (tf.Tensor): Mass of the second pendulum.
        - L1 (tf.Tensor): Length of the first pendulum.
        - L2 (tf.Tensor): Length of the second pendulum.
        - y_true (tf.Tensor): True values of the double pendulum angles.
    
        Returns:
        - tf.Tensor: The computed initial energy of the double pendulum system.
        
        Note:
        This method is intended to be used as part of the custom loss function.
        UNUSED
        """

        theta1 = y_true[0, 0]
        theta2 = y_true[0, 1]
        
        kinetic_energy1 = 0.5 * M1 * L1**2 * tf.square(theta1)
        potential_energy1 = -M1 * -9.81 * L1 * tf.cos(theta1)
        kinetic_energy2 = 0.5 * M2 * (L1**2 * tf.square(theta1) + L2**2 * tf.square(theta2) + 2 * L1 * L2 * theta1 * theta2 * tf.cos(theta1 - theta2))
        potential_energy2 = -M2 * -9.81 * (L1 * tf.cos(theta1) + L2 * tf.cos(theta2))
    
        initial_energy = tf.reduce_sum(kinetic_energy1 + potential_energy1 + kinetic_energy2 + potential_energy2)
    
        return initial_energy