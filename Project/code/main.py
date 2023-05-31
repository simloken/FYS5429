from networks import Tensorflow
from data import read_file, strip_time
from plotter import animate_double_pendulum, plot_losses, plot_traced_path

def plot_random_sample(obj=None, save=None):
    """
    Plot the animation of a random or predicted sample of a double pendulum.

    Parameters:
    - obj (Tensorflow, optional): Trained Tensorflow model. Default is None.
    - save (bool, optional): File path to save the animation as a GIF. Default is None.
    """
    if not obj:
        initials_data = read_file('../data/initials.txt')
        double_pendulum_data = strip_time(read_file('../data/double_pendulum.txt', collapse=True))
        title = 'Double Pendulum movement of a random RK4 generated sample'
        
    else:
        initials_data = obj.initial_test
        double_pendulum_data = obj.predict(verbose=False)
        modelType = obj.typeStr
        title = 'Double Pendulum movement of a random %s predicted sample' %(modelType)
        
        
    animate_double_pendulum(initials_data, double_pendulum_data, title, save=save)
        
    
def train_and_plot(name=None, replace=False):
    """
    Train a Tensorflow model and plot the animation of a random predicted sample.

    Parameters:
    - name (str, optional): Type of model to train. Options are 'RNN', 'CNN' or None. None is NN and default.
    - replace (bool, optional): Whether to replace an existing model with the same name. Default is False.
    """
    if name==None:
        nn = Tensorflow(replace=replace)
        nn.train()
        nn.predict()
    elif name.upper() in ['RNN', 'CNN']:
        if name.upper() == 'RNN':
            nn = Tensorflow('RNN', replace=replace)
            nn.train()
            nn.predict()
        else:
            nn = Tensorflow('CNN', replace=replace)
            nn.train()
            nn.predict()
    
    plot_random_sample(nn)
    
def plot_all_losses():
    """
    Train Tensorflow models (NN, RNN, CNN), predict samples, and plot their training losses.

    Returns:
    - numpy.ndarray: The loss values for each model.
    """
    nn = Tensorflow()
    nn.train()
    nn.predict()
    rnn = Tensorflow('RNN')
    rnn.train()
    rnn.predict()
    cnn = Tensorflow('CNN',)
    cnn.train()
    cnn.predict()
    loss = plot_losses(nn, rnn, cnn)
    return loss

def compare_movements(save=None):
    """
    Train Tensorflow models (NN, RNN, CNN), predict samples, and plot the animation of a random predicted sample for each model.

    Parameters:
    - save (bool, optional): File path to save the animations as GIFs. Default is None.

    """
    nn = Tensorflow()
    nn.train()
    rnn = Tensorflow('RNN')
    rnn.train()
    cnn = Tensorflow('CNN')
    cnn.train()
    
    plot_random_sample(save=save)
    for model in [nn, rnn, cnn]:
        plot_random_sample(model, save=save)
        

def paths_traced(obj=None, heat=False):
    """
    Plot the traced path of a random or predicted sample of a double pendulum.

    Parameters:
    - obj (Tensorflow, optional): Trained Tensorflow model. Default is None.
    - heat (bool, optional): Whether to create a heatmap plot. Default is False.
    """
    if not obj:
        initials_data = read_file('../data/initials.txt')
        double_pendulum_data = strip_time(read_file('../data/double_pendulum.txt', collapse=True))
        if heat:
            title = "Density of all RK4 samples"
        else:
            title = 'Path traced by 50 RK4 samples'
        
    else:
        initials_data = obj.initial_test
        double_pendulum_data = obj.predict(verbose=False)
        modelType = obj.typeStr
        if heat:
            title = "Density of all %s samples" %(modelType)
        else:
            title = 'Path traced by 50 %s samples' %(modelType)
    plot_traced_path(initials_data, double_pendulum_data, title=title, heat=heat)
    
def paths_traced_pass_obj(name=None, heat=False):
    """
    Train a Tensorflow model, predict samples, and plot the traced path of the predicted samples.

    Parameters:
    - name (str, optional): Type of model to train. Options are 'RNN', 'CNN' or None. None is NN and default.
    - heat (bool, optional): Whether to create a heatmap plot. Default is False.
    """
    if name==None:
        nn = Tensorflow()
        nn.train()
        nn.predict()
    elif name.upper() in ['RNN', 'CNN']:
        if name.upper() == 'RNN':
            nn = Tensorflow('RNN')
            nn.train()
            nn.predict()
        else:
            nn = Tensorflow('CNN')
            nn.train()
            nn.predict()
    
    paths_traced(nn, heat)
    
plot_all_losses()