import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def plot_losses(*objs):
    """
    Plot the training losses of multiple trained models.

    Parameters:
    - *objs: Variable number of trained models.

    Returns:
    - numpy.ndarray: The loss values for each model.

    Raises:
    - ValueError: If an untrained model is provided.
    - ValueError: If only one object is provided. More than one is required.
    """
    for obj in objs:
        if obj.is_trained == False:
            raise ValueError('Model is untrained, only input trained models')
    if len(list(objs)) > 1:
        loss = []
        names = []
        for obj in objs:
            loss.append(obj.lossList)
            names.append(obj.typeStr)
        
        loss = np.array(loss)
        
        for i in loss:
            plt.plot(i)
            
        plt.xlabel('Epoch')
        plt.ylabel('Loss [MSE]')
        plt.legend(names)
        plt.title('Mean Squared Error for NN, RNN and CNN with 10 epochs')
        
        plt.show()
    else:
        raise ValueError('Only one object provided. More than one required.')
        
    return loss

def animate_double_pendulum(initials, thetas, title='A random, unspecified run', save=None):
    """
    Animate the motion of a double pendulum.

    Parameters:
    - initials (numpy.ndarray): Initial conditions for the double pendulum.
    - thetas (numpy.ndarray): Angular positions of the pendulum over time.
    - title (str, optional): Title of the animation. Default is 'A random, unspecified run'.
    - save (bool, optional): File path to save the animation as a GIF. Default is None.
    """
    x = np.random.randint(len(thetas))
    theta1_vals = thetas[x,:,0]
    theta2_vals = thetas[x,:,1]
    L1 = initials[x, 0]
    L2 = initials[x, 1]

    x1 = L1 * np.sin(theta1_vals)
    y1 = -L1 * np.cos(theta1_vals)
    x2 = x1 + L2 * np.sin(theta2_vals)
    y2 = y1 - L2 * np.cos(theta2_vals)

    fig = plt.figure()
    ax = fig.add_subplot(111, autoscale_on=False, xlim=(-2, 2), ylim=(-2, 2), title=title)
    ax.set_aspect('equal')
    ax.grid()

    line, = ax.plot([], [], 'o-', lw=2, color='red')
    trail, = ax.plot([], [], '-', lw=1, color='gray')

    def init():
        line.set_data([], [])
        trail.set_data([], [])
        return line, trail

    def animate(i):
        x_vals = [0, x1[i], x2[i]]
        y_vals = [0, y1[i], y2[i]]
        line.set_data(x_vals, y_vals)

        trail_length = 100
        if i >= trail_length:
            trail_x = x2[i - trail_length:i]
            trail_y = y2[i - trail_length:i]
        else:
            trail_x = np.concatenate((x2[:i], x2[:i][::-1]))
            trail_y = np.concatenate((y2[:i], y2[:i][::-1]))

        trail.set_data(trail_x, trail_y)

        return line, trail
    
    keyframes = np.arange(0, len(theta1_vals), 3)

    ani = animation.FuncAnimation(fig, animate, frames=keyframes,
                                  interval=22, blit=True, init_func=init)
    
    if 'CNN' in title:
        saveStr = 'CNN'
    elif 'RNN' in title:
        saveStr = 'RNN'
    elif 'NN' in title:
        saveStr = 'NN'
    elif 'RK4' in title:
        saveStr = 'RK4'
    else:
        saveStr = '0'
    
    
    if save:
        ani.save('../figures/randomly_sampled_run_%s.gif' %saveStr, writer='pillow')

    plt.show()
    
def plot_traced_path(initials, thetas, title, heat=False):
    """
   Plot the traced path of a double pendulum.

   Parameters:
   - initials (numpy.ndarray): Initial conditions for the double pendulum.
   - thetas (numpy.ndarray): Angular positions of the pendulum over time.
   - title (str): Title of the plot.
   - heat (bool, optional): Whether to create a heatmap plot. Default is False.
   """
    L1 = initials[:, 0]
    L2 = initials[:, 1]
    plt.figure()
    if heat:
        all_x = np.concatenate([L1[i] * np.sin(thetas[i, :, 0]) + L2[i] * np.sin(thetas[i, :, 1]) for i in range(thetas.shape[0])])
        all_y = np.concatenate([-L1[i] * np.cos(thetas[i, :, 0]) - L2[i] * np.cos(thetas[i, :, 1]) for i in range(thetas.shape[0])])

        plt.hist2d(all_x, all_y, bins=100, cmap='hot', alpha=0.8)
        plt.colorbar(label='Density')
    
    else:
        if thetas.shape[1] > 50:
            thetas = thetas[:50, :, :]
            
        for i in range(thetas.shape[0]):
            theta1 = thetas[i, :, 0]
            theta2 = thetas[i, :, 1]
            x = L1[i] * np.sin(theta1) + L2[i] * np.sin(theta2)
            y = -L1[i] * np.cos(theta1) - L2[i] * np.cos(theta2)    
            plt.plot(x, y, color='grey', alpha=0.4)
        
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(title)
    plt.grid(True)
    plt.show()
