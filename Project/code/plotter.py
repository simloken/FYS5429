import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from data import strip_time, read_file


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

def animate_double_pendulum(initials, thetas, title='A random, unspecified run', save=None, fname=None, denoise=False):
    """
    Animate the motion of a double pendulum.

    Parameters:
    - initials (numpy.ndarray): Initial conditions for the double pendulum.
    - thetas (numpy.ndarray): Angular positions of the pendulum over time.
    - title (str, optional): Title of the animation. Default is 'A random, unspecified run'.
    - save (bool, optional): File path to save the animation as a GIF. Default is None.
    - fname (str, optional): Define a filename
    - denoise (bool, optional): Flag to enable denoising. Default is False.
    """
    x = np.random.randint(len(thetas))
    theta1_vals = thetas[x, :, 0]
    theta2_vals = thetas[x, :, 1]
    L1 = initials[x, 0]
    L2 = initials[x, 1]

    x1 = L1 * np.sin(theta1_vals)
    y1 = -L1 * np.cos(theta1_vals)
    x2 = x1 + L2 * np.sin(theta2_vals)
    y2 = y1 - L2 * np.cos(theta2_vals)

    if denoise:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))
        fig.suptitle(title)
        ax1.set_aspect('equal')
        ax1.set_xlim(-2, 2)
        ax1.set_ylim(-2, 2)
        ax1.grid()
        ax1.set_title('Original Motion')    
        line1, = ax1.plot([], [], 'o-', lw=2, color='red')
        trail1, = ax1.plot([], [], '-', lw=1, color='gray')
        ax2.set_aspect('equal')
        ax2.set_xlim(-2, 2)
        ax2.set_ylim(-2, 2)
        ax2.grid()
        ax2.set_title('Denoised Motion')
        line2, = ax2.plot([], [], 'o-', lw=2, color='blue')
        trail2, = ax2.plot([], [], '-', lw=1, color='gray')
    else:
        fig = plt.figure()
        ax = fig.add_subplot(111, autoscale_on=False, xlim=(-2, 2), ylim=(-2, 2), title=title)
        ax.set_aspect('equal')
        ax.grid()
        line1, = ax.plot([], [], 'o-', lw=2, color='red')
        trail1, = ax.plot([], [], '-', lw=1, color='gray')
        
        

    def init():
        line1.set_data([], [])
        trail1.set_data([], [])
        if denoise:
            line2.set_data([], [])
            trail2.set_data([], [])
            return line1, trail1, line2, trail2
        return line1, trail1

    def animate(i):
        x_vals1 = [0, x1[i], x2[i]]
        y_vals1 = [0, y1[i], y2[i]]
        line1.set_data(x_vals1, y_vals1)

        if denoise:
            smoothed_x2 = np.convolve(x2, np.ones(10) / 10, mode='same')
            smoothed_y2 = np.convolve(y2, np.ones(10) / 10, mode='same')

            smoothed_x1 = x1 + smoothed_x2 - x2
            smoothed_y1 = y1 + smoothed_y2 - y2

            line2.set_data([0, smoothed_x1[i], smoothed_x2[i]], [0, smoothed_y1[i], smoothed_y2[i]])

            trail_length = 100
            if i >= trail_length:
                trail_x2 = smoothed_x2[i - trail_length:i]
                trail_y2 = smoothed_y2[i - trail_length:i]
                trail_x1 = x2[i - trail_length:i]
                trail_y1 = y2[i - trail_length:i]
            else:
                trail_x2 = np.concatenate((smoothed_x2[:i], smoothed_x2[:i][::-1]))
                trail_y2 = np.concatenate((smoothed_y2[:i], smoothed_y2[:i][::-1]))
                trail_x1 = np.concatenate((x2[:i], x2[:i][::-1]))
                trail_y1 = np.concatenate((y2[:i], y2[:i][::-1]))

            trail2.set_data(trail_x2, trail_y2)
            trail1.set_data(trail_x1, trail_y1)
            
            return line1, trail1, line2, trail2
        
        else:
            trail_length = 100
            if i >= trail_length:
                trail_x2 = x2[i - trail_length:i]
                trail_y2 = y2[i - trail_length:i]
            else:
                trail_x2 = np.concatenate((x2[:i], x2[:i][::-1]))
                trail_y2 = np.concatenate((y2[:i], y2[:i][::-1]))

            trail1.set_data(trail_x2, trail_y2)
        

            return line1, trail1

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
    if save:
        if fname:
                ani.save('../figures/%s.gif' % fname, writer='pillow')
        elif not denoise:
            ani.save('../figures/updated_randomly_sampled_run_%s.gif' % saveStr, writer='pillow')
        else:
            ani.save('../figures/updated_randomly_sampled_run_%s_with_denoise.gif' % saveStr, writer='pillow')

    plt.show()

    
def plot_traced_path(initials, thetas, title, heat=False, retPlot=False):
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
    if not retPlot:
        plt.figure()
    if heat:
        all_x = np.concatenate([L1[i] * np.sin(thetas[i, :, 0]) + L2[i] * np.sin(thetas[i, :, 1]) for i in range(thetas.shape[0])])
        all_y = np.concatenate([-L1[i] * np.cos(thetas[i, :, 0]) - L2[i] * np.cos(thetas[i, :, 1]) for i in range(thetas.shape[0])])
        if retPlot:
            return all_x, all_y

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
    
def overlay_heatmaps(obj):
    """
   Overlay heatmaps generated from RK4 and predicted values.

   Parameters:
   - obj (object): A trained NN object
   """
    initials_data = read_file('../data/initials.txt')
    double_pendulum_data = strip_time(read_file('../data/double_pendulum.txt', collapse=True))
    rkx, rky = plot_traced_path(initials_data, double_pendulum_data, 'None', True, True)
    initials_data = obj.initial_test
    double_pendulum_data = obj.predict(verbose=False)
    nnx, nny = plot_traced_path(initials_data, double_pendulum_data, 'None', True, True)
    nnL = len(nnx)
    x = np.random.randint(0, len(rkx)-nnL)

    rkx = rkx[x:x+nnL]
    rky = rky[x:x+nnL]
    rkheatmap, xedges, yedges = np.histogram2d(rkx,rky, bins=250)
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    nnheatmap, _, _ = np.histogram2d(nnx,nny, bins=250)
    diff = np.abs(rkheatmap-nnheatmap)
    
    plt.figure()
    plt.imshow(diff.T, extent=extent, origin='lower', cmap='hot')
    plt.colorbar(label='Overlap (lower is better)')
    plt.title('Comparing heatmaps between RK4 and %s predicted values\n' % obj.typeStr)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True)
    plt.show()


