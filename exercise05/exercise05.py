import pickle
import numpy as np
import sys
from copy import deepcopy
from robo.fmin import bayesian_optimization


rf = pickle.load(open("./rf_surrogate_cnn.pkl", "rb"))
cost_rf = pickle.load(open("./rf_cost_surrogate_cnn.pkl", "rb"))
lower = np.array([-6, 32, 4, 4, 4])
upper = np.array([0, 512, 10, 10, 10])


def objective_function(x, epoch=40):
    """
        Function wrapper to approximate the validation error of the hyperparameter configurations x by the prediction of a surrogate regression model,
        which was trained on the validation error of randomly sampled hyperparameter configurations.
        The original surrogate predicts the validation error after a given epoch. Since all hyperparameter configurations were trained for a total amount of 
        40 epochs, we will query the performance after epoch 40.
    """
    
    # Normalize all hyperparameter to be in [0, 1]
    x_norm = deepcopy(x)
    x_norm[0] = (x[0] - (-6)) / (0 - (-6))
    x_norm[1] = (x[1] - 32) / (512 - 32)
    x_norm[2] = (x[2] - 4) / (10 - 4)
    x_norm[3] = (x[3] - 4) / (10 - 4)
    x_norm[4] = (x[4] - 4) / (10 - 4)
    

    x_norm = np.append(x_norm, epoch)
    y = rf.predict(x_norm[None, :])[0]

    return y

def runtime(x, epoch=40):
    """
        Function wrapper to approximate the runtime of the hyperparameter configurations x.
    """
    
    # Normalize all hyperparameter to be in [0, 1]
    x_norm = deepcopy(x)
    x_norm[0] = (x[0] - (-6)) / (0 - (-6))
    x_norm[1] = (x[1] - 32) / (512 - 32)
    x_norm[2] = (x[2] - 4) / (10 - 4)
    x_norm[3] = (x[3] - 4) / (10 - 4)
    x_norm[4] = (x[4] - 4) / (10 - 4)
    

    x_norm = np.append(x_norm, epoch)
    y = cost_rf.predict(x_norm[None, :])[0]

    return y

def random_search(objective_function, runtime, iterations=50):
    best_res = sys.maxsize
    cur_time = 0.0
    plot_error = []
    plot_time = []
    for i in range(iterations):
        x = [np.random.uniform(-6, 0),
             np.random.randint(32, 513),
             np.random.randint(4, 11),
             np.random.randint(4, 11),
             np.random.randint(4, 11)]
        cur_error = objective_function(x)
        if cur_error < best_res:
            best_res = cur_error
        plot_error.append(best_res)
        cur_time += runtime(x)
        plot_time.append(cur_time)
    return plot_error, plot_time

if __name__ == '__main__':
    r_all_errors = []
    r_all_time = []
    b_all_errors = []
    b_all_time = []
    for i in range(10):
        print(i)
        sys.stdout.flush()
        plot_error, plot_time = random_search(objective_function, runtime, iterations=50)
        r_all_errors.append(plot_error)
        r_all_time.append(plot_time)
        b_results = bayesian_optimization(objective_function, lower, upper, num_iterations=50)
        b_all_errors.append(b_results['incumbent_values'])
        cur_time = 0.0
        b_plot_time = []
        for x in b_results['X']:
            cur_time += runtime(x)
            b_plot_time.append(cur_time)
        b_all_time.append(b_plot_time)
    for e in np.mean(r_all_errors, axis=0):
        print('{:.6f}'.format(e))
    print('=' * 80)
    for t in np.mean(r_all_time, axis=0):
        print('{:.2f}'.format(t))
    print('=' * 80)
    for e in np.mean(b_all_errors, axis=0):
        print('{:.6f}'.format(e))
    print('=' * 80)
    for t in np.mean(b_all_time, axis=0):
        print('{:.2f}'.format(t))
