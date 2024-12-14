import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from scipy.optimize import minimize

def plot_hidden_fr_spk(time, hidden, fr, spk):
    fig = plt.figure(figsize=(12, 6))
    canvas = FigureCanvas(fig)
    plt.subplot(3,1,1)
    plt.plot(time, hidden.T)
    plt.title("Hidden dynamics")
    plt.subplot(3,1,2)
    plt.plot(time, fr.T)
    plt.title("Firing rate")
    plt.subplot(3,1,3)
    plt.plot(time, spk.T, '.', markersize=1)
    plt.title("Spike")
    fig.canvas.draw()
    graph = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
    w, h = fig.canvas.get_width_height()
    graph = graph.reshape((h, w, 3))
    return graph

def plot_hidden(time,hidden, index = None, return_graph = False):
    fig = plt.figure(figsize=(12, 6))
    canvas = FigureCanvas(fig)
    # in case hidden is not in the right shape
    if hidden.shape[1] > hidden.shape[0]:
        hidden = hidden.T
    plt.plot(time[:index], hidden[:index])
    plt.title("Hidden dynamics")
    if return_graph:
        fig.canvas.draw()
        graph = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
        w, h = fig.canvas.get_width_height()
        graph = graph.reshape((h, w, 3))
        return graph

def plot_fr(time,fr, return_graph = False):
    fig = plt.figure(figsize=(12, 6))
    canvas = FigureCanvas(fig)
    # in case fr is not in the right shape
    if fr.shape[1] > fr.shape[0]:
        fr = fr.T
    plt.plot(time, fr)
    plt.title("Firing rate")
    if return_graph:
        fig.canvas.draw()
        graph = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
        w, h = fig.canvas.get_width_height()
        graph = graph.reshape((h, w, 3))
        return graph
    
def plot_log_fr(time,fr, return_graph = False):
    fig = plt.figure(figsize=(12, 6))
    canvas = FigureCanvas(fig)
    # in case fr is not in the right shape
    if fr.shape[1] > fr.shape[0]:
        fr = fr.T
    plt.plot(time, fr)
    plt.title("Log firing rate")
    if return_graph:
        fig.canvas.draw()
        graph = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
        w, h = fig.canvas.get_width_height()
        graph = graph.reshape((h, w, 3))
        return graph
    
def plot_log_fr_wrt_hidden(hidden,fr, return_graph = False):
    fig = plt.figure(figsize=(12, 6))
    canvas = FigureCanvas(fig)
    # in case hidden is not in the right shape
    if hidden.shape[1] > hidden.shape[0]:
        hidden = hidden.T
    # in case fr is not in the right shape
    if fr.shape[1] > fr.shape[0]:
        fr = fr.T
    plt.plot(hidden, fr)
    plt.title("Log firing rate w.r.t hidden")
    if return_graph:
        fig.canvas.draw()
        graph = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
        w, h = fig.canvas.get_width_height()
        graph = graph.reshape((h, w, 3))
        return graph

def plot_spk(time,spk, return_graph = False):
    fig = plt.figure(figsize=(12, 6))
    canvas = FigureCanvas(fig)
    # in case spk is not in the right shape
    if spk.shape[1] > spk.shape[0]:
        spk = spk.T
    plt.plot(time, spk, '.', markersize=1)
    plt.title("Spike")
    if return_graph:
        fig.canvas.draw()
        graph = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
        w, h = fig.canvas.get_width_height()
        graph = graph.reshape((h, w, 3))
        return graph

def plot_hidden_estimation(time, hidden, hidden_est, hidden_0 = None, x_lim = None, y_lim = None, return_graph = False):
    if hidden_est.shape[0] > hidden_est.shape[1]:
        hidden_est = hidden_est.T
    fig = plt.figure(figsize=(12, 6))
    canvas = FigureCanvas(fig)
    for dim in range(hidden.shape[0]):
        plt.subplot(hidden.shape[0],1,dim+1)
        plt.plot(time, hidden[dim], label="True")
        plt.plot(time, hidden_est[dim], label="Estimation")
        if hidden_0 is not None:
            plt.plot(time, hidden_0[dim], label="Initial")
        plt.legend()
        if x_lim is not None:
            plt.xlim(x_lim)
        if y_lim is not None:
            plt.ylim(y_lim)
    if return_graph:
        fig.canvas.draw()
        graph = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
        w, h = fig.canvas.get_width_height()
        graph = graph.reshape((h, w, 3))
        return graph

    
def plot_log_fr_estimation(time, fr, fr_est, fr0 = None, x_lim = None, y_lim = None, return_graph = False):
    if fr_est.shape[0] > fr_est.shape[1]:
        fr_est = fr_est.T
    fig = plt.figure(figsize=(12, 6))
    canvas = FigureCanvas(fig)
    for dim in range(fr.shape[0]):
        plt.subplot(fr.shape[0],1,dim+1)
        plt.plot(time, fr[dim], label="True")
        plt.plot(time, fr_est[dim], label="Estimation")
        if fr0 is not None:
            plt.plot(time, fr0[dim], label="Initial")
        plt.legend()
        if x_lim is not None:
            plt.xlim(x_lim)
        if y_lim is not None:
            plt.ylim(y_lim)
    if return_graph:
        fig.canvas.draw()
        graph = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
        w, h = fig.canvas.get_width_height()
        graph = graph.reshape((h, w, 3))
        return graph

def train_test_separator(time, hidden, fr, spk, train_ratio = 0.8):
    n_train = int(len(time) * train_ratio)
    n_test = len(time) - n_train
    train_time = time[:n_train]
    test_time = time[n_train:]
    train_hidden = hidden[:,:n_train]
    test_hidden = hidden[:,n_train:]
    train_fr = fr[:,:n_train]
    test_fr = fr[:,n_train:]
    train_spk = spk[:,:n_train]
    test_spk = spk[:,n_train:]
    return train_time, test_time, train_hidden, test_hidden, train_fr, test_fr, train_spk, test_spk

def affine_to(X0,X):
    def affine_mse(a, X1,X2):
        return LA.norm(X1 - (a[0]*X2 + a[1]))
    affine_coe = minimize(affine_mse, [1,0], args=(X0, X)).x
    return affine_coe[0]*X + affine_coe[1]
