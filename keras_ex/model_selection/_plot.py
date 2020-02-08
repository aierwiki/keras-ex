import matplotlib.pyplot as plt


class ParamError(Exception):
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return repr(self.value)

def plot_learning_curve(x, y, title = "Learning Curve",
                      xlabel = "Epochs", ylabel="Loss",
                      ylim = None,
                      labels = None, colors = None):
    """
    Generate a simple plot of learning curve.
    Parameters
    --------------
    x: array-like, shape (n_epochs,)
    y: array-like, or list of array-like, shape (n_epochs,) or (n_scores, n_epochs)
    """
    check_params = {"labels":labels, "colors":colors}
    for k, v in check_params.items():
        if v != None and (type(v) is not list or len(v) != len(y)):
            raise ParamError("param {} must have same length with y".format(k))

    plt.figure()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid()
    if type(y) is list:
        for i, yy in enumerate(y):
            color = colors[i] if colors is not None else 'r'
            label = labels[i] if labels is not None else 'Score {}'.format(i)
            plt.plot(x, yy, 'o-', color=color, label=label)
    else:
        plt.plot(x, y, 'o-', color='r', label=ylabel)
    if ylim is not None:
        plt.ylim(ylim)
    plt.legend(loc='best')
    plt.show()
