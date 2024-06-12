import matplotlib.pyplot as plt

def plot_histogram(data, title, xlabel, ylabel, bins=5):
    plt.hist(data, bins=bins)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.xticks(range(0,bins+1))
    plt.ylabel(ylabel)
    # save the plot as pdf with dpi=300
    plt.savefig(title.replace(" ", "_") + ".pdf", dpi=300, bbox_inches='tight', pad_inches=0.1, format='pdf')