import matplotlib.pyplot as plt
import numpy as np
import tikzplotlib

files=["size11optim1.csv","size11optim2.csv","size11optim4.csv"]
labels=["$n_{optim}="+str(n)+"$" for n in [1,2,4]]

# Remove headers
parsed=[np.genfromtxt(file,delimiter=",")[1:] for file in files]

if __name__ == "__main__":
    plt.style.use("ggplot")
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title("Loss over Iterations")
    ax.set_xlabel("Number of Iterations")
    ax.set_ylabel("Loss")
    ax.ticklabel_format(axis='y', style="scientific")

    for i in range(len(parsed)):
        ax.plot(parsed[i][:,1], parsed[i][:,2],label=labels[i])

    ax.legend()
    ax.grid(True)
    tikzplotlib.save("loss.tex")
    plt.savefig("loss.png")
