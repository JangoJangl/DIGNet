import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


if __name__ == "__main__":
    parser = argparse.ArgumentParser("WandB Export Reader", description="@Kevin")
    parser.add_argument("-f", "--file", type=str, help="CSV File to generate plot from", default="/home/temp_store/jandl/Downloads/variance.csv")
    parser.add_argument("-p", "--path", type=str, help="Path to save the plot to",
                        default="plots/wandb")
    parser.add_argument("-c", "--colors", nargs="*", help="The colors to draw the plots with, in order (Use Hexcode #RRGGBB). If less then then number of runs are given, default matplotlib colors are used", default=["#007749", "#3368A7", "#ec9d27", "#8c000F", "#800080", "#06c2ac"])
    parser.add_argument("-a", "--alpha", type=str, help="The alpha part of the shaded areas as hexstring AA", default="22")
    parser.add_argument("--fs", type=int, help="Fontsize in the plot", default=8)
    parser.add_argument("--ticksize", type=int, help="Interval between ticks on the X-Axis", default=5000)
    parser.add_argument("-s", "--size", nargs="+", help="Figure Size in inches. If only one is given, it is assumed to be the height, and width is set to 5.90666 inch", default=[3.3225])
    parser.add_argument("--dpi", type=int, help="DPI to save the figure with", default=600)
    parser.add_argument("--output", type=str, help="Output file to save the figure to", default=None)
    args = parser.parse_args()

    if len(args.size) < 2:
        args.size = [5.9066, args.size[0]]
    print(args)
    save_path = args.path
    pth = args.file
    with open(pth, "r") as f:
        lines = f.readlines()
        cols = len(lines[1].split(","))

    converters = {0: lambda s: int(s.replace("\"", ""))}
    for i in range(1, cols):
        converters[i] = lambda s: float(s.replace("\"", ""))
    data = np.genfromtxt(
        pth, 
        names=True, 
        delimiter=',', 
        dtype=None, 
        encoding="ascii", 
        deletechars="\"",
        replace_space=False, 
        converters=converters
    )
    
    runs = [s for s in data.dtype.names[1:] if not (s.endswith("__MIN") or s.endswith("__MAX"))]
    experiment = runs[0].split("-")[-1]
    
    if args.output is None:
        args.output = f"{save_path}/{experiment}.pdf"

    labels = [s.replace(experiment, "").replace("-", "").replace("_", " ").title().strip() for s in runs]
    experiment = experiment.replace("_", " ").title().strip()

    x = data["Step"]

    xticks = list(filter(lambda x: not x % args.ticksize, x))
    xticklabels = [f"{t // 1000:d}K" for t in xticks]

    fig, ax = plt.subplots(nrows=1, ncols=1)   
    fig.set_size_inches(*args.size)
    plt.rc("font", size=args.fs)
    for run, label, c in zip(runs, labels, args.colors):
        # create a dataframe
        df = pd.DataFrame(
            {'data': data[run]})
        # df = df.ewm(com=10).mean()
        ax.plot(x, df, color=c, label=label)
        ax.fill_between(x, data[run + "__MIN"], data[run + "__MAX"], color=c + "22")
    
    ax.set_xticklabels(xticklabels[1:])
    ax.set_xticks(xticks[1:])

    ax.set_ylim(top=1, bottom=0)
    ax.set_xlim(left=0)
    plt.title(experiment, pad=args.fs + 20)
    
    # fig.legend(loc="upper center", ncol=len(runs), bbox_to_anchor=(.52, .91))
    fig.tight_layout()
    plt.legend('', frameon=False)
    plt.savefig(args.output)
    #
    # plt.show()
