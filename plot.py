import seaborn
from matplotlib import pyplot
import matplotlib.pyplot as plt
import matplotlib

from typing import List
import numpy as np


def bar_plot_top_n(data: List[np.ndarray], version: str) -> None:
    raw_data = {"Words": data[0],
                "Counts": data[1]}
    a4_dims = (9.7, 6.27)
    fig, ax = pyplot.subplots(figsize=a4_dims)
    plot = seaborn.barplot(x="Words", y="Counts", ax=ax, data=raw_data)
    patches = [matplotlib.patches.Patch(
        color=seaborn.color_palette()[i], label=t) for i, t in enumerate(
        t.get_text() for t in plot.get_xticklabels())]
    plt.legend(handles=patches, loc="upper right")
    plt.title(f"Top-10 words for {version} version")
