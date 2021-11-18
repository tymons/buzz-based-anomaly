import numpy as np

from matplotlib.ticker import FormatStrFormatter


def scatter_feature_temperature_for_temperature_with_std(ax, x, y, x_trend, y_trend, residuals, start_x, ref_x):
    # plot feature trend
    ax.plot(x_trend, y_trend, 'C0--')

    # scatter features
    ax.scatter(x, y, c='r', alpha=0.6)
    ax.set_xlabel('temperature')
    ax.set_xticks(x_trend)
    ax.set_xticklabels(ax.get_xticks(), rotation=90)
    ax.set_ylabel('features')

    ax_twin = ax.twinx()
    ax_twin.set_ylabel('std (poly/mean)')

    bars = ax_twin.bar(x_trend, residuals, width=1, color='C0', alpha=0.6)
    ax_twin.set_ylim((0, 3.5 * max(residuals)))

    start_idx = (np.where(x_trend == start_x)[0][0])
    common_temperature_idx = (np.where(x_trend == ref_x)[0][0])
    bars[common_temperature_idx].set_color('k')

    for idx in np.arange(start_idx, len(bars), 1):
        bars[idx].set_color('g')
        height = bars[idx].get_height()
        ax_twin.annotate(f'{height:.2f}', xy=(bars[idx].get_x() + bars[idx].get_width() / 2, height),
                         xytext=(1, 3),  # 3 points vertical offset
                         textcoords="offset points",
                         ha='center', va='bottom', rotation='90', fontsize='small')


def histogram_feature_weather(ax, histogram_values, bins, weather_values, bar_width):
    ax.grid(True, axis='y')
    ax.bar(bins, histogram_values, width=bar_width, edgecolor='k')
    ax.set_xlabel('feature')
    ax.set_ylabel('count')
    ax.set_ylim((0, 28))
    ax.set_yticks(range(0, 32, 4))
    ax.set_xticks(bins)
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    ax.set_xticklabels([str(round(float(label), 3)) for label in ax.get_xticks() if label != ''], rotation=90)

    ax_twin = ax.twinx()
    bars = ax_twin.bar(bins, weather_values, width=bar_width, color='r', alpha=0.3)
    ax_twin.set_ylim((0, 50))
    ax_twin.set_yticks(range(0, 55, 5))
    ax_twin.set_ylabel('weather')
    for bar in bars:
        height = bar.get_height()
        ax_twin.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                         xytext=(1, 3),  # 3 points vertical offset
                         textcoords="offset points",
                         ha='center', va='bottom', rotation='90', fontsize='small')
