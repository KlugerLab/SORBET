"""Helper plotting functions for results. 
"""
from typing import Union, List, Optional
from dataclasses import dataclass
from itertools import combinations

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from sklearn import metrics as metrics

from ray.tune import ResultGrid 
from .models import BaseGraphModel

@dataclass
class PlotStyle:
    """Data class assigning colors by model type.
    """
    svm: Union[str, List[int]] = "#FFC61E"
    mlp: Union[str, List[int]] = "#AF58BA"
    basic_gcn: Union[str, List[int]] = "009ADE"
    sorbet: Union[str, List[int]] = "#FF1F5B"
    external: Union[str, List[int]] = "#00CD6C"
    # GeoDataVis Colour Palette, accessed at https://www.molecularecologist.com/2020/04/23/simple-tools-for-mastering-color-in-scientific-figures/

def plot_combined_metrics(computed_data: List[str, List[float], List[int]], plot_roc: Optional[bool] = True, plot_pr: Optional[bool] = True,
        figsize: Optional[List[int]] = None) -> plt.figure:
    """Plots the model performance for each of the passed in data attributes.

    Args:
        computed_data: list of tuples including the model, predictions and labels
        plot_roc: whether to plot the ROC curve
        plot_pr: whether to plot Precision-Recall curve
        figsize: passed figsize

    Returns:
        A matplotlib figure with the plotted metrics
    """
    if plot_roc and plot_pr:
        figsize = (10,5) if figsize is None else figsize
        fig, axes = plt.subplots(1,2,figsize=figsize)
        roc_idx, pr_idx = 0, 1
    elif plot_roc:
        figsize = (5,5) if figsize is None else figsize
        fig, axes = plt.subplots(1,1,figsize=figsize)
        roc_idx, axes = 0, [axes]
    elif plot_pr:
        figsize = (5,5) if figsize is None else figsize
        fig, axes = plt.subplots(1,1,figsize=figsize)
        pr_idx, axes = 0, [axes]
    else:
        raise ValueError("No plotting function called. Re-run with choice of ROC and / or PR curve")
    
    if plot_roc:
        for data_type, predictions, labels in computed_data:
            fpr, tpr, _ = metrics.roc_curve(labels, predictions)
            curve_color, curve_ls = "k", "-"
            axes[roc_idx].plot(fpr, tpr, color=curve_color, ls=curve_ls, label=f'{data_type}: {metrics.roc_auc_score(labels, predictions):.3f}')
        axes[roc_idx].plot([0,1],[0,1],'k--')
        axes[roc_idx].grid(True)
        axes[roc_idx].set_xlim(-0.01,1.01); axes[roc_idx].set_ylim(-0.01, 1.01)
        axes[roc_idx].legend(loc='lower right')
        axes[roc_idx].set(title="Receiver Operator Characteristic", xlabel="FPR", ylabel="TPR", xlim=(-0.01, 1.01), ylim=(-0.01, 1.01))

    if plot_pr:
        for data_type, predictions, labels in computed_data:
            prec, recall, _ = metrics.precision_recall_curve(labels, predictions)
            curve_color, curve_ls = "k", "-"
            axes[pr_idx].plot(recall, prec, color=curve_color, ls=curve_ls, label=f'{data_type}: {metrics.average_precision_score(labels, predictions):.3f}')
        
        for f_score in np.linspace(0.2, 0.8, 4):
            x = np.linspace(0.01, 1.0, 50)
            y = f_score * x / (2 * x - f_score)

            axes[pr_idx].plot(x[y >= 0], y[y >= 0], color='gray', ls='--', alpha=0.2)
            axes[pr_idx].annotate(f'F1={f_score:.1f}', xy=(0.85, y[45] + 0.03))
        axes[pr_idx].legend(loc='lower left')
        axes[pr_idx].set(title="Precision-Recall", xlabel="Recall", ylabel="Precision", ylim=(-0.01, 1.01), xlim=(-0.01, 1.01))

    return fig

def plot_model_calibration(labels: List[int], predictions: List[float], 
        plot_style: PlotStyle = PlotStyle(), window_size: int = 15, bins: np.ndarray = np.linspace(0,1,11)) -> plt.figure:
    """Plots model calibration. Assumes model predictions are in range [0,1].

    Args:
        labels: list of labels
        predictions: list of predictions
        plot_style: a data class describing the figures style
        window_size: [Deprecated] Ignore. 
        bins: bins to use in the histogram.

    Returns:
        A matplotlib figure with the plotted model calibration.
    """
    predictions, labels = np.array(predictions), np.array(labels)

    fig, axes = plt.subplots(1,3,figsize=(15,5))

    axes[0].plot([0,1],[0,1], c="#DCDCDC", ls='--')
    for i,j in zip(bins, bins[1:]):
        mask = np.logical_and(predictions >= i, predictions <= j)
        mid = np.mean(labels[mask])
        axes[0].plot([i,j],[mid,mid], 'k-')
        print(i,j,np.count_nonzero(mask))

    axes[0].grid(True)
    axes[0].set(title="Calibration", xlabel="Prediction", ylabel="Empirical")
    
    axes[1].hist(labels, bins = bins, histtype='step', color='k', label="Observed") 
    axes[1].hist(predictions, bins = bins, color='#FF1F5B', label="Predictions")
    axes[1].set(title="Predicted Distribution", xlabel="Prediction", ylabel="Count")
    axes[1].legend()
    
    for ax in axes[:2]:
        ax.set_xlim(-0.01, 1.01)
    
    neg_predictions = predictions[labels == 0]
    pos_predictions = predictions[labels == 1]

    axes[2].boxplot([neg_predictions, pos_predictions], notch=True, labels = ["0", "1"], positions=[0,1])
    axes[2].scatter(np.random.normal(0, 0.05, len(neg_predictions)), neg_predictions, color='k', s=3)
    axes[2].scatter(np.random.normal(1, 0.05, len(pos_predictions)), pos_predictions, color='k', s=3)
    axes[2].set_ylim(-0.01, 1.01)
    axes[2].set(title="Conditional Predictions", xlabel="Observation", ylabel="Prediction", ylim=(-0.05, 1.05))
    axes[2].grid(True)

    return fig

def _match_config_names(param_name: str, dataframe_cols: List[str], serializable: bool, nlayer: bool = True) -> str:
    """Helper function to decode names in the config dictionary

    Args:
        param_name: the parameter name using Optuna's structure
        dataframe_cols: the names of all values in the config
        serializable: whether the parameter is a serializable parameter
        nlayer: whether to match on the number of layers for serializable parameters

    Returns:
        The stripped config name
    """

    config_names = [ci for ci in dataframe_cols if 'config/' in ci]

    if serializable and nlayer:
        condition = lambda s: 'config/' in s and 'n_layers' in s and param_name in s
        param_tune_name = next(filter(condition, dataframe_cols)) 
        return param_tune_name
    elif serializable:
        condition = lambda s: 'config/' in s and param_name in s and 'n_layers' not in s
        param_tune_names = sorted(filter(condition, dataframe_cols), key=lambda x: int(x.split("_")[-1]))
        return param_tune_names
    else:
        condition = lambda s: 'config/' in s and param_name in s
        param_tune_name = next(filter(condition, dataframe_cols)) 
        return param_tune_name

def _is_logscale(data: np.ndarray, log_thr: float = 2):
    """Helper function to understand if hyperparameter has a log-like structure.
    Rough estimate; not always correct.

    NOTE: If this fails, the appropriate axis can still be set manually (e.g., ax.set_xscale('log') for the relevant axis).
    Scale may be inappropriate otheriwse. 

    Args:
        data: a numpy array with the sampled hyperparameters
        log_thr: the number of log changes necessary to call a data range log-like

    Returns:
        Whether it is likely that the hyperparameter has log-structure.
    """
    if 0 in data:
        return False
    log_data = np.log(data)
    
    has_fold_change = (np.max(log_data) - np.min(log_data)) > log_thr
    is_float = np.max(log_data) <= 1
    return has_fold_change and is_float


def plot_hyperparameter_performance(results: ResultGrid, parameter: Union[str, List[str]], metric: str = "auroc", ncols: int = 3
        ) -> plt.figure:
    """Plots model performance as a function of hyperparameters. Primarily useful w.r.t. training hyperparamters.

    Args:
        results: a results grid object to plot optimization results
        parameter: the parameter or list of parameters to plot
        metric: the key for the value to plot. Default: AUROC
        ncols: number of columns to split plotted parameters over. Corresponds to n // 3 rows.

    Returns:
        A plot of the sampled hyperparameters with associated performance
    """
    def _plot_config_axis(ax, param_name, df=results.get_dataframe()):
        config_name = _match_config_names(param_name, df.columns, serializable=False)
        print(config_name)
        data = df[[metric, config_name]].to_numpy()

        ax.scatter(data[:,1], data[:,0], color='k')
        if _is_logscale(data[:,1]):
            ax.set_xscale('log')
            xlabel = f'log({param_name})'
        else:
            xlabel = f'{param_name}'
        ax.grid(True)
        ax.set(title=f'Tuning: {param_name}', xlabel=xlabel, ylabel=f'AUROC', ylim=(0.0, 1.01)) # Should the minimum on ylim be higher?

    if isinstance(parameter, list): 
        # Bit ugly. Have to do this for cases where ncols perfectly divides parmaeter length.
        nr, nc = int(np.ceil(len(parameter) / ncols)), ncols
        fig, axes = plt.subplots(nr, nc, figsize=(5 * nc, 5 * nr))

        for idx, param in enumerate(parameter):
            i, j = idx // nc, idx % nc
            ax = axes[i,j] if nr > 1 else axes[idx % nc]
            print(ax)
            _plot_config_axis(ax, param)
    
        for idx in range(len(parameter), nr * nc):
            i, j = idx // nc, idx % nc
            axes[i,j].axis('off')

    else:
        fig, axes = plt.subplots(figsize=(5, 5))
        _plot_config_axis(axes, parameter)
            
    return fig

def plot_serializable_model_performance(results: ResultGrid, parameter: Union[None, str, List[str]], model_type: BaseGraphModel, 
        nlayers: bool = True, metric: str = "auroc", auroc_cutoff: Union[None, float] = None, cmap="coolwarm", ncols: int = 3
        ) -> plt.figure:
    """Plots model performance as a function of serializable model structure.

    Args:
        results: a results grid object to plot optimization results
        parameter: the parameter or list of parameters to plot
        model_type: the choice of model to train. See models/.
        nlayers: number of layers included in plots
        metric: the key for the value to plot. Default: AUROC
        auroc_cutoff: a minimum value for configuration parameters to include in final plot
        cmap: the chose colormap for plotting parameter results
        ncols: number of columns to split plotted parameters over. Corresponds to n // 3 rows.

    Returns:
        Returns the plotted figure for serializable parameters. 
    """
    df = results.get_dataframe()
    def _plot_serializable_by_layer_size(ax, param_name, aurocs, layers, auroc_cutoff = auroc_cutoff):
        xs = np.arange(1, layers.shape[1] + 1)

        cm = plt.get_cmap(cmap)
        n = colors.Normalize(0,1)
        
        thr = 0 if auroc_cutoff is None else auroc_cutoff 
        for auroc, row in zip(aurocs, layers):
            if auroc < thr: continue 
            crop = min((idx for idx in range(len(row)) if np.isnan(row[idx])), default=len(row))
            ax.plot(xs[:crop], row[:crop], color=cm(n(auroc)))

        ax.grid(True)
        ax.set(title=f'Model: {param_name}', xlabel="Layer Index (n)", ylabel="Layer Width (n)")
        ax.set_xticks(xs)

    def _plot_serializable_by_n_layers(ax, param_name, aurocs, xs, ys = None, auroc_cutoff = auroc_cutoff):
        if ys is None:
            ax.scatter(xs, aurocs, color='k')
            ax.set(title=f'Model Struct: {param_name}', xlabel=f'{param_name} Layers (n)', ylabel="AUROC", ylim=(0.0, 1.01))
            ax.set_xticks(np.arange(min(xs), max(xs)+1))
        else:
            jiggle_eps = 0.075
            xs_jiggle = xs + np.random.normal(0, jiggle_eps, len(xs))
            ys_jiggle = ys + np.random.normal(0, jiggle_eps, len(ys))
            
            thr = 0 if auroc_cutoff is None else auroc_cutoff
            ma = aurocs >= thr

            ax.scatter(xs_jiggle[ma], ys_jiggle[ma], c = aurocs[ma], norm = colors.Normalize(vmin=0, vmax=1), cmap=cmap)
            ax.set(title=f'Model: {param_name[0]} vs {param_name[1]}', xlabel=f'{param_name[0]} (n)', ylabel=f'{param_name[1]} Layers (n)')
            
            ax.set_xticks(np.arange(np.min(xs), np.max(xs + 1)))
            ax.set_yticks(np.arange(np.min(ys), np.max(ys + 1)))

        ax.grid(True)

    if parameter is None or isinstance(parameter, list):
        parameter = parameter if parameter is not None else model_type.get_model_specification().model_serializable_parameters
        nr, nc = len(parameter), len(parameter)
        
        fig, axes = plt.subplots(nr, nc, figsize = (nr * 5, nc * 5))

        for idx, p in enumerate(parameter):
            config_names = _match_config_names(p, df.columns, serializable=True, nlayer=False)
            dat = df[[metric, *config_names]].to_numpy()
            _plot_serializable_by_layer_size(axes[idx, idx], p, dat[:,0], dat[:,1:])
        
        for (i, pi), (j, pj) in combinations(enumerate(parameter), 2):
            axes[j,i].axis('off')
            
            config_1_name =  _match_config_names(pi, df.columns, serializable=True, nlayer=True)
            config_2_name =  _match_config_names(pj, df.columns, serializable=True, nlayer=True)
            
            dat = df[[metric, config_1_name, config_2_name]].to_numpy()
            _plot_serializable_by_n_layers(axes[i,j], [pi, pj], dat[:,0], dat[:,1], dat[:,2])
        
        fig.subplots_adjust(right = 0.95)
        cbar_ax = fig.add_axes([0.125, 0.125, 0.5, 0.05])
        fig.colorbar(matplotlib.cm.ScalarMappable(norm=colors.Normalize(0,1), cmap=cmap), cax=cbar_ax, orientation='horizontal', label='AUROC')

    else:
        fig, axes = plt.subplots(figsize=(5,5))
        config_names = _match_config_names(parameter, df.columns, serializable=True, nlayer=nlayers)
        if nlayers:
            dat = df[[metric, config_names]].to_numpy()
            _plot_serializable_by_n_layers(axes, parameter, dat[:,0], dat[:,1]) 
        else:
            dat = df[[metric, *config_names]].to_numpy()
            _plot_serializable_by_layer_size(axes, parameter, dat[:,0], dat[:,1:])

    return fig

def plot_repeated_validation_curves(predictions: List[List[float]], labels: List[List[float]], name: str,
        color: Union[str, List[float]] = "k", line_args: dict = {'lw': 2.5, 'ls': '-'}, fill_args: dict = {'alpha': 0.1}, ax: plt.axis = None
        ) -> Optional[plt.figure]:
    """Plot combined curves

    Args:
        predictions: a list of lists of predictions
        labels: a corresponding list of lists of labels
        name: the name to assign to the curve
        color: a matplotlib color to apply to the curve
        line_args: additional line style keyword arguments passed to the plot
        fill_args: color to shade between the confidence interval lines
        ax: a matplotlib axis to plot the curve on

    Returns:
        Optionally returns a matplotlib figure if no axis is passed 
    """
    aurocs = list()
    fpr_thresholds = np.linspace(0,1,201)
    thresholded = list()

    for _pred, _lab in zip(predictions, labels):
        auroc = metrics.roc_auc_score(_lab, _pred)
        aurocs.append(auroc)

        fpr, tpr, _ = metrics.roc_curve(_lab, _pred)
        offs, _thr = 0, list()
        for fi in fpr_thresholds:
            if fi <= fpr[offs + 1]:
                _thr.append(tpr[offs])
            else:
                offs = next((j for j in range(offs+1, len(fpr)-2) if fi < fpr[j+1]), len(fpr)-2)
                _thr.append(tpr[offs])
        thresholded.append(_thr)

    mean_auroc, std_auroc = np.mean(aurocs), np.std(aurocs)
    print(f'{name} AUROC: {mean_auroc:.3f} (STD: {std_auroc:.3f})')

    thresholded = np.array(thresholded)

    return_fig = False
    if ax is None:
        fig, ax = plt.subplots(1,1,figsize=(5,5))
        return_fig = True

    label = f'{name}: {mean_auroc:.3f} ({std_auroc:.3f})'
    thr_means, thr_stds = np.mean(thresholded, axis=0), np.std(thresholded, axis=0)
    ax.plot(fpr_thresholds, thr_means, color=color, **line_args, label=label)
    ax.fill_between(fpr_thresholds, (thr_means - thr_stds), (thr_means + thr_stds), color=color, **fill_args)

    if return_fig:
        return fig

def format_roc_curve_axis(ax: plt.axis, xlim:List[float]=(-0.01, 1.01), ylim:List[float]=(-0.01, 1.01)):
    """A helper function to format the ROC curve

    Args:
        ax: a matplotlib axis to be formatted
        xlim: the x-range to plot over
        ylim: the y-range to plot over
    """
    ax.plot([0,1],[0,1],'k--')
    ax.grid(True)
    ax.legend(loc='lower right')
    ax.set(xlabel="FPR", ylabel="TPR", xlim=xlim, ylim=ylim)
