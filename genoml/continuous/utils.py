# Copyright 2020 The GenoML Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.formula.api as sm
from itertools import repeat
from genoml import utils
from sklearn import metrics


def export_prediction_data(out_dir, ids, step, algorithm, y, x, is_using_outer_cv, y_withheld=None, x_withheld=None, ids_withheld=None):
    if is_using_outer_cv:
        # Store prediction results for each fold
        results = []
        withheld_results = []
        for fold, algo in enumerate(algorithm):
            y_predicted = algo.predict(x[fold])
            results.append(pd.DataFrame(
                zip(ids[fold], y[fold], y_predicted, repeat(fold + 1)), 
                columns=["ID", "REPORTED", "PREDICTED", "FOLD"],
            ))
            if x_withheld is not None:
                y_withheld_predicted = algo.predict(x_withheld[fold])
                withheld_results.append(pd.DataFrame(
                    zip(ids_withheld[fold], y_withheld[fold], y_withheld_predicted, repeat(fold + 1)), 
                    columns=["ID", "REPORTED", "PREDICTED", "FOLD"],
                ))

        # Combine all folds
        results = pd.concat(results, ignore_index=True)
        if step == "training":
            withheld_results = pd.concat(withheld_results, ignore_index=True)
            
    else:
        y_predicted = algorithm.predict(x)
        if x_withheld is not None:
            y_withheld_predicted = algorithm.predict(x_withheld)

        # Training results.
        results = pd.DataFrame(
            zip(ids, y, y_predicted), 
            columns=["ID", "REPORTED", "PREDICTED"],
        )
        if step == "training":
            withheld_results = pd.DataFrame(
                zip(ids_withheld, y_withheld, y_withheld_predicted), 
                columns=["ID", "REPORTED", "PREDICTED"],
            )
    
    with utils.DescriptionLoader.context(
        "utils/export_predictions",
        output_path=out_dir.joinpath(f"{step}_predictions.tsv"), 
        data=results.head(),
    ):
        results.to_csv(
            out_dir.joinpath(f"{step}_predictions.tsv"), 
            index=False, 
            sep="\t",
        )
    if step == "training":
        with utils.DescriptionLoader.context(
            "utils/export_predictions/withheld_data",
            output_path=out_dir.joinpath(f"{step}_predictions_withheld.tsv"), 
            data=withheld_results.head(),
        ):
            withheld_results.to_csv(
                out_dir.joinpath(f"{step}_predictions_withheld.tsv"), 
                index=False, 
                sep="\t",
            )

    # Plot results
    _plot_results(
        out_dir.joinpath(f"regression_summary.txt"),
        out_dir.joinpath(f"regression.png"), 
        results,
        is_using_outer_cv,
    )
    if step == "training":
        _plot_results(
            out_dir.joinpath(f"regression_summary_withheld.txt"),
            out_dir.joinpath("regression_withheld.png"), 
            withheld_results,
            is_using_outer_cv,
        )


def _plot_results(regression_summary_path, regression_plot_path, df, is_using_outer_cv):
    plt.figure()

    # Fit overall regression line and add it to the plot
    reg_model = sm.ols("REPORTED ~ PREDICTED", data=df).fit()

    with utils.DescriptionLoader.context(
        "utils/export_predictions/plot",
        output_path=regression_plot_path, 
        data=reg_model.summary(),
    ):
        # Plot reported vs predicted value for each sample
        if is_using_outer_cv:
            sns.scatterplot(
                data=df,
                x="REPORTED",
                y="PREDICTED",
                style="FOLD",
                hue="FOLD",
                palette="tab10",
            )
        else:
            sns.scatterplot(
                data=df,
                x="REPORTED",
                y="PREDICTED",
                palette="tab10",
            )
        
        with open(regression_summary_path, "w") as f:
            f.write(reg_model.summary().as_csv().replace(",", "\t"))
        sns.regplot(
            data=df,
            x="REPORTED",
            y="PREDICTED",
            scatter=False,
            color="black",
            label="Best-fit trendline",
        )

        # Add dashed line at y=x
        ax = plt.gca()
        lims = [
            np.min([ax.get_xlim(), ax.get_ylim()]),
            np.max([ax.get_xlim(), ax.get_ylim()]),
        ]
        ax.plot(
            lims, 
            lims, 
            'k--', 
            alpha=0.5, 
            zorder=0, 
            label="Predicted = Reported",
        )

        plt.text(
            0.95, 0.05,
            f"$R^2 = {reg_model.rsquared:.3f}$",
            ha="right",
            va="bottom",
            transform=plt.gca().transAxes,
            fontsize=14,
        )

        plt.legend(fontsize=8)

        plt.savefig(regression_plot_path, dpi=600)
        plt.clf()


def additional_sumstats(algorithm_name, y_test, x_test, algorithm, prefix, is_using_outer_cv):
    if is_using_outer_cv:
        for fold, y_test_fold in enumerate(y_test):
            y_pred_fold = algorithm[fold].predict(x_test[fold])
            _additional_sumstats(algorithm_name, y_test_fold, y_pred_fold, prefix, fold=fold)
    else:
        y_pred = algorithm.predict(x_test)
        _additional_sumstats(algorithm_name, y_test, y_pred, prefix)


def _additional_sumstats(algorithm_name, y_test, y_pred, prefix, fold=None):
    suffix = f"_fold{fold+1}" if fold is not None else ""
    log_table = pd.DataFrame(
        data=[[algorithm_name] + list(_calculate_accuracy_scores(y_test, y_pred))], 
        columns=["Algorithm", "Explained Variance", "Mean Squared Error", "Median Absolute Error", "R-Squared_Error"],
    )
    log_outfile = prefix.joinpath(f"performance_metrics{suffix}.tsv")
    log_table.to_csv(log_outfile, index=False, sep="\t")


def calculate_accuracy_scores(x, y, algorithm):
    """
    Calculate accuracy metrics for the chosen continuous prediction model.

    Args:
        x (pandas.DataFrame): Model input features.
        y (pandas.DataFrame): Reported output features.
        algorithm: Contonuous prediction algorithm.

    :return: accuracy_metrics *(list)*: \n
        Accuracy metrics used for the continuous prediction module.
    """

    y_pred = algorithm.predict(x)
    return _calculate_accuracy_scores(y, y_pred)


def _calculate_accuracy_scores(y, y_pred):
    """
    Calculate accuracy metrics for the chosen continuous prediction model.

    Args:
        y (pandas.DataFrame): Reported output features.
        y_pred (pandas.DataFrame): Predicted output features.

    :return: evs *(float)*: \n
        Explained Variance Score.
    :return: mse *(float)*: \n
        Mean Squared Error.
    :return: mae *(float)*: \n
        Median Absolute Error.
    :return: r2s *(float)*: \n
        R^2 Score.
    """

    evs = metrics.explained_variance_score(y, y_pred)
    mse = metrics.mean_squared_error(y, y_pred)
    mae = metrics.median_absolute_error(y, y_pred)
    r2s = metrics.r2_score(y, y_pred)

    return evs, mse, mae, r2s
