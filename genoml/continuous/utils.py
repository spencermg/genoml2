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
import pandas as pd
import seaborn as sns
import statsmodels.formula.api as sm
from genoml import utils
from sklearn import metrics


def export_prediction_data(out_dir, ids, step, algorithm, y, x, y_withheld=None, x_withheld=None, ids_withheld=None):

    if isinstance(algorithm, list):
        all_results = []
        all_withheld_results = []

        for fold, algo in enumerate(algorithm):
            results, withheld_results = _collect_prediction_data(
                ids[fold], 
                step, 
                algo, 
                y[fold], 
                x[fold],
                y_withheld=y_withheld[fold] if y_withheld is not None else None,
                x_withheld=x_withheld[fold] if x_withheld is not None else None,
                ids_withheld=ids_withheld[fold] if ids_withheld is not None else None,
                fold=fold,
            )

            all_results.append(results)
            if withheld_results is not None:
                all_withheld_results.append(withheld_results)

        # Combine all folds
        combined_results = pd.concat(all_results, ignore_index=True)

        combined_results.to_csv(
            out_dir.joinpath(f"{step}_predictions.txt"),
            index=False,
            sep="\t"
        )

        # Plot combined
        _plot_combined_results(out_dir, combined_results)

        # Still do per-fold regression summaries
        for fold_df in all_results:
            _run_regression_summary(out_dir, fold_df)

    else:
        _export_prediction_data(out_dir, ids, step, algorithm, y, x, y_withheld=y_withheld, x_withheld=x_withheld)


def _collect_prediction_data(ids, step, algorithm, y, x, y_withheld=None, x_withheld=None, ids_withheld=None, fold=None):

    y_predicted = algorithm.predict(x)

    results = pd.DataFrame({
        "ID": ids,
        "REPORTED": y,
        "PREDICTED": y_predicted,
        "fold": fold + 1
    })

    withheld_results = None

    if x_withheld is not None:
        y_withheld_predicted = algorithm.predict(x_withheld)

        withheld_results = pd.DataFrame({
            "ID": ids_withheld,
            "REPORTED": y_withheld,
            "PREDICTED": y_withheld_predicted,
            "fold": fold + 1
        })

    return results, withheld_results


def _plot_combined_results(out_dir, df):

    plt.figure()

    sns.scatterplot(
        data=df,
        x="REPORTED",
        y="PREDICTED",
        style="fold",
        hue="fold",
        palette="tab10"
    )

    # Fit overall regression line
    reg_model = sm.ols("REPORTED ~ PREDICTED", data=df).fit()

    sns.regplot(
        data=df,
        x="REPORTED",
        y="PREDICTED",
        scatter=False,
        color="black"
    )

    plt.text(
        0.95, 0.05,
        f"$R^2 = {reg_model.rsquared:.3f}$",
        ha="right",
        va="bottom",
        transform=plt.gca().transAxes,
        fontsize=14,
    )

    plt.savefig(out_dir.joinpath("regression.png"), dpi=600)
    plt.clf()


def _run_regression_summary(out_dir, df):

    fold = df["fold"].iloc[0]

    reg_model = sm.ols("REPORTED ~ PREDICTED", data=df).fit()

    with open(out_dir.joinpath(f"regression_summary_fold{fold}.txt"), "w") as f:
        f.write(reg_model.summary().as_csv().replace(",", "\t"))


# def export_prediction_data(out_dir, ids, step, algorithm, y, x, y_withheld=None, x_withheld=None):
#     """
#     Save table with predicted vs. reported phenotypes for each sample and generate regression model.

#     Args:
#         out_dir (pathlib.Path): Path to output directory.
#         ids (numpy.ndarray): Array of sample IDs.
#         step (str): Step of the GenoML workflow. 
#         y (numpy.ndarray): Array of reported phenotypes for each training sample.
#         x (numpy.ndarray): Array of input data for each training sample.
#         y_withheld (numpy.ndarray, optional): Array of reported phenotypes for each validation sample (Default: None).
#         x_withheld (numpy.ndarray, optional): Array of input data for each validation sample (Default: None).
#     """

#     if isinstance(algorithm, list):
#         for fold, algo in enumerate(algorithm):
#             if y_withheld is not None and x_withheld is not None:
#                 _export_prediction_data(out_dir, ids[fold], step, algo, y[fold], x[fold], y_withheld=y_withheld[fold], x_withheld=x_withheld[fold], fold=fold)
#             else:
#                 _export_prediction_data(out_dir, ids[fold], step, algo, y[fold], x[fold], y_withheld=None, x_withheld=None, fold=fold)
#     else:
#         _export_prediction_data(out_dir, ids, step, algorithm, y, x, y_withheld=y_withheld, x_withheld=x_withheld)


def _export_prediction_data(out_dir, ids, step, algorithm, y, x, y_withheld=None, x_withheld=None, fold=None):
    """
    Save table with predicted vs. reported phenotypes for each sample and generate regression model.

    Args:
        out_dir (pathlib.Path): Path to output directory.
        ids (numpy.ndarray): Array of sample IDs.
        step (str): Step of the GenoML workflow. 
        y (numpy.ndarray): Array of reported phenotypes for each training sample.
        x (numpy.ndarray): Array of input data for each training sample.
        y_withheld (numpy.ndarray, optional): Array of reported phenotypes for each validation sample (Default: None).
        x_withheld (numpy.ndarray, optional): Array of input data for each validation sample (Default: None).
        fold (int): If using outer cross-validation, fold number corresponding to current data/algorithm (Default: None).
    """

    suffix = f"_fold{fold+1}" if fold is not None else ""

    y_predicted = algorithm.predict(x)
    if x_withheld is not None:
        y_withheld_predicted = algorithm.predict(x_withheld)

    output_columns = ["ID", "REPORTED", "PREDICTED"]

    # Training results.
    results = pd.DataFrame(
        zip(ids, y, y_predicted), 
        columns=output_columns,
    )
    with utils.DescriptionLoader.context(
        "utils/export_predictions",
        output_path=out_dir.joinpath(f"{step}_predictions{suffix}.txt"), 
        data=results.head(),
    ):
        results.to_csv(out_dir.joinpath(f"{step}_predictions{suffix}.txt"), index=False, sep="\t")

    # Withheld results, if applicable.
    if step == "training":
        results = pd.DataFrame(
            zip(ids, y_withheld, y_withheld_predicted), 
            columns=output_columns,
        )
        with utils.DescriptionLoader.context(
            "utils/export_predictions/withheld_data",
            output_path=out_dir.joinpath(f"withheld_predictions{suffix}.txt"), 
            data=results.head(),
        ):
            results.to_csv(out_dir.joinpath(f"withheld_predictions{suffix}.txt"), index=False, sep="\t")

    # Regression model on withheld results.
    reg_model = sm.ols(formula=f'REPORTED ~ PREDICTED', data=results)
    fitted = reg_model.fit()
    with utils.DescriptionLoader.context(
        "utils/export_predictions/plot",
        output_path=out_dir.joinpath(f"regression{suffix}.png"), 
        data=fitted.summary(),
    ):
        sns_plot = sns.regplot(
            data=results, 
            x=f"REPORTED", 
            y=f"PREDICTED", 
            scatter_kws={"color": "blue"},
            line_kws={"color": "red"},
        )
        plt.text(
            0.95, 0.05,
            f"$R^2 = {fitted.rsquared:.3f}$",
            ha="right", 
            va="bottom",
            transform=plt.gca().transAxes,
            fontsize=18,
        )
        sns_plot.set_xlabel(f"Reported", fontsize=16)
        sns_plot.set_ylabel(f"Predicted", fontsize=16)

        sns_plot.figure.savefig(out_dir.joinpath(f"regression{suffix}.png"), dpi=600)
        plt.clf()
        with open(out_dir.joinpath(f"regression_summary{suffix}.txt"), "w") as f:
            f.write(fitted.summary().as_csv().replace(",", "\t"))


def additional_sumstats(algorithm_name, y_test, x_test, algorithm, run_prefix):
    if isinstance(y_test, list):
        for fold, y_test_fold in enumerate(y_test):
            y_pred_fold = algorithm[fold].predict(x_test[fold])
            _additional_sumstats(algorithm_name, y_test_fold, y_pred_fold, run_prefix, fold=fold)
    else:
        y_pred = algorithm.predict(x_test)
        _additional_sumstats(algorithm_name, y_test, y_pred, run_prefix)


def _additional_sumstats(algorithm_name, y_test, y_pred, run_prefix, fold=None):
    suffix = f"_fold{fold+1}" if fold is not None else ""
    log_table = pd.DataFrame(
        data=[[algorithm_name] + list(_calculate_accuracy_scores(y_test, y_pred))], 
        columns=["Algorithm", "Explained Variance", "Mean Squared Error", "Median Absolute Error", "R-Squared_Error"],
    )
    log_outfile = run_prefix.joinpath(f"performance_metrics{suffix}.txt")
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
