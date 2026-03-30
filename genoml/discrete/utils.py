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
from genoml import utils
from sklearn import metrics


def plot_results(out_dir, y, x, algorithm, is_using_outer_cv):
    """
    Generate ROC and precision-recall plots for each class.

    Args:
        out_dir (pathlib.Path): Directory where results are saved.
        y (numpy.ndarray): Ground truth phenotypes.
        x (numpy.ndarray): Input values.
        algorithm: Discrete prediction algorithm.
    """

    if is_using_outer_cv:
        algorithm_name = utils.get_algorithm_name(algorithm[0])
        for fold, algo in enumerate(algorithm):
            _plot_results(out_dir, y[fold], x[fold], algo, algorithm_name, fold=fold)
    else:
        algorithm_name = utils.get_algorithm_name(algorithm)
        _plot_results(out_dir, y, x, algorithm, algorithm_name)


def _plot_results(out_dir, y, x, algorithm, algorithm_name, fold=None):
    """
    Generate ROC and precision-recall plots for each class.

    Args:
        out_dir (pathlib.Path): Directory where results are saved.
        y (numpy.ndarray): Ground truth phenotypes.
        x (numpy.ndarray): Input values.
        algorithm: Discrete prediction algorithm.
        algorithm_name (str): Classifier model name.
        fold (int): If using outer cross-validation, fold number corresponding to current data/algorithm (Default: None).
    """

    suffix = f"_fold{fold+1}" if fold is not None else ""
    y_pred_prob = algorithm.predict_proba(x)[:,1]
    roc_path = out_dir.joinpath(f"roc{suffix}.png")
    precision_recall_path = out_dir.joinpath(f"precision_recall{suffix}.png")
    ROC(roc_path, y, y_pred_prob, algorithm_name)
    precision_recall_plot(precision_recall_path, y, y_pred_prob, algorithm_name)


def ROC(plot_path, y, y_pred_prob, algorithm_name):
    """
    Generate ROC plots for each class given ground-truth values and corresponding predictions.

    Args:
        plot_path (str): File path where plot will be saved to.
        y (numpy.ndarray): Ground truth phenotypes.
        y_pred_prob (numpy.ndarray): Predicted case probabilities.
        algorithm_name (str): Label to add to plot title.
    """

    plt.figure()
    plt.plot([0, 1], [0, 1], 'r--')

    fpr, tpr, _ = metrics.roc_curve(y, y_pred_prob)
    roc_auc = metrics.roc_auc_score(y, y_pred_prob)
    plt.plot(fpr, tpr, color='purple', label=f'ROC curve (area = {roc_auc:.3f})')

    plt.xlim([0.0, 1.05])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title(f'Receiver operating characteristic (ROC) - {algorithm_name}', fontsize=10)
    plt.legend(loc="lower right")
    plt.savefig(plot_path, dpi=600)
    print(f"We are also exporting an ROC curve for you here {plot_path} this is a graphical representation of AUC "
          f"in the withheld test data for the best performing algorithm.")


def precision_recall_plot(plot_path, y, y_pred_prob, algorithm_name):
    """
    Generate precision-recall plots for each class given ground-truth values and corresponding predictions.

    Args:
        plot_path (str): File path where plot will be saved to.
        y (numpy.ndarray): Ground truth phenotypes.
        y_pred_prob (numpy.ndarray): Predicted case probabilities.
        algorithm_name (str): Label to add to plot title.
    """

    plt.figure()

    precision, recall, _ = metrics.precision_recall_curve(y, y_pred_prob)
    plt.plot(precision, recall, label="Precision-Recall curve")

    plt.xlim([0.0, 1.05])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision vs. Recall Curve - {algorithm_name}", fontsize=10)
    plt.legend(loc="lower left")
    plt.savefig(plot_path, dpi=600)
    print(f"We are also exporting a Precision-Recall plot for you here {plot_path}. This is a graphical "
          f"representation of the relationship between precision and recall scores in the withheld test data for "
          f"the best performing algorithm.")


### TODO: Why is this organized different from the same function in the continuous module? (pd vs np, train vs withheld, including both sets of IDs, split into functions, etc)
def export_prediction_data(out_dir, algorithm, y, x, ids, is_using_outer_cv, y_train=None, x_train=None, ids_train=None):
    """
    Save probability histograms and tables with accuracy metrics.

    Args:
        out_dir (pathlib.Path): Directory where results are saved.
        algorithm: Discrete prediction algorithm.
        y (pandas.DataFrame): Ground truth phenotypes for each training sample.
        x (pandas.DataFrame): Input data for each training sample.
        ids (pandas.Series): ids for participants corresponding to the datasets.
        y_train (optional, pandas.DataFrame): Ground truth phenotypes from the training dataset (Default: None).
        x_train (optional, pandas.DataFrame): Input data from the training dataset (Default: None).
        ids_train (optional, pandas.Series): ids for participants in the training dataset (Default: None).
    """

    if is_using_outer_cv:
        for fold, algo in enumerate(algorithm):
            if y_train is not None and x_train is not None and ids_train is not None:
                _export_prediction_data(out_dir, algo, y[fold], x[fold], ids[fold], y_train=y_train[fold], x_train=x_train[fold], ids_train=ids_train[fold], fold=fold)
            else:
                _export_prediction_data(out_dir, algo, y[fold], x[fold], ids[fold], y_train=None, x_train=None, ids_train=None, fold=fold)
    else:
        _export_prediction_data(out_dir, algorithm, y, x, ids, y_train=y_train, x_train=x_train, ids_train=ids_train)


def _export_prediction_data(out_dir, algorithm, y, x, ids, y_train=None, x_train=None, ids_train=None, fold=None):
    """
    Save probability histograms and tables with accuracy metrics.

    Args:
        out_dir (pathlib.Path): Directory where results are saved.
        y (pandas.DataFrame): Ground truth phenotypes for each training sample.
        x (pandas.DataFrame): Input data for each training sample.
        ids (pandas.Series): ids for participants corresponding to the datasets.
        y_train (optional, pandas.DataFrame): Ground truth phenotypes from the training dataset (Default: None).
        x_train (optional, pandas.DataFrame): Input data from the training dataset (Default: None).
        ids_train (optional, pandas.Series): ids for participants in the training dataset (Default: None).
        fold (int): If using outer cross-validation, fold number corresponding to current data/algorithm (Default: None).
    """

    suffix = f"_fold{fold+1}" if fold is not None else ""

    y_pred_prob = algorithm.predict_proba(x)
    if x_train is not None:
        y_train_pred = algorithm.predict_proba(x_train)

    if y_train is not None and y_train_pred is not None and ids_train is not None:
        export_prediction_tables(
            y_train,
            y_train_pred,
            ids_train,
            out_dir.joinpath(f"train_predictions{suffix}.txt"),
            dataset="training",
        )

    df_prediction = export_prediction_tables(
        y,
        y_pred_prob,
        ids,
        out_dir.joinpath(f"predictions{suffix}.txt"),
    )

    export_prob_hist(
        df_prediction,
        out_dir.joinpath(f"probabilities{suffix}"),
    )


def additional_sumstats(algorithm_name, y_test, x_test, algorithm, prefix, is_using_outer_cv):
    if is_using_outer_cv:
        for fold, y_test_fold in enumerate(y_test):
            y_pred_fold = algorithm[fold].predict_proba(x_test[fold])
            _additional_sumstats(algorithm_name, y_test_fold, y_pred_fold, prefix, fold=fold)
    else:
        y_pred = algorithm.predict_proba(x_test)
        _additional_sumstats(algorithm_name, y_test, y_pred, prefix)


def _additional_sumstats(algorithm_name, y_test, y_pred, prefix, fold=None):
    suffix = f"_fold{fold+1}" if fold is not None else ""
    log_table = pd.DataFrame(
        data=[[algorithm_name] + list(_calculate_accuracy_scores(y_test, y_pred))], 
        columns=["Algorithm", "AUC", "Accuracy", "Balanced_Accuracy", "Log_Loss", "Sensitivity", "Specificity", "PPV", "NPV"],
    )
    log_outfile = prefix.joinpath(f"performance_metrics{suffix}.txt")
    log_table.to_csv(log_outfile, index=False, sep="\t")


def calculate_accuracy_scores(x, y, algorithm):
    """
    Apply discrete prediction model and calculate accuracy metrics.

    Args:
        x (pandas.DataFrame): Model input features.
        y (pandas.DataFrame): Reported output features.
        algorithm: Contonuous prediction algorithm.

    :return: accuracy_metrics *(list)*: \n
        Accuracy metrics used for the discrete prediction module.
    """

    y_pred_prob = algorithm.predict_proba(x)
    return _calculate_accuracy_scores(y, y_pred_prob)


def _calculate_accuracy_scores(y, y_pred_prob):
    """
    Calculate accuracy metrics for the chosen discrete prediction model.

    Args:
        y (pandas.DataFrame): Reported output features.
        y_pred_prob (pandas.DataFrame): Predicted case probabilities.
    
    :return: rocauc *(float)*: \n
        ROC AUC value.
    :return: acc *(float)*: \n
        Accuracy value.
    :return: balacc *(float)*: \n
        Balanced accuracy value.
    :return: ll *(float)*: \n
        Log loss value.
    :return: sens *(float)*: \n
        Sensitivity value.
    :return: spec *(float)*: \n
        Specificity value.
    :return: ppv *(float)*: \n
        Positive predictive value.
    :return: npv *(float)*: \n
        Negative predictive value.
    """

    y_pred = np.argmax(y_pred_prob, axis=1)
    y_pred_prob = y_pred_prob[:,1]

    rocauc = metrics.roc_auc_score(y, y_pred_prob)
    acc = metrics.accuracy_score(y, y_pred) * 100
    balacc = metrics.balanced_accuracy_score(y, y_pred) * 100
    ll = metrics.log_loss(y, y_pred_prob)
    
    cm = metrics.confusion_matrix(y, y_pred)
    tn = cm[0][0]
    fn = cm[1][0]
    tp = cm[1][1]
    fp = cm[0][1]
    sens = (tp / (tp + fn) if (tp + fn) > 0 else 0)
    spec = (tn / (tn + fp) if (tn + fp) > 0 else 0)
    ppv  = (tp / (tp + fp) if (tp + fp) > 0 else 0)
    npv  = (tn / (tn + fn) if (tn + fn) > 0 else 0)

    return rocauc, acc, balacc, ll, sens, spec, ppv, npv


def export_prediction_tables(y, y_pred, ids, output_path, dataset="withheld test"):
    """
    Generate and save tables with prediction probabilities and predicted classes for each sample.

    Args:
        y (pandas.DataFrame): Ground truth phenotypes.
        y_pred (pandas.DataFrame): Predicted phenotypes.
        ids (pandas.Series): ids for participants corresponding to the datasets.
        output_path (pathlib.Path): Where to save output files.
        dataset (str): Indicator of whether analyzing training, tuning, or testing data.

    :return: df_prediction *(pandas.DataFrame)*: \n
        Table of reported and predicted phenotypes.
    """

    y_pred = pd.DataFrame(y_pred, dtype=float)
    df_predicted_cases = y_pred.idxmax(axis=1)
    case_probs = pd.DataFrame(y_pred.iloc[:,1])
    ids = pd.DataFrame(ids)
    df_prediction = pd.concat(
        [
            ids.reset_index(drop=True),
            y.reset_index(drop=True),
            case_probs.reset_index(drop=True),
            df_predicted_cases.reset_index(drop=True),
        ],
        axis=1,
        ignore_index=True,
    )

    df_prediction.columns = ['ID', "CASE_REPORTED", "CASE_PROBABILITY", "CASE_PREDICTED"]
    df_prediction.to_csv(output_path, index=False, sep="\t")

    print("")
    print(f"Preview of the exported predictions for the {dataset} data that has been exported as {output_path}.")
    print("")
    print("#" * 70)
    print(df_prediction.head())
    print("#" * 70)

    return df_prediction


def export_prob_hist(df_plot, plot_prefix):
    """
    Save probability histograms for each class.

    Args:
        df_plot (pandas.DataFrame): Table of predicted phenotypes.
        plot_prefix (pathlib.Path): Prefix for output files.
    """

    # Using the withheld sample data
    df_plot[f'Probability (%)'] = (df_plot[f'CASE_PROBABILITY'] * 100).round(decimals=0)
    df_plot['Reported Status'] = df_plot['CASE_REPORTED']
    df_plot['Predicted Status'] = df_plot['CASE_PREDICTED']

    # Start plotting
    plt.figure()
    sns.histplot(
        data=df_plot,
        x=f"Probability (%)",
        hue="Predicted Status",
        kde=True,
        alpha=0.2,
        multiple='dodge',
    )
    path = f"{plot_prefix}.png"
    plt.savefig(path, dpi=300)
    plt.clf()
    print(f"We are also exporting probability density plots to the file {path} this is a plot of the probability "
          f"distributions for each case, stratified by case status in the withheld test samples.")
