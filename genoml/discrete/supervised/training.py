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

import genoml.discrete.utils as discrete_utils
import sys
from genoml import utils
from genoml.models import get_candidate_algorithms
from pathlib import Path
from sklearn import model_selection


class Train:
    @utils.DescriptionLoader.function_description("info", cmd="Discrete Supervised Training")
    def __init__(self, prefix, metric_max, train_split):
        utils.DescriptionLoader.print(
            "training/info",
            python_version=sys.version,
            prefix=prefix,
            metric_max=metric_max,
        )

        ### TODO: Add condition for if nothing is there, in which case they have not munged
        if Path(prefix).joinpath("Munge").joinpath(f"train_dataset.h5").exists():
            df_train = utils.read_munged_data(Path(prefix).joinpath("Munge").joinpath(f"train_dataset.h5"))
            x_train, x_valid, y_train, y_valid = utils.train_valid_split(df_train, train_split)
        elif Path(prefix).joinpath("Munge").joinpath(f"train_dataset_fold1.h5").exists():
            x_train = []
            x_valid = []
            y_train = []
            y_valid = []
            train_datasets = [f for f in Path(prefix).joinpath("Munge").iterdir() if f.is_file() and f.name.startswith("train_dataset")]
            for train_dataset in train_datasets:
                df_train = utils.read_munged_data(train_dataset)
                x_train_fold, x_valid_fold, y_train_fold, y_valid_fold = utils.train_valid_split(df_train, train_split)
                x_train.append(x_train_fold)
                x_valid.append(x_valid_fold)
                y_train.append(y_train_fold)
                y_valid.append(y_valid_fold)

        candidate_algorithms = get_candidate_algorithms("discrete_supervised")

        self._column_names = [
            "Algorithm",
            "Runtime_Seconds",
            "AUC",
            "Accuracy",
            "Balanced_Accuracy",
            "Log_Loss",
            "Sensitivity",
            "Specificity",
            "PPV",
            "NPV",
        ]
        self._run_prefix = Path(prefix).joinpath("Train")
        if not self._run_prefix.is_dir():
            self._run_prefix.mkdir()
        if isinstance(x_train, list):
            self._x_train = [x_train_fold.drop(columns=['ID']) for x_train_fold in x_train]
            self._x_valid = [x_valid_fold.drop(columns=['ID']) for x_valid_fold in x_valid]
            self._y_train = [y_train_fold.drop(columns=['ID']) for y_train_fold in y_train]
            self._y_valid = [y_valid_fold.drop(columns=['ID']) for y_valid_fold in y_valid]
            self._ids_train = [x_train_fold.ID for x_train_fold in x_train]
            self._ids_valid = [x_valid_fold.ID for x_valid_fold in x_valid]
        else:
            self._x_train = x_train.drop(columns=['ID'])
            self._x_valid = x_valid.drop(columns=['ID'])
            self._y_train = y_train
            self._y_valid = y_valid
            self._ids_train = x_train.ID
            self._ids_valid = x_valid.ID
        self._algorithms = {algorithm.__class__.__name__: algorithm for algorithm in candidate_algorithms}
        self._metric_max = metric_max
        self._best_algorithm = None
        self._best_algorithm_name = None
        self._log_table = []


    def compete(self):
        """ Compete the algorithms. """
        self._log_table, self._algorithms = utils.fit_algorithms(
            self._run_prefix,
            self._algorithms,
            self._x_train,
            self._y_train,
            self._x_valid,
            self._y_valid,
            self._column_names,
            discrete_utils.calculate_accuracy_scores,
        )


    def select_best_algorithm(self):
        """ Determine the best-performing algorithm. """
        # Drop those that have an accuracy less than 50%, balanced accuracy less than 50%, delta between sensitivity
        # and specificity greater than 0.85, sensitivity equal to 0 or 1, or specificity equal to 0 or 1.
        filtered_table = self._log_table[
            (self._log_table["AUC"] > 0.5)
            & (self._log_table["Balanced_Accuracy"] > 50)
            & (self._log_table["Sensitivity"].sub(self._log_table["Specificity"], axis=0).abs() < 0.85)
            & (self._log_table["Sensitivity"] != 0.0)
            & (self._log_table["Sensitivity"] != 1.0)
            & (self._log_table["Specificity"] != 0.0)
            & (self._log_table["Specificity"] != 1.0)
        ]

        # If for some reason ALL the algorithms are overfit...
        if filtered_table.empty:
            print('It seems as though all the algorithms are over-fit in some way or another... We will report the best algorithm based on your chosen metric instead and use that moving forward.')
            filtered_table = self._log_table

        self._best_algorithm, self._best_algorithm_name = utils.select_best_algorithm(
            filtered_table, 
            self._metric_max, 
            self._algorithms,
        )
        with open(self._run_prefix.parent.joinpath("algorithm.txt"), "w") as file:
            file.write(self._best_algorithm_name)


    def export_model(self):
        """ Save best-performing algorithm """
        utils.export_model(
            self._run_prefix.parent, 
            self._best_algorithm,
        )


    def plot_results(self):
        """ Plot results from best-performing algorithm. """
        discrete_utils.plot_results(
            self._run_prefix,
            [y_valid.values for y_valid in self._y_valid] if isinstance(self._y_valid, list) else self._y_valid.values,
            [x_valid.values for x_valid in self._x_valid] if isinstance(self._x_valid, list) else self._x_valid.values,
            self._best_algorithm,
        )


    def export_prediction_data(self):
        """ Save results from best-performing algorithm. """
        discrete_utils.export_prediction_data(
            self._run_prefix,
            self._best_algorithm,
            self._y_valid,
            self._x_valid,
            self._ids_valid,
            y_train = self._y_train,
            x_train = self._x_train,
            ids_train = self._ids_train,
        )
