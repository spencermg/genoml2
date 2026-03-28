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

import genoml.continuous.utils as continuous_utils
import joblib
import sys
from genoml import utils
from pathlib import Path
from sklearn import metrics
from sklearn.model_selection import KFold


class Tune():
    @utils.DescriptionLoader.function_description("info", cmd="Continuous Supervised Tuning")
    def __init__(self, run_prefix, metric_tune, max_iter, cv_count):
        utils.DescriptionLoader.print(
            "tuning/info",
            python_version=sys.version,
            run_prefix=run_prefix,
            max_iter=max_iter,
            cv_count=cv_count,
        )

        ### TODO: Add condition for if nothing is there, in which case they have not munged
        if Path(run_prefix).joinpath("Munge").joinpath(f"train_dataset.h5").exists():
            df_tune = utils.read_munged_data(Path(run_prefix).joinpath("Munge").joinpath(f"train_dataset.h5"))
            model_path = Path(run_prefix).joinpath("model.joblib")
            self._y_tune = df_tune.PHENO
            self._ids_tune = df_tune.ID.values
            self._x_tune = df_tune.drop(columns=["PHENO", "ID"])
            self._algorithm = joblib.load(model_path)
            algorithm_name = self._algorithm.__class__.__name__
        elif Path(run_prefix).joinpath("Munge").joinpath(f"train_dataset_fold1.h5").exists():
            self._y_tune = []
            self._ids_tune = []
            self._x_tune = []
            self._algorithm = []
            train_datasets = [f for f in Path(run_prefix).joinpath("Munge").iterdir() if f.is_file() and f.name.startswith("train_dataset")]
            for fold, train_dataset in enumerate(train_datasets):
                df_tune = utils.read_munged_data(train_dataset)
                model_path = Path(run_prefix).joinpath(f"model_fold{fold+1}.joblib")
                self._y_tune.append(df_tune.PHENO)
                self._ids_tune.append(df_tune.ID.values)
                self._x_tune.append(df_tune.drop(columns=["PHENO", "ID"]))
                self._algorithm.append(joblib.load(model_path))
            algorithm_name = self._algorithm[0].__class__.__name__

        dict_hyperparams = utils.get_tuning_hyperparams("continuous")

        if metric_tune == "Explained_Variance":
            self._scoring_metric = metrics.make_scorer(metrics.explained_variance_score)
        elif metric_tune == "Mean_Squared_Error":
            self._scoring_metric = metrics.make_scorer(metrics.mean_squared_error)
        elif metric_tune == "Median_Absolute_Error":
            self._scoring_metric = metrics.make_scorer(metrics.median_absolute_error)
        elif metric_tune == "R-Squared_Error":
            self._scoring_metric = metrics.make_scorer(metrics.r2_score)
        
        self._run_prefix = Path(run_prefix).joinpath("Tune")
        if not self._run_prefix.is_dir():
            self._run_prefix.mkdir()
        self._max_iter = max_iter
        self._cv = KFold(n_splits=cv_count, shuffle=True, random_state=3)
        # self._cv_count = cv_count
            
        self._hyperparameters = dict_hyperparams[algorithm_name]
        self._cv_tuned = None
        self._cv_baseline = None
        self._cv_results = None
        self._algorithm_tuned = None
        self._tune_results = None
        self._y_predicted = None

        # Communicate to the user the best identified algorithm 
        print(f"From previous analyses in the training phase, we've determined that "
              f"the best algorithm for this application is {algorithm_name}... so "
              "let's tune it up and see what gains we can make!")


    def tune_model(self):
        """ Determine best-performing hyperparameters. """
        self._cv_results, self._algorithm_tuned = utils.tune_model(
            self._algorithm,
            self._x_tune,
            self._y_tune,
            self._hyperparameters,
            self._scoring_metric,
            self._max_iter,
            self._cv,
        )


    def report_tune(self):
        """ Save best-performing fine-tuning iterations. """
        utils.report_best_tuning(
            self._run_prefix, 
            self._cv_results, 
            10,
        )


    def summarize_tune(self):
        """ Report results for baseline and tuned models. """
        self._cv_baseline, self._cv_tuned = utils.summarize_tune(
            self._run_prefix,
            self._algorithm, 
            self._algorithm_tuned, 
            self._x_tune, 
            self._y_tune, 
            self._scoring_metric, 
            self._cv, 
        )
    

    def compare_performance(self):
        """ Compare tuned model with baseline model. """
        self._algorithm, _ = utils.compare_tuning_performance(
            self._run_prefix, 
            self._cv_tuned, 
            self._cv_baseline, 
            self._algorithm_tuned, 
            self._algorithm, 
            x = self._x_tune,
        )


    def export_prediction_data(self):
        """ Save results from best-performing algorithm. """
        continuous_utils.export_prediction_data(
            self._run_prefix,
            self._ids_tune,
            "tuning",
            self._algorithm,
            [y_tune.values for y_tune in self._y_tune] if isinstance(self._y_tune, list) else self._y_tune.values,
            [x_tune.values for x_tune in self._x_tune] if isinstance(self._x_tune, list) else self._x_tune.values,
        )
        