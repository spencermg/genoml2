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


from sklearn import discriminant_analysis, ensemble, linear_model, neighbors, neural_network, svm
import xgboost


CONTINUOUS_BASE_ESTIMATORS_ADABOOST = [
    linear_model.ElasticNet(),
    neural_network.MLPRegressor(),
    linear_model.SGDRegressor(),
    svm.SVR(gamma='auto'),
]
CONTINUOUS_BASE_ESTIMATORS_BAGGING = [
    linear_model.ElasticNet(),
    neighbors.KNeighborsRegressor(),
    neural_network.MLPRegressor(),
    linear_model.SGDRegressor(),
    svm.SVR(gamma='auto'),
]
DISCRETE_BASE_ESTIMATORS_ADABOOST = [
    linear_model.LogisticRegression(),
    neural_network.MLPClassifier(),
    linear_model.SGDClassifier(loss='modified_huber'),
    svm.SVC(probability=True, gamma='scale'),
]
DISCRETE_BASE_ESTIMATORS_BAGGING = [
    neighbors.KNeighborsClassifier(),
    discriminant_analysis.LinearDiscriminantAnalysis(),
    linear_model.LogisticRegression(),
    neural_network.MLPClassifier(),
    discriminant_analysis.QuadraticDiscriminantAnalysis(),
    linear_model.SGDClassifier(loss='modified_huber'),
    svm.SVC(probability=True, gamma='scale'),
]

CANDIDATE_ALGORITHMS = {
    "continuous_supervised": [
        ensemble.AdaBoostRegressor(estimator=estimator, random_state=3) for estimator in CONTINUOUS_BASE_ESTIMATORS_ADABOOST
    ] + [
        ensemble.BaggingRegressor(estimator=estimator, n_jobs=-1, random_state=3) for estimator in CONTINUOUS_BASE_ESTIMATORS_BAGGING
    ] + [
        ensemble.GradientBoostingRegressor(random_state=3),
        ensemble.RandomForestRegressor(random_state=3),
        xgboost.XGBRegressor(random_state=3),
    ],
    "discrete_supervised": [
        ensemble.AdaBoostClassifier(estimator=estimator, random_state=3) for estimator in DISCRETE_BASE_ESTIMATORS_ADABOOST
    ] + [
        ensemble.BaggingClassifier(estimator=estimator, n_jobs=-1, random_state=3) for estimator in DISCRETE_BASE_ESTIMATORS_BAGGING
    ] + [
        ensemble.GradientBoostingClassifier(random_state=3),
        ensemble.RandomForestClassifier(n_estimators=50, random_state=3),
        xgboost.XGBClassifier(random_state=3),
    ],
    "multiclass_supervised": [
        ensemble.AdaBoostClassifier(estimator=estimator, random_state=3) for estimator in DISCRETE_BASE_ESTIMATORS_ADABOOST
    ] + [
        ensemble.BaggingClassifier(estimator=estimator, n_jobs=-1, random_state=3) for estimator in DISCRETE_BASE_ESTIMATORS_BAGGING
    ] + [
        ensemble.GradientBoostingClassifier(random_state=3),
        ensemble.RandomForestClassifier(n_estimators=50, random_state=3),
        xgboost.XGBClassifier(objective="multi:softprob", random_state=3),
    ],
}


def get_candidate_algorithms(module_name):
    return CANDIDATE_ALGORITHMS.get(module_name, {})
