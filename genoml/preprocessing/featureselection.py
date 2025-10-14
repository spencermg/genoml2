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

import pandas as pd
from sklearn import ensemble, feature_selection
import joblib
import pandas as pd
from statsmodels.stats import outliers_influence


### TODO: Migrate print statements to use utils.DescriptionLoader.print
class FeatureSelection:
    def __init__(self, prefix, df, data_type, n_est, vif_iter, vif_threshold, chunk_size):
        df.dropna(axis=1, inplace=True)
        self.prefix = prefix
        self.n_est = n_est
        self.data_type = data_type
        self.df = df
        self.y = self.df.PHENO
        self.ids = self.df.ID
        self.x = self.df.drop(columns=['PHENO','ID'])
        self.vif_iter = vif_iter
        self.vif_threshold = vif_threshold
        self.chunk_size = chunk_size
        self.df_list = None

    def select_features(self):
        if self.n_est > 0:
            self.extra_trees()
        if self.vif_iter > 0:
            self.vif()
        return self.df

    def extra_trees(self):
        # Fit extraTrees model.
        print(f"Using {self.n_est} extraTrees estimators...")
        if self.data_type == "d":
            clf = ensemble.ExtraTreesClassifier(n_estimators=self.n_est)
        if self.data_type == "c":
            clf = ensemble.ExtraTreesRegressor(n_estimators=self.n_est)
        clf.fit(self.x, self.y)
        
        # Drop the features below threshold.
        model = feature_selection.SelectFromModel(clf, prefit=True)
        df_feature_scores = pd.DataFrame(
            zip(self.x.columns, clf.feature_importances_),
            columns=["Feature_Name", "Score"]
        ).sort_values(by=['Score'], ascending=False)
        df_feature_scores.to_csv(
            self.prefix.joinpath("approx_feature_importance.txt"), 
            index=False, 
            sep="\t",
        )

        ### TODO: Should we define a threshold somewhere? Currently only keeping features with importance > mean
        # Filter to only include important features.
        x_reduced = self.x.iloc[:, model.get_support()]
        self.df = pd.concat([
            self.ids.reset_index(drop=True), 
            self.y.reset_index(drop=True), 
            x_reduced.reset_index(drop=True),
        ], axis = 1, ignore_index=False)

        ### TODO: This should be after VIF
        # Save features
        with open(self.prefix.joinpath("list_features.txt"), 'w') as f:
            for feature in self.df.columns.values.tolist():
                f.write(feature + "\n")

        print(f"An updated list of the {self.df.shape[1] - 2} features, plus ID and PHENO, in your munged dataForML.h5 file can be found at {self.prefix.joinpath("list_features.txt")}")

    def vif(self):
        self.df.columns = self.df.columns.str.strip()

        ### TODO: Should we sample more than this? And/or sample within each iteration?
        print("Sampling 100 rows at random to reduce memory overhead...")
        df_cleaned = self.df.sample(n=100, random_state=42).copy().reset_index()

        print("Dropping columns that are not features...")
        df_cleaned.drop(columns=["index", "PHENO", "ID"], inplace=True)
        df_cleaned = df_cleaned.astype(float)

        ### TODO: each iteration of VIF does the same thing... what is this supposed to do?
        for iteration in range(self.vif_iter):
            print(f"\n\nIteration {iteration + 1}\n\n")
            print("Shuffling columns...")
            df_cleaned = df_cleaned.sample(frac=1, axis=1, random_state=42)

            print("Generating chunked, randomized dataframes...")
            self.df_list = []
            for i in range((df_cleaned.shape[1] + self.chunk_size - 1) // self.chunk_size):
                self.df_list.append(df_cleaned.iloc[:, i*self.chunk_size : (i+1)*self.chunk_size].copy())
            
            print(f"Dropping columns with a VIF threshold greater than {self.vif_threshold}")
            for df in self.df_list:
                while True:
                    if df.shape[1] == 1:
                        break

                    # Calculate VIF for all columns
                    vif_vals = joblib.Parallel(n_jobs=5)(
                        joblib.delayed(outliers_influence.variance_inflation_factor)(df.values, i)
                        for i in range(df.shape[1])
                    )

                    max_vif = max(vif_vals)
                    if max_vif <= self.vif_threshold:
                        break

                    # Drop the column with the highest VIF
                    max_idx = vif_vals.index(max_vif)
                    col_to_drop = df.columns[max_idx]
                    print(f'Dropping "{col_to_drop}" with VIF = {max_vif:.2f}')
                    df.drop(columns=[col_to_drop], inplace=True)

            print("\nVIF calculation on all chunks complete! \n")
            df_cleaned = pd.concat(self.df_list, axis=1)
            df_cleaned = df_cleaned.loc[:, ~df_cleaned.columns.duplicated()]

        # Use the original dataframe with the features to keep 
        self.df = self.df[["PHENO", "ID"] + df_cleaned.columns.values.tolist()]
               