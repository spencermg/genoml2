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

import joblib
import numpy as np
import pandas as pd
from statsmodels.stats import outliers_influence


# Define the VIF class to be used in munging
class VIF:
    def __init__(self, iterations, vif_threshold, df, chunk_size, run_prefix):
        self.iterations = iterations
        self.vif_threshold = vif_threshold
        self.df = df
        self.chunk_size = chunk_size
        self.run_prefix = run_prefix
        self.df_cleaned = None
        self.df_list = None
        self.df_concat = None

    def check_df(self):
        """
        Strips the dataframe of missing values and non-numerical information.
        """

        print("Stripping erroneous space, dropping non-numeric columns...")
        self.df.columns = self.df.columns.str.strip()

        print("Drop any rows where at least one element is missing...")
        self.df.replace([np.inf, -np.inf], np.nan, inplace=True)
        self.df.dropna(how='any', inplace=True)

        ### TODO: Should we sample more than this? And/or sample within each iteration?
        print("Sampling 100 rows at random to reduce memory overhead...")
        self.df_cleaned = self.df.sample(n=100, random_state=42).copy().reset_index()

        print("Dropping columns that are not features...")
        self.df_cleaned.drop(columns=["index", "PHENO", "ID"], inplace=True)
        self.df_cleaned = self.df_cleaned.astype(float)
        print("Dropped!")


    def randomize_chunks(self):
        """
        Radomizes features in the dataframe and splits them into chunks of 
        specified size to be used in VIF calculations.
        """

        chunk_size = self.chunk_size
        df_cleaned = self.df_cleaned

        print("Shuffling columns...")
        df_cleaned = df_cleaned.sample(frac=1, axis=1, random_state=42)
        print("Shuffled!")

        print("Generating chunked, randomized dataframes...")
        self.df_list = []
        for i in range((df_cleaned.shape[1] + chunk_size - 1) // chunk_size):
            self.df_list.append(df_cleaned.iloc[:, i*chunk_size : (i+1)*chunk_size].copy())

        print(f"The number of dataframes you have moving forward is {len(self.df_list)}")
        print("Complete!")


    def calculate_vif(self):
        """
        Removes any features from each chunked dataframe with VIF greater than
        the specified threshold to combat multicolinearity between the variables. 
        """

        print(f"Dropping columns with a VIF threshold greater than {self.vif_threshold}")
        for df in self.df_list:
            while True:
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
        self.df_concat = pd.concat(self.df_list, axis=1)
        self.df_concat = self.df_concat.loc[:, ~self.df_concat.columns.duplicated()]
        print("Full VIF-filtered dataframe generated!")

    def vif_calculations(self):
        self.check_df()

        ### TODO: each iteration of VIF does the same thing... what is this supposed to do?
        for iteration in range(self.iterations):
            print(f"\n\nIteration {iteration + 1}\n\n")
            self.randomize_chunks()
            self.calculate_vif()

        # Return the original dataframe with the features to keep 
        return self.df[["PHENO", "ID"] + self.df_concat.columns.values.tolist()]
