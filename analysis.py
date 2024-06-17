# pychometrics/analysis.py
import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis
import re
import os


class AssessmentAnalysis:
    def __init__(self, data_file):
        self.stats = None
        self.df = pd.read_csv(data_file)
        self.marks_df = None
        self.total_max_marks = 0
        self.mean = 0
        self.median = 0
        self.std_dev = 0
        self.skewness = 0
        self.kurt = 0
        self.alpha = 0
        self.sem = 0
        self.item_analysis = None
        self.n_students =0
        self.n_items=0
        self.help_text = """
                        pychometrics Library Help
                        ---------------------
                        Available Commands and Syntax:              


                        1. AssessmentAnalysis(csv_file name)
                           - Description: This function provides input data for item analysis. 
                                Data must be prepared using the template. Once read successfully it creates an object for further processing.
                           - Syntax: AssessmentAnalysis('data.csv')

                        2. info()
                           - Description: To get information about the loaded data at any time. 
                                It displays identified number of students and items. 
                           - Syntax: analysis.info()

                        3. calc_desc()
                           - Description: The descriptive statistics of the entire test can be obtained by running.
                           - Syntax: analysis.calc_desc()

                        4. calc_skew()
                           - Description: Calculates the skewness of the dataset.
                           - Syntax: analysis.skew()

                        5. calc_kurt()
                           - Description: Calculates the kurtosis of the dataset.
                           - Syntax: analysis.kurt()

                        6. calc_fi()
                           - Description: Calculates the facility index for all questions.
                           - Syntax: analysis.calc_fi()

                        7. calc_di()
                           - Description: Calculates the discrimination index for all questions.
                           - Syntax: analysis.calc_di()
                        
                        8. export_indices()
                           - Description: Calculates both DI and FI and export the results as csv file.
                           - Syntax: analysis.export_indices()

                        9. calc_alpha()
                           - Description: Calculates Cronbach's Alpha for the dataset.
                           - Syntax: analysis.calc_alpha()
                        10. calc_sem()
                           - Description: Calculates SEM for the dataset.
                           - Syntax: analysis.calc_sem()                  

                        11. export_report()
                            - Description: A comprehensive pdf report can be generated .
                            - Syntax: analysis.export_report()

                        """

    def filter_columns(self):
        self.df = self.df[self.df['Surname'] != 'Overall average']
        pattern = r'Q\..*/'
        filtered_columns = self.df.filter(regex=pattern).columns
        self.marks_df = self.df.loc[:, filtered_columns].copy()

    def info(self):
        self.filter_columns()
        self.n_students = len(self.marks_df.axes[0])
        self.n_items = len(self.marks_df.axes[1])
        print("No of students", self.n_students)
        print("No of items", self.n_items)

    def extract_max_marks(self):
        max_marks = {col: float(re.search(r'/([0-9.]+)', col).group(1)) for col in self.marks_df.columns}
        self.total_max_marks = sum(max_marks.values())

    def calculate_totals(self):
        self.marks_df.loc[:, 'Total'] = self.marks_df.sum(axis=1, numeric_only=True)
        self.marks_df.loc[:, 'Total_Percentage'] = (self.marks_df['Total'] / self.total_max_marks) * 100

    def calculate_statistics(self):
        self.mean = round(self.marks_df['Total_Percentage'].mean(), 2)
        self.median = round(self.marks_df['Total_Percentage'].median(), 2)
        self.std_dev = round(self.marks_df['Total_Percentage'].std(), 2)
        self.skewness = round(skew(self.marks_df['Total_Percentage']), 2)
        self.kurt = round(kurtosis(self.marks_df['Total_Percentage']), 2)

    def calc_desc(self):
        self.filter_columns()
        self.extract_max_marks()
        self.calculate_totals()
        self.calculate_statistics()
        self.mean = round(self.marks_df['Total_Percentage'].mean(), 2)
        self.median = round(self.marks_df['Total_Percentage'].median(), 2)
        self.std_dev = round(self.marks_df['Total_Percentage'].std(), 2)
        print("Mean of total score (%)", self.mean)
        print("Median of total score (%)", self.median)
        print("Std dev. of total score (%)", self.std_dev)

    def calc_skew(self):
        self.filter_columns()
        self.extract_max_marks()
        self.calculate_totals()
        skewness = round(skew(self.marks_df['Total_Percentage']), 2)
        print("Skewness of total score", skewness)

    def calc_kurt(self):
        self.filter_columns()
        self.extract_max_marks()
        self.calculate_totals()
        kurt_value = round(kurtosis(self.marks_df['Total_Percentage']), 2)
        print("Kurtosis of total score", kurt_value)

    def calc_alpha(self):
        self.filter_columns()
        self.extract_max_marks()
        self.calculate_totals()
        marks = self.marks_df.drop(columns=['Total', 'Total_Percentage'])
        k = marks.shape[1]
        item_variances = np.var(marks, axis=0, ddof=1)
        total_scores = marks.sum(axis=1)
        total_variance = np.var(total_scores, ddof=1)
        self.alpha = round((k / (k - 1)) * (1 - (item_variances.sum() / total_variance)) * 100, 2)
        total_std_dev = np.sqrt(total_variance)
        self.sem = round(total_std_dev * np.sqrt(1 - self.alpha / 100), 2)

    def calc_sem(self):
        self.filter_columns()
        self.extract_max_marks()
        self.calculate_totals()
        marks = self.marks_df.drop(columns=['Total', 'Total_Percentage'])
        k = marks.shape[1]
        item_variances = np.var(marks, axis=0, ddof=1)
        total_scores = marks.sum(axis=1)
        total_variance = np.var(total_scores, ddof=1)
        self.alpha = round((k / (k - 1)) * (1 - (item_variances.sum() / total_variance)) * 100, 2)
        total_std_dev = np.sqrt(total_variance)
        self.sem = round(total_std_dev * np.sqrt(1 - self.alpha / 100), 2)
        print("Std Error of measurement of the test is", self.sem)
    def combine_stats(self):
        self.stats = {
            'mean': self.mean,
            'median': self.median,
            'std_dev': self.std_dev,
            'skewness': self.skewness,
            'kurt': self.kurt,
            'alpha': self.alpha,
            'sem': self.sem
        }

    def calc_indices(self):
        self.filter_columns()
        self.extract_max_marks()
        self.calculate_totals()
        facility_index = {}
        for column_name in self.marks_df.columns[:-2]:  # Excluding 'Total' and 'Total_Percentage'
            max_marks = float(column_name.split('/')[1].strip())
            mean_score = self.marks_df[column_name].mean()
            facility_idx = 100 * (mean_score / max_marks)
            facility_index[column_name] = facility_idx
        fi_df = pd.DataFrame.from_dict(facility_index, orient='index', columns=['Facility Index'])
        discrimination_indices = {}
        for column in self.marks_df.columns[:-2]:  # Excluding 'Total' and 'Total_Percentage'
            item_responses = self.marks_df[column].values
            total_score = self.marks_df.drop(columns=[column, 'Total', 'Total_Percentage']).sum(axis=1).values
            covariance = np.cov(item_responses, total_score, ddof=1)[0, 1]
            mark_variance = np.var(item_responses, ddof=1)
            total_score_variance = np.var(total_score, ddof=1)
            #print('mark_variance', mark_variance)
            if np.isnan(covariance) or np.isnan(mark_variance) or np.isnan(total_score_variance):
                discrimination_index = np.nan
            elif mark_variance == 0 or total_score_variance == 0:
                discrimination_index = np.nan
            else:
                discrimination_index = 100 * covariance / np.sqrt(mark_variance * total_score_variance)

            discrimination_indices[column] = discrimination_index

            # discrimination_index = 100 * covariance / np.sqrt(mark_variance * total_score_variance)
            # discrimination_indices[column] = discrimination_index
        di_df = pd.DataFrame.from_dict(discrimination_indices, orient='index', columns=['Discrimination Index'])
        item_analysis = pd.merge(fi_df, di_df, left_index=True, right_index=True)
        item_analysis = item_analysis.round(2)
        # Define the mapping dictionary
        facility_index_mapping = {
            (0, 40): 'Very hard',
            (40, 50): 'Hard',
            (50, 60): 'Appropriate',
            (60, 70): 'Fairly Easy',
            (70, 80): 'Easy',
            (80, float('inf')): 'Very Easy'
        }

        # Convert 'Facility Index' column to numeric (if it's not already)
        # item_analysis['Facility Index'] = pd.to_numeric(item_analysis['Facility Index'], errors='coerce')

        # Create a new column 'Facility Index Comment' based on the mapping
        item_analysis['Interpretation 1'] = pd.cut(item_analysis['Facility Index'],
                                                   bins=[0, 40, 50, 60, 70, 80, float('inf')],
                                                   labels=facility_index_mapping.values(),
                                                   right=False)

        # di mapping
        # Define the mapping dictionary

        discrimination_index_mapping = {
            (50, float('inf')): 'Very Good',
            (40, 50): 'Adequate',
            (20, 40): 'Weak',
            (-float('inf'), 20): 'Very Weak',
            'Negative': 'Probably Invalid'
        }

        # Function to apply the mapping
        def map_discrimination_index(value):
            for key, comment in discrimination_index_mapping.items():
                if isinstance(key, tuple):
                    if key[0] <= value < key[1]:
                        return comment
                elif key == 'Negative' and value < 0:
                    return comment
            return None

        # Apply the mapping function to create 'Discrimination Index Comment' column
        item_analysis['Interpretation 2'] = item_analysis['Discrimination Index'].apply(map_discrimination_index)

        item_analysis = item_analysis[
            ['Facility Index', 'Interpretation 1', 'Discrimination Index', 'Interpretation 2']]

        self.item_analysis = item_analysis

    def calc_fi(self):
        self.calc_indices()
        fi = self.item_analysis.iloc[:, 0]
        print("Facility Indices")
        print(fi)

    def calc_di(self):
        self.calc_indices()
        di = self.item_analysis.iloc[:, 2]
        print("Discrimination Indices")
        print(di)

    def export_indices(self):
        self.calc_indices()
        base_filename = "indices.csv"
        counter = 1
        filename = base_filename

        # Check if the file already exists
        while os.path.exists(filename):
            filename = f"indices{counter}.csv"
            counter += 1

        # Export DataFrame to CSV with the unique filename
        self.item_analysis.to_csv(filename, index=False)
        print(f"{filename} is created in the working directory.")

    def generate_report(self):
        from .report import generate_pdf_report
        generate_pdf_report(self.marks_df, self.total_max_marks, self.item_analysis, self.stats)

    def run_analysis(self):
        self.filter_columns()
        self.extract_max_marks()
        self.calculate_totals()
        self.calculate_statistics()
        self.calc_alpha()
        self.combine_stats()
        self.calc_indices()
        self.generate_report()

    def help(self):
        print(self.help_text)

    def __getattr__(self, attr):
        if attr == 'help':
            return self.help
        else:
            raise AttributeError(f"'{self.__class__.__name__}' there is no such function currently available: '{attr}'. "
                                 f"Use analysis.help() for available functions.")

# Example usage:
# analysis = AssessmentAnalysis('data.csv')
# analysis.info()

