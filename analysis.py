# pychometrics/analysis.py
import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis
import re


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

    def filter_columns(self):
        self.df = self.df[self.df['Surname'] != 'Overall average']
        pattern = r'Q\..*/'
        filtered_columns = self.df.filter(regex=pattern).columns
        self.marks_df = self.df.loc[:, filtered_columns].copy()

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

    def calculate_cronbach_alpha(self):
        marks = self.marks_df.drop(columns=['Total', 'Total_Percentage'])
        k = marks.shape[1]
        item_variances = np.var(marks, axis=0, ddof=1)
        total_scores = marks.sum(axis=1)
        total_variance = np.var(total_scores, ddof=1)
        self.alpha = round((k / (k - 1)) * (1 - (item_variances.sum() / total_variance)) * 100, 2)
        total_std_dev = np.sqrt(total_variance)
        self.sem = round(total_std_dev * np.sqrt(1 - self.alpha / 100), 2)

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
            print('mark_variance', mark_variance)
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

    def generate_report(self):
        from .report import generate_pdf_report
        generate_pdf_report(self.marks_df, self.total_max_marks, self.item_analysis, self.stats)

    def run_analysis(self):
        self.filter_columns()
        self.extract_max_marks()
        self.calculate_totals()
        self.calculate_statistics()
        self.calculate_cronbach_alpha()
        self.combine_stats()
        self.calc_indices()
        self.generate_report()

# Example usage:
# analysis = AssessmentAnalysis('data.csv')
# analysis.run_analysis()
