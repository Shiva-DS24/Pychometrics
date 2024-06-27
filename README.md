# Pychometrics Python Library
## _Item Analysis Simplified_





Pychometrics is a simplified item analysis python library that analyses assessment and questions (items). It gives straight forward interpretation based on the calculated parameters.


## Features

- Read csv file containing students scores
- Caclulates measures of central tendency of the assessment
- Aslo computes coefficient of internal consistency (Cronbach Alpha) to assess the consistency and reliability
- Item analysis for each question through Facility Index and Discrimination Index
- Item level feedback based on these parameters

## Dependencies
List of dependencies required to use your package:
- pandas~=2.2.2
- setuptools~=68.2.0
- reportlab~=4.2.0
- numpy~=1.26.4
- scipy~=1.13.1

## Online Demo
A minimal working online demo is available <a href= "https://apps.odc.edu.om/pychometrics/">here</a>

## Installation
Pychometrics is a python library and requires latest python library (tested in python 3.11).

Install the library with the dependencies through github or PyPI.

```sh
pip install -U git+https://github.com/Shiva-DS24/pychometrics.git@main

# or from pyPI: pip install pychometrics==0.1.4
# from jupyter: !pip install pychometrics==0.1.4

```
After installation, prepare a score data file as per the template provided [here](https://github.com/Shiva-DS24/pychometrics/blob/main/data.csv). Please note that this is similar to Moodle quiz export responses file and therefore that file can be used as it is. If assessments are conducted outside any Learning Management System, please prepare a csv file having column header Question No or Name with '/Max Marks'. Please refer the sample data file provided [here](https://github.com/Shiva-DS24/pychometrics/blob/main/data.csv).

Import the library
```sh
from pychometrics import AssessmentAnalysis, generate_pdf_report
```
Then read the data file and run the analysis
```sh
# read data file
analysis = AssessmentAnalysis('data.csv')
# run the analysis
analysis.run_analysis()
```

More functions: 

1.	If the data file is in the working directory and named as ‘data.csv’, data can be read using: `analysis = AssessmentAnalysis('data.csv')`. once read the message will indicate the number of students and items (questions) identified along with the missing values if any. 
2.	To get information about the loaded data at any time, we can run `analysis.info()`.
3.	The descriptive statistics of the entire test can be obtained by running `analysis.calc_desc()`.
4.	The skewness and kurtosis can be calculating by running `analysis.calc_skew()` and `analysis.calc_kurt()` respectively.
5.	The cronbach’s alpha can be obtained by running `analysis. calc_alpha()`.
6.	The Standard Error of Measurement (SEM) can be calculated using `analysis.sem()`.
7.	Calculation of FIs and DIs for each item can be obtained using `analysis.calc_fi()` and `analysis.calc_di()` respectively. Further to this `analysis.indices_export()` export the data as a csv file in the working directory. 
8.	A comprehensive pdf report can be generated using `analysis.export_report()` command.
9.	To obtain help about this library: `analysis.help()`, this will display available functions and their syntax.

    The generated pdf file will be available in the working directory.





## Contributing
If you would like to contribute to this project, you can fork the repository and send us a pull request. We welcome contributions!
## License
This project is licensed under the MIT License. See the LICENSE file for more details.

## Support
If you have any questions, issues, or suggestions regarding this package, feel free to create issue [here](https://github.com/Shiva-DS24/pychometrics/issues).


