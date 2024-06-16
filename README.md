# Pychometrics Python Library
## _Item Analysis Simplified_



[![Build Status](https://travis-ci.org/joemccann/dillinger.svg?branch=master)](https://travis-ci.org/joemccann/dillinger)

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


## Installation
Pychometrics is a python library and requires latest python library (tested in python 3.11).

Install the library with the dependencies through github or PyPI.

```sh
pip install git+https://github.com/jkbr/httpie.git#egg=httpie
```
After installation, prepare a score data file as per the template provided here:. Please note that this is similar to Moodle quiz export responses file and therefore that file can be used as it is. 

Import the library
```sh
from analysis import AssessmentAnalysis
from report import generate_pdf_report
```
Then read the data file and run the analysis
```sh
# read data file
analysis = AssessmentAnalysis('data.csv')
# run the analysis
analysis.run_analysis()
```
The generated pdf file will be available in the working directory.

## Contributing
If you would like to contribute to this project, you can fork the repository and send us a pull request. We welcome contributions!
## License
This project is licensed under the MIT License. See the LICENSE file for more details.

## Support
If you have any questions, issues, or suggestions regarding this package, feel free to contact us at your.email@example.com.


