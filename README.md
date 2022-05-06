
# Project Title: Credit Risk Classification

We will use various techniques to train and evaluate models with imbalanced classes. You’ll use a dataset of historical lending activity from a peer-to-peer lending services company to build a model that can identify the creditworthiness of borrowers.


---

## Technologies:

The project uses python 3.7 with the following packages:

* [pandas](https://pandas.pydata.org/) - For providing data analysis and manipulation tool built on top of the Python programming language

* [numpy](https://numpy.org/doc/stable/) - It is a fundamental package for scientific computing in Python

* [pathlib](https://docs.python.org/3/library/pathlib.html) - Offers classes representing filesystem paths with semantics appropriate for different operating systems

* [warnings](https://docs.python.org/3/library/warnings.html) - Typically issued in situations where it is useful to alert the user of some condition in a program, where that condition (normally) doesn’t warrant raising an exception and terminating the program

* [sklearn](https://scikit-learn.org/stable/) - Machine learning package in Python

* [imblearn](https://imbalanced-learn.org/stable/) - open source, MIT-licensed library relying on scikit-learn (imported as sklearn) and provides tools when dealing with classification with imbalanced classes


---

## Installation Guide


Before running the application first install the following dependencies:

```python
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import confusion_matrix
from imblearn.metrics import classification_report_imbalanced

import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

from imblearn.over_sampling import RandomOverSampler
```

---

## Usage

To run the program, simply clone the repository, and go through the written steps in the *.ipynb file.

---

## Contributors

Jung Kim


---

## License

MIT License
