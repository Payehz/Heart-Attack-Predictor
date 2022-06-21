![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![SciPy](https://img.shields.io/badge/SciPy-%230C55A5.svg?style=for-the-badge&logo=scipy&logoColor=%white)
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)


# Heart Attack Predictor

The objective of this model to predict whether a person has a chance to get a heart disease.
Heart attack has been a common occurence among humans. So a precatious measure is highly
recommended to avoid any unwanted incidents.

## Results

![Accuracy](static/classification_report.png)

![Confusion Matrix](static/confusion_matrix.png)

The model scored an accuracy of 78%. The best pipeline for this model is to use 
MinMaxScaler and Logistic Regression. 
The model can be further improven if we have more datas.

## Application

An app is created for users to input their information to predict heart attack possibilities
using the model developed.

![First half](static/app_ss_1.PNG)

![Second half](static/app_ss_2.PNG)

## Credits

The data is downloaded from
[Kaggle](https://www.kaggle.com/datasets/rashikrahmanpritom/heart-attack-analysis-prediction-dataset)











