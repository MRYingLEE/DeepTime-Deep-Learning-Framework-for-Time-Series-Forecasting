# DeepTime-Deep-Learning-Framework-for-Time-Series-Forecasting
This is an INTERACTIVE deep learning framework for time series forecasting. It uses Tensorflow 2+tf.karas.

# Why?

I have done a few time series forecasting projects. I hate I have to start from scratch every time, which always take a few hours. I want to focus my problem instead of typing and copying. And I want a framework can interactively assist me to go through the whole procedure smoothly. So that I can start quickly and end successfully. 

I couldn't find a good framework to do so. So I decided to develop one for myself. And share with others.

# Who?

This framework targets data scientists who wants to quickly start a time series forecasting model.

I am trying to divide the whole workflow as the different steps and for different roles:

## Data Explorary Analysis and Feature Engineering.<br>
  This step is for a data scientist who knows the domain data well.
  
## Model Design and Pre-defined models maintenance.<br>
  This step is for a data scientist who understands neural networks well.
  BTW, an idea of models in file format is too difficult to be implemented due to possible subclassed models. Also, it seems data inpendent neural networks are not reasonable. So we can only have models in code instead of an external source. And maybe pre-defined models can only be used as a reference only.
  
## Training and Hyperparameters Tuning.<br>
  This step is for a data scientist who understand hyperparameter tuning well.
  
## Forecasting/ Inferencing or Predicting. <br>
  This step is for an end users to use.
  
# What?

This is an INTERACTIVE deep learning framework for time series forecasting. It uses Tensorflow 2+tf.karas.
So, no pure statistics model was covered. Technically I use deep learning model, maybe along with some statistics model as part.
Time series (TS) data is focused. Technically 2 kinds of time-series data are supported.

Logically, time-series data has the 3D shape of (n_ series, n_timestamp, n_features).

Usually, time-series data can be categorized as:
## Tabular Data Source
Technically, Pandas (2D) with chunks (series, the 3rd D ).

Pros: Intuitive, suitable to complicated situation

Cons: Not efficient

## (Anonymous) 3D Array
Technically, NumPy array of (N_ series, n_timestamp, n_features).

Pros: Efficient and Neutral to NN

Cons: Not intuitive, suitable to simple situation

These 2 kinds of data can be converted to each other easily.

## Different Usages for 2 kinds of TS data 

Some AutoML tools support tabular data source, some support 3d array with shuffle=False. 
Some neurual network scripts support tabular one, somes do 3d one.

## Some related questions:
Is this an autoML platform? NO!<br>
Are there any built-in models? Yes, but only for reference.<br>
Are there hyperparamerters tuning built in? Maybe, but not yet.<br>

# Which?
Is this a whole pipeline framework? No, I won't cover data preprocessing, which can be covered by Time-series-Preprocessing-Studio-in-Jupyter (https://github.com/MRYingLEE/Time-series-Preprocessing-Studio-in-Jupyter ）.

# Technology
1. Tensorflow 2+tf.karas as deep learning framework. 

2. tf.data.Dataset will be used as basic data format. 

# Progress

## Interactive Features Builder

For a demo Pandas dataframe:

>Column| Description| Feature Type | Data Type
>------------|--------------------|----------------------|-----------------
>Age | Age in years | Numerical | integer
>Sex | (1 = male; 0 = female) | Categorical | integer
>CP | Chest pain type (0, 1, 2, 3, 4) | Categorical | integer
>Trestbpd | Resting blood pressure (in mm Hg on admission to the hospital) | Numerical | integer
>Chol | Serum cholestoral in mg/dl | Numerical | integer
>FBS | (fasting blood sugar > 120 mg/dl) (1 = true; 0 = false) | Categorical | integer
>RestECG | Resting electrocardiographic results (0, 1, 2) | Categorical | integer
>Thalach | Maximum heart rate achieved | Numerical | integer
>Exang | Exercise induced angina (1 = yes; 0 = no) | Categorical | integer
>Oldpeak | ST depression induced by exercise relative to rest | Numerical | float
>Slope | The slope of the peak exercise ST segment | Numerical | integer
>CA | Number of major vessels (0-3) colored by flourosopy | Numerical | integer
>Thal | 3 = normal; 6 = fixed defect; 7 = reversable defect | Categorical | string
>Target | Diagnosis of heart disease (1 = true; 0 = false) | Classification | integer
>is_male | Whether a person is male (true or false) | Numerical | bool

We may INTERACTIVELY build the features base on the dataframe.

Inline-style: 
![builder](https://github.com/MRYingLEE/DeepTime-Deep-Learning-Framework-for-Time-Series-Forecasting/blob/master/Features-Builder/builder.JPG "builder")

```
["input_features.append(numeric_column('age',normalizer_fn=lambda by_train: (tf.cast(by_train,tf.float32) -29.0)/(77.0-29.0)))",
 "input_features.append(numeric_column('trestbps',normalizer_fn=lambda by_train: (tf.cast(by_train,tf.float32) -94.0)/(200.0-94.0)))",
 "input_features.append(numeric_column('chol',normalizer_fn=lambda by_train: (tf.cast(by_train,tf.float32) -131.0)/(564.0-131.0)))",
 "input_features.append(numeric_column('thalach',normalizer_fn=lambda by_train: (tf.cast(by_train,tf.float32) -71.0)/(202.0-71.0)))",
 "input_features.append(numeric_column('oldpeak',normalizer_fn=lambda by_train: (by_train -0.0)/(6.2-0.0)))",
 "input_features.append(categorical_identitys('slope',[3,2,1]))",
 "input_features.append(categorical_identitys('ca',[0,3,2,1]))",
 "label_features.append(categorical_identitys('target',[0,1]))",
 "input_features.append(numeric_column('is_male'))"]
```
