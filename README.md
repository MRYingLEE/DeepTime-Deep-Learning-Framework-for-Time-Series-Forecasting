# NEW-DeepTime-Deep-Learning-Framework-for-Time-Series-Forecasting
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

