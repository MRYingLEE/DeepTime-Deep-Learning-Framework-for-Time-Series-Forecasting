# DeepTime-Deep-Learning-Framework-for-Time-Series-Forecasting
This is an INTERACTIVE deep learning framework for time series forecasting. It uses Tensorflow 2+tf.karas.

# Roadmap <August 2021>
More than 1 year passed. I plan to upgrade the deeptime greatly. 

## TFX (Tensorflow Extended) will be used, besides Tensorflow

## More advanced model will be supported
So far, the model of Informer, a variant of Transformer, is my preference.

## Time Series specific datetime embedding will be supported
Datetime embedding is very specific to time series data. Time2Vec (https://github.com/ojus1/Time2Vec-PyTorch) will be the prefered embedding method.

## Some critical choice for the next version

### Tensorflow vs Pytorch
As a practioner instead of a researcher, I will continue to use Tensorflow for it is easy to put Tensorflow models into production due to its bigger ecosystem.

### Tensorflow vs Keras
I will continue to use Keras as an extension of Tensorflow.

###  Keras (Sequential or Functional API vs Subclassing)
There is a good comparion, What are Symbolic and Imperative APIs in TensorFlow 2.0? (
https://blog.tensorflow.org/2019/01/what-are-symbolic-and-imperative-apis.html).

The following is digested from the book of "Machine Learning Design Patterns: Solutions to Common Challenges in Data Preparation, Model Building, and MLOps".(https://www.amazon.com/Machine-Learning-Design-Patterns-Preparation/dp/1098115783/ref=sr_1_8?crid=7WCGGZGRVFM2&dchild=1&keywords=Machine+Learning&qid=1627808217&sprefix=math+data+s%2Caps%2C377&sr=8-8)

```
TensorFlow 2.0 supports both of these styles out of the box, so you can choose the right level of abstraction
(and complexity) for your project.
If your goal is ease of use, low conceptual overhead, and you like to think about your models as graphs of layers:
use the Keras Sequential or Functional API (like plugging together LEGO bricks) and the built-in training loop. 
This is the right way to go for most problems.
If you like to think about your models as an object-oriented Python/Numpy developer, and you prioritize flexibility
and hackability, Keras Subclassing is the right API for you.
```

```
The two styles are fully interoperable as well, so you can mix and match (for example, you can nest one model type
in another). 
You take a symbolic model and use it as a layer in a subclassed model, or the reverse.
```
I prefer an easy and flexible style, so I will choose Functional API. And Subclassing can be embedded.


#### Data Transform (Feature column vs TFT vs Keras Preprocessing)
In the future, I will use TFT, Tensorflow Transform.

```
It is helpful to differentiate between instance-level transformations that can be part of the modeldirectly 
(where the only drawback is applying them on each training iteration) and dataset-leveltransformations, 
where we need a full pass to compute overall statistics or the vocabulary of acategorical variable. 
Such dataset-level transformations cannot be part of the model and have tobe applied as a scalable preprocessing step, 
which produces the Transform, capturing the logicand the artifacts (mean, variance, vocabulary, and so on)
to be attached to the model. Fordataset-level transformations, use tf.transform.
```

```
The tf.transform library (which is part of TensorFlow Extended)provides an efficient way of carrying out 
transformations over apreprocessing pass through the data and saving the resulting features andtransformation 
artifacts so that the transformations can be applied byTensorFlow Serving during prediction time.
```

Is tensorflow.keras.layers.experimental.preprocessing a good solution for transform? This question has puzzled me for some time. Finally I realized that it is not a efficient way to do preprocessing this way for the transform will be done for every epoch. 

So in short, TFT is an efficient tranform library at a cost of complication, especially when Apache Beam is involved.

### Data API
Pandas is easy, but not suitable for big dataset.

I will use ExampleGen, a TFX Pipeline component, to ingest data.

```
The ExampleGen TFX Pipeline component ingests data into TFX pipelines. It consumes external files/services to 
generate Examples which will be read by other TFX components. It also provides consistent and configurable 
partition, and shuffles the dataset for ML best practice.

Consumes: Data from external data sources such as CSV, TFRecord, Avro, Parquet and BigQuery.
Emits: tf.Example records, tf.SequenceExample records, or proto format, depending on the payload format.
```

In conclusion, I will use TFX as the underlying platform and Informer as the default model.

# A quick demo

## 1. To define your problem by code
```
def get_configs():
  if not path.exists('bike_data/hour.csv'):
    ! wget https://archive.ics.uci.edu/ml/machine-learning-databases/00275/Bike-Sharing-Dataset.zip
    ! unzip Bike-Sharing-Dataset.zip -d bike_data

  df = pd.read_csv('bike_data/hour.csv', index_col='instant')
  cols_to_keep = [       
    'cnt',
    'temp',
    'hum',
    'windspeed',
    'yr',
    'mnth', 
    'hr', 
    'holiday', 
    'weekday', 
    'workingday'
  ]
  dataframe = df[cols_to_keep]
  default_inputs=['cnt','temp','hum','windspeed']  # The default features list for inputs
  default_labels=['cnt']  # The default features list for labels
  default_future=list(set(cols_to_keep)-set(default_inputs))

  dataframe_train, dataframe_test = train_test_split(dataframe, test_size=0.2)
  dataframe_train, dataframe_val = train_test_split(dataframe_train, test_size=0.2)
  print(len(dataframe_train), 'train examples')
  print(len(dataframe_val), 'validation examples')

  past_history = 24 * 7 * 3 
  future_target = 24 * 5
  categories_limit=20
  batch_size = 32
  single_step=False

  return dataframe, dataframe_train, dataframe_val, dataframe_test,default_inputs,default_labels,default_future, past_history, future_target,categories_limit, batch_size,single_step
  # The order of the variables is very important. 
```

## 2. To normalize data by mouse
![features](https://github.com/MRYingLEE/DeepTime-Deep-Learning-Framework-for-Time-Series-Forecasting/blob/master/images/seq2seq_features.JPG "features")

## 3. To define sequence by mouse
![periods](https://github.com/MRYingLEE/DeepTime-Deep-Learning-Framework-for-Time-Series-Forecasting/blob/master/images/seq2seq_periods.JPG "periods")

# 4. To train
![training](https://github.com/MRYingLEE/DeepTime-Deep-Learning-Framework-for-Time-Series-Forecasting/blob/master/images/seq2seq_training.JPG "training")

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
Time series (TS) data is focused. 

Logically, time-series data has the 3D shape of (n_series, n_timestamp, n_features).

## Some related questions:
Is this an autoML platform? NO!<br>
Are there any built-in models? Yes, but 1 vanilla LSTM and 1 Seq2Seq model was built-in only for reference.<br>
Are there hyperparamerters tuning built in? Maybe, but not yet.<br>
Can I define a new model? Yes, you may call 
```
def train_by_model(self,model_fn, epochs=1)
```
# Which?
Is this a whole pipeline framework? No, I won't cover data preprocessing, which can be covered by Time-series-Preprocessing-Studio-in-Jupyter (https://github.com/MRYingLEE/Time-series-Preprocessing-Studio-in-Jupyter ）.

# Technology
1. Tensorflow 2+tf.karas as deep learning framework. 

2. tf.data.Dataset will be used as basic data format. 

# Progress

## Interactive Features Builder

(I published an introduction at:
https://medium.com/@mr.ying.lee/interactive-features-builder-in-jupyter-notebook-for-tensorflow-2-2322575733c9).

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

We may INTERACTIVELY build the features based on the dataframe.

Inline-style: 
![builder](https://github.com/MRYingLEE/DeepTime-Deep-Learning-Framework-for-Time-Series-Forecasting/blob/master/Features-Builder/builder.JPG "builder")

The builder will interactively generate some source code for us:
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

You may use the code directly or modify it further.


# Credit
The built-in models are modified from the following:

[Alessandro Angioi] Time Series Forecasting with an LSTM Encoder/Decoder in TensorFlow 2.0(https://www.angioi.com/time-series-encoder-decoder-tensorflow/)<br>
[Tensorflow Team] Time series forecasting (https://www.tensorflow.org/tutorials/structured_data/time_series)

