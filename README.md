# Music Recommendation

In this project I create and train the model for dataset https://www.kaggle.com/c/kkbox-music-recommendation-challenge/data. The aim is to predict if a user will listen a song for a second time within a month after the first time. Based on this information a recommendation system can be built.

## Cleaning the data, creating and training the model

The corresponding notebook with step by step explanation is [data_cleaning_and_model_training.ipynb](data_cleaning_and_model_training.ipynb).

## Serving the model using REST API

The file that creates REST API to serve the model is [Model_Server.py](Model_Server.py).

## Testing

To test the server, you can use notebook [test_model_server_on_sample_data.ipynb](test_model_server_on_sample_data.ipynb). In this notebook the request is created to sent to the running server and get the response. Make sure the server is ready to receive requests before running the test.