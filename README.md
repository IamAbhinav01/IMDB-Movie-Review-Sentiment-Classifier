# IMDB-Movie-Review-Sentiment-Classifier
##### small description: a movie sentiment classifier gives the output as positive and negative along with prediciton score.

### PACKAGES REQUIRED

streamlit==1.38.0
tensorflow==2.12.0
numpy==1.23.5

### PLATFORMS TO USE
vs code (use if you are having powerful machine)
google colab (Recommended Use T4 GPU for fast performance)

## WORKING OF THE MODEL

In this model we are using SimpleRNN along with max_features=10000, which itself doesn't give accurate outputs for all the sentences for that we need to have an upgraded one like LSTM and high number of max_features .
comming to the model here we loaded the imdb dataset using the code : tensorflow.keras.dataset import imdb ( for imdb dataset)
after that we sorted out the key:value pair , but when we input the text we need value:key so we reveresed the dictionary to get this output, 
since the data was already pre-trained we just need to set or training_review and testing_reviews and afterwards we used optimizers like 'Adam' and loss like 'binaryclassentropy'
then in order to maintain uniform length we used pad-sequence and max_length, after that using sequential we trained the model using 'relu' activation and finally measured the model.fit with the epochs an trainig data , and used earlystopping with a patience level of 5 and obtained the accuracy and loss of this model then saved the model as 'SimpleRNN.h5', then loaded the model on streamlit file for converting to a webApp.



### SCREENSHOTS

![Screenshot 2025-06-10 172032](https://github.com/user-attachments/assets/954dbf2c-15c3-4c30-a5f1-2d5504c48063)



## DEPLOYEMENT

streamlit.io


## REFERENCES
https://keras.io/api/losses/probabilistic_losses/#binarycrossentropy-class
https://keras.io/api/datasets/imdb/
https://github.com/keras-team/keras/blob/v3.3.3/keras/src/datasets/imdb.py#L143-L188
https://www.tensorflow.org/api_docs/python/tf/keras/datasets/imdb/get_word_index



