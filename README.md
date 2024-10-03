# Deep Learning on Stock Price Prediction Integrated with Sentiment Analysis

In this notebook, I create a model for forecasting stock prices which takes into account not only historical data and technical indicators, but also such external factors influencing the market as the mood of traders and brand reputation, which is represented in social media posts. The sentiment and stock data are also provided in the Kaggle link below. 

## Reference: 
https://www.kaggle.com/code/vsmanichrome/stock-prediction-twitter-sentiment-analysis  \
https://www.youtube.com/watch?v=CbTU92pbDKw 

## Objectives: 
•	To extract public sentiment from sentiment data using NLP to enhance the accuracy of the prediction. \
•	To design and develop a DL model to predict the stock price using history data and sentiment data. \
•	To evaluate proposed DL model to find the best performing model using the evaluation metrics.  

## What is the project about?
The main task of this project is the same as for every ML project, where most of the time is used on preparing the data for model development. For our data, namely Stock data and twitter data, we created Technical indicators to increase the ability of the model to perform  prediction. This method is the so called Data Transformation. For the twitter data, we obtained the sentiment score from the tweets using the VADER library from NLTK. Then, we combined the data by integrating them on the same time and stock name column, this combination allows the Deep learning model to determine patterns not only related to the historical price data, but also with the involvement of public sentiment. 


After that, the LSTM model is then trained using these combined features. Hyperparameter tuning is also performed to ensure that the model achieves its best performance. The model is evaluated using mean absolute error, and we have a baseline model that does not use the sentiment score in the training data to compare with other models. As a result, we find that using technical data and also sentiment data improves the model performance. 

## Conclusion:
LSTM is a popular approach in time-series forecasting. This is due to its ability to remember the dependencies between different variables, and it takes the output of previous data as input to the next layer. However, deep neural networks are Blackbox models that are hard to interpret. Luckily, we can still use the evaluation metrics to evaluate the performance of the model.

•	We apply the VADER module from NLTK library to obtain the sentiment score of the tweets. Later, the sentiment score is separated by the time of creation of the tweets. This is useful since the sentiment that is collected from the public after the stock trading time will affect the price of the stock not on the same day, but on the next day. 

•	Besides, we built the model and apply hyperparameter tuning via GridSearchCV to further improve the model. The model that is trained with technical indicators are improved by 0.02 for the validation MAE. A function for creating the LSTM model is also defined, this eases the process of tuning the model since we will only need to use the function instead of repeating the whole bunch of code. 

•	To summarize the final objective, this study creates the visuals of the prediction results to further evaluation the model. Besides, the performance of the model is obtained via the MAE metrics. The inverse transform of the MinMaxscaler is used to convert the scaled price back to actual price. The actual price is then used to calculate the evaluation metrics. 

•	TSLA stock has obtained the highest accuracy, if compared to the other two, which suggests us include more sentiment and stock data to improve the accuracy of the model. 

Finally, predicting the stock price is not an easy task. The high volatility nature in the stock market implies that investors should be cautious before investing their hard-earned money. This project just provides an academic application, and the model is not really recommended to be used in the real world.

## Limitations & Recommendations:
•Based on the limitations of the model that is built, future work of this project can include a time step for the previous data. \
For instance, we could use the data of the past 3 days to predict the next day closing price. This will help to include more potential of the LSTM model since it is able to remember long-term dependencies in the data. 

•Besides, more layers and indicators can be added to the model to improve the power of the model. Macroeconomic indicators, such as GDP, inflation rate (CPI), and the global unemployment rate can be added to build the model.
Upload cp2 slides

## Appendix
Visuals.pdf (Uploaded) is the file that i use to present my work. Feel free to have a look on the details for the visualizations and other info. of this project.
