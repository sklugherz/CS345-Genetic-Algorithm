# CS345-LSTM
A project for CS345 - Introduction to Machine Learning.


## Project Proposal

### Topic 
A day trading bot that utilizes LSTM, Long Short Term Memory, Machine Learning to predict opening and closing positions to make buy and sell decisions. 

### Interest
We are interested in exploring the applications of Machine Learning to analyze stock market data for potential use in the real world. We are both interested in trading stock and building portfolios and want to see if an accurately trained Machine Learning Model can assist in portfolio development. 

### Timeline
* Collect data 
* Data Visualizations
* Data Preprocessing
* Create Training Data
* Building the LSTM Model
* Training the model
* Making predictions
* Visualize Predictions
* Evaluate Model Performance
* Determine viability of model for real world use

### Timeline/Roles 
This project requires several preparatory steps. We need to create a training and testing set ourselves because our strategy will be unique. In order, Stephen will collect the data for this project from Kaggle. The data can then be visualized and preprocessed. Then, Thomas will create a script to label data as a winning or losing trade given a fixed risk/reward ratio. This will form our training and test set for the model. The strategy that will be used is based on a prediction of the next day’s opening and closing price based on the previous ones. Now that we have a workable set of data, we will build the LSTM model that will actually give us buy and sell signals. This will be a combined effort of both Thomas and Stephen, since this is the bulk of the project. This implementation of LSTM will include a draw_trade method that will help us visualize the trade timing being made. Later, this can be changed to a buy or sell with a daytrading API. The model will make predictions, and Stephen will plot the model’s predictions. Thomas will create a function to determine the accuracy of the model. Then we will graph the model’s performance as a function of the number of trailing days in the strategy to optimize the performance. Finally, we will determine the profitability of the model by comparing the accuracy with the risk reward ratio. For example, if the risk reward ratio is 50%, and we make a winning trade 45% of the time, we know the model is losing money in the long run. The accuracy of the predictions for a 50% risk reward ratio must be above 50% in order to be called profitable.

### Datasets 
Large data set with various IPO’s to train based on data ranging from 1984-2017: https://www.kaggle.com/datasets/borismarjanovic/price-volume-data-for-all-us-stocks-etfs 
Large data set from single IPO with range from 2000-2021: https://www.kaggle.com/datasets/rohanrao/nifty50-stock-market-data?select=ASIANPAINT.csv 
Stock market sentiment data from twitter with news marked in binary good/bad: https://www.kaggle.com/datasets/yash612/stockmarket-sentiment-dataset 
Most relevant stock market data ranging from 2019-2024: https://www.kaggle.com/datasets/saketk511/2019-2024-us-stock-market-data 
