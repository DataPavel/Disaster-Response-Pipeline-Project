# Disaster Response Pipeline Project

## Project Description

#### What my application does

This project includes a web app where an emergency worker can input a new message and get classification results in several categories. The web app also displays visualizations of the data.

#### Technologies used
- python  
- bootstrap  
- html  

#### Future improvements

The visualization of the data on the index page shows that most of the categories are imbalanced.  
In ML pipeline, I used MultiOutput Classifier, that predicts multiple categories. The Multioutput classifier does not allow to resample the data and as a result some categories have f_1 score of 0.  
This can be improved by taking a separate model for each category and for each model the data should be resampled.

## Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/disaster.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/disaster.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. When you run the application, please use port 3001

## Credits

- [Udacity.com](https://www.udacity.com)
- [Stackoverflow](https://stackoverflow.com)
