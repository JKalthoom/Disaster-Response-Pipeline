## Disaster-Response-Pipeline
In this project, I will apply some of the skills I learned through the Data Engineering course provided by Udacity to analyze disaster data from <a href="https://www.appen.com/"><strong>Appen</strong></a>, to build a robust machine learning pipeline capable of categorizing disaster messages, so we can send these massages to the apprpriate disaster relief agency. The dataset comprises real messages sent during various disaster events, presenting a unique opportunity to apply data engineering and machine learning techniques in a critical, real-world scenario. <br> <br>
Udacity were making sure to make this project accessible and user-friendly, they provided a web application where emergency workers can input new messages and receive real-time classification results. Additionally, the web app will feature visualizations of the data, providing valuable insights into the distribution and frequency of different types of disaster-related messages.

## Project Motivation <a name="motivation"></a>
In times of disaster, timely and accurate communication is critical for effective response and relief efforts. During such events, a massive volume of messages and information is generated, which can overwhelm relief agencies and delay critical response actions. By leveraging the power of data science and machine learning, we can automate the classification of these messages, ensuring that they reach the appropriate disaster relief agencies quickly and efficiently. This project aims to develop a machine learning model and an associated web application to classify disaster-related messages, thereby improving the speed and accuracy of disaster response.

## Table of contents
- [Project Motivation](#motivation)
- [Installation](#installation)
- [File Descriptions](#files)
- [How to Interact](#interaction)
- [Results](#results)
- [Licensing, Authors, Acknowledgements](#licensing)

## Installation <a name="installation"></a>
The whole project is provided above as a Python scripts, it should work properly in the terminal but you might need to install some libraries like: <br> <br>
  - <a href="https://numpy.org/install/"><strong>numpy</strong></a> <br>
  - <a href="https://pandas.pydata.org/pandas-docs/stable/getting_started/install.html"><strong>pandas</strong></a> <br>
  - <a href="https://flask.palletsprojects.com/en/3.0.x/installation/"><strong>flask</strong></a> <br>
  - <a href="https://pypi.org/project/plotly/"><strong>plotly</strong></a> <br>
  - <a href="https://pypi.org/project/nltk/"><strong>nltk</strong></a> <br>
  - <a href="https://pypi.org/project/scikit-learn/"><strong>scikit-learn</strong></a> <br>
  
## File Descriptions <a name="files"></a>
There are three main components in this project. <br>
- The ETL Pipeline (process_data.py), this script will loads the messages and the categoies datasets, merge them, cleans the data, and finaly stores it in a SQLite database.
- The ML Pipeline (train_classifier.py), this script will loads data from the SQLite database, splits the dataset into training and test sets, builds a text processing and machine learning pipeline, trains and tunes a model using GridSearchCV, outputs results on the test set, exports the final model as a pickle file.
- The Flask Web App (run.py), this script should runs all the project together and provide for you a web app to interact and visualize some important information about the dataset. <br>
Finally, all the files are well commented so it should be easy to understand. 

## How to Interact <a name="interaction"></a>
In your terminal. <br>
1. Run the following commands in the project's root directory to set up your database and model.
   - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
   - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`
2. Go to `app` directory: `cd app`
3. Run your web app: `python run.py`

## Results <a name="results"></a>


## Licensing, Authors, Acknowledgements <a name="licensing"></a>
