import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask, render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine

app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# Load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('InsertTableName', engine)

# Load model
model = joblib.load("../models/classifier.pkl")

# Index webpage displays visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # Extract data for visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    # first visualization: Distribution of Message Genres
    graph_genre = {
        'data': [
            Bar(
                x=genre_names,
                y=genre_counts,
                marker=dict(color='rgb(55, 83, 109)')
            )
        ],
        'layout': {
            'title': 'Distribution of Message Genres',
            'yaxis': {
                'title': "Count"
            },
            'xaxis': {
                'title': "Genre"
            }
        }
    }
    
    # second visualization: Distribution of Message Categories
    category_counts = df.drop(['id', 'message', 'original', 'genre'], axis=1).sum().sort_values(ascending=False)
    category_names = list(category_counts.index)
    
    graph_category = {
        'data': [
            Bar(
                x=category_names,
                y=category_counts,
                marker=dict(color='rgb(55, 83, 109)')
            )
        ],
        'layout': {
            'title': 'Distribution of Message Categories',
            'yaxis': {
                'title': "Count"
            },
            'xaxis': {
                'title': "Category",
                'tickangle': -45
            }
        }
    }
    
    # Combine both graphs into a list
    graphs = [graph_genre, graph_category]
    
    # Encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # Render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)

# Web page that handles user query and displays model results
@app.route('/go')
def go():
    # Save user input in query
    query = request.args.get('query', '') 

    # Use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # Render the go.html page
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )

def main():
    app.run(host='0.0.0.0', port=3000, debug=True)

if __name__ == '__main__':
    main()
