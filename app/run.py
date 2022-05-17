import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    """
    This function takes text and convert it into clean tokens

    INPUT: string - this can be a sentance or text
    OUTPUT: clean tokens

    """
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# Load data
engine = create_engine('sqlite:///../data/disaster.db')
df = pd.read_sql_table('disaster', engine)

# Load model
model = joblib.load("../models/classifier.pkl")



@app.route('/')
@app.route('/index')
def index():
    """
    Index webpage displays visuals and receives user input text for model

    INPUT: None
    OUTPUT: webpage with plotly graphs

    """

    columns_list = list(df.iloc[:,4:].columns)
    count_ones = df.iloc[:,4:].sum()
    for_pie = df.genre.value_counts().reset_index()
    graph_one = dict({
    "data": [{"type": "bar",
              "x": columns_list,
              "y": count_ones}],
    "layout": {"title": dict(text="Frequency of categories"),
    'xaxis': dict(title='Categories', tickangle=30),
    'yaxis': dict(title='Frequency')}
    })
    graph_two = dict({
    'data': [{'type': 'pie',
    'labels': for_pie['index'],
    'values': for_pie['genre'],
    'textinfo': 'label+percent',
    'hole': 0.4}],
    'layout': {'title': dict(text='Distribution of information source')}
    })



    graphs = [dict(data=graph_one), dict(data=graph_two)]



    # Encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # Render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)



@app.route('/go')
def go():
    """
    Web page that handles user query and displays model results

    INPUT: None
    OUTPUT: webpage with predictions
    """
    # Save user input in query
    query = request.args.get('query', '')

    # Use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))


    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()
