import json
import pickle

from flask import Flask, render_template, request, jsonify
from backend.expansion.QueryExpansion import QueryExpansion
from backend.irviz.VizMatrix import VizMatrix
from backend.utils.TextClean import TextClean
from backend.utils.processData import DataReader
import os
import logging

app = Flask(__name__)
log = logging.getLogger('werkzeug')
log.disabled = True
logging.basicConfig(filename='backend/logs/irwebapp.log', level=logging.DEBUG)

dataReader = DataReader()
DATA_FLAG = ""
ori_doc = dataReader.read_dataset(os.path.join("backend", "data", DATA_FLAG, "data_csvs", "ORI_"+DATA_FLAG+".csv"))
all_doc = dataReader.read_dataset(os.path.join("backend", "data", DATA_FLAG, "data_csvs", "SNOW_"+DATA_FLAG+".csv"))
retrieval = VizMatrix(ori_doc, all_doc, 3500)
queryExpansion = QueryExpansion()
queryExpansion.set_params_em_model({})
stemFile = open(os.path.join("backend", "data", DATA_FLAG, "data_csvs", "stem_highlight.pkl"), "rb")
highlight_map = pickle.load(stemFile)


@app.route("/home")
def index():
    """
    :return: Renders the home screen index page
    """
    return render_template("index.html", data=None)


@app.route("/scatterplot", methods=["POST"])
def get_scatter_plot():
    """
    :request_params: gets queryId and queryText in post request
                     [Optional]: If there are feedback terms selected by the user the probablities of the terms
    :return: returns a json of scatterplot to plot the points.
                                word_gram: group of terms from query occuring together in a document
                                retrieval_results: top-k results near to query vector
    """
    logging.debug("============================================")
    data = request.get_json()
    query_id = data['queryId']
    query_text = data['queryText']
    logging.debug("Query Text: "+query_text)
    query_feedback = json.loads(data['feedbackDocs'])
    expansion_terms = data['expansionTerms']
    logging.debug("Expansion Terms: " + ",".join(expansion_terms))
    expansion_terms_map = json.loads(data['expansionMap'])
    try:
        scatter_plot_json, word_gram, retrieval_score, metric_result =\
            retrieval.create_matrix((query_id, query_text),expansionTerms = expansion_terms,
                                    expansionMap = expansion_terms_map,feedback_docs=query_feedback)
        response = {'scatterplot': scatter_plot_json, 'word_gram':word_gram, 'retrieval_results': [
            {'docname': k[0], 'title': ori_doc[k[0]].split('\n')[0], 'text': ori_doc[k[0]]} for k in
            retrieval_score[:20]]}
        return jsonify(response)
    except:
        return ''


@app.route("/getQueryText", methods=["GET"])
def get_query_text():
    """
    :request_params: gets the queryId
    :return: returns the respective query for the given query id
    """
    question_id = request.args.get('queryId', '')
    print(question_id)
    query_map = retrieval.short_queries
    query_relevance = retrieval.gold_std[question_id]
    try:
        response = {'query_text': query_map[question_id].lower(),'query_relevance': query_relevance}
    except:
        response = ""
    return jsonify(response)


@app.route("/getDocument", methods=["GET"])
def get_document():
    """
    :request_param: get docname
    :return: Returns the text for the document to be rendered in document view
    """
    doc_id = request.args.get('docname', '')
    return ori_doc[doc_id]


@app.route("/wordsToHighlight", methods=["GET"])
def words_to_highlight():
    """
    :request_params: get query words
    :return: returns all the possible set of morphological words to be highlighted
    """
    words_highlight = request.args.get('words', '')
    cleaned_words = TextClean().normalizeNltkLemma(words_highlight, 'SNOWSTEM').split()
    final_word_list = []
    for x in cleaned_words:
        try:
            final_word_list.extend(highlight_map[x])
        except:
            final_word_list.extend([x])
    return json.dumps(final_word_list)


@app.route("/getRelevantDocs", methods=["GET"])
def get_relevant_docs():
    # question_id = int(request.args.get('queryId', ''))
    return json.dumps([])


@app.route("/queryExpansionTerms", methods=["GET"])
def query_expansion_terms():
    """
    :request_params: get feedback documents as input request
    :return:
    """
    feedback_doc = json.loads(request.args.get('feedbackDocs', ''))
    curr_query = request.args.get('queryText', '')
    expansion_terms,word_weight_dict = queryExpansion.getQueryResults(curr_query, feedback_doc)
    response = {'expansion_terms': expansion_terms,'word_weight_dict':word_weight_dict}
    return jsonify(response)


if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=False, port=5000)
