var relevant_list;
var suggestion_list;
var notrelevant_list;
var similardocuments_list;
var currRelDoc;
var colorFocus = "#17a2b8";
var colorSuggestion = "#ffc107";
var colorNotRelevant = "#d75a4a";
var colorBase = "#007527";
var highlightRe = /<span class="highlight">(.*?)<\/span>/g;
var highlightHtml = '<span class="highlight">$1</span>';
var highlightQEHtml = '<span class="highlight-qe">$1</span>';
var query_relevance;

function saveCurrRelevant(relList){
    currRelDoc = relList;
}

function getQueryText(){
    var queryId = document.getElementById('queryId').value;
    if (parseInt(queryId) >= 1 && parseInt(queryId) < 100  ){
        $.get('/getQueryText?queryId='+queryId,function(resp) {
            documentFeedback = {};
            $("#resultbody tr").remove();
            $("#gramBody tr").remove();
            documentFeedback = {};
            previousQueryText = "";
            suggestionProbMap = {};
            previousSelectedVal = []
            previousDocumentKeys = [];
            docTextMap = {};
            docTitleMap = {};
            getRelevantList();
            $("#qeterm-selection").html("");
            $("#qeterm-selection").trigger("chosen:updated")
            document.getElementById('scatterplotcontainer').innerHTML = "";
            document.getElementById('docview').innerHTML = "";
            query_text = resp['query_text'];
            query_relevance = resp['query_relevance'];
            document.getElementById("queryText").value = query_text;
           /* document.getElementById("queryDescription").innerHTML = resp;*/
       });
    }
}

function getRelevantList(){
    var queryId = document.getElementById('queryId').value;
    if ( parseInt(queryId) >= 0 && parseInt(queryId) < 31  ){
        $.get('/getRelevantDocs?queryId='+queryId,function(resp) {
            saveCurrRelevant(JSON.parse("[" + resp + "]")[0]);
       });
    }
}



function docRelevance(document){
    if (typeof documentFeedback !== 'undefined' && documentFeedback[document]  !== 'undefined') {
        if(documentFeedback[document] == 2){
           return 2
        }
        if(documentFeedback[document] == 0){
            return 0
        }
    }
    return -1;
}

// .range(["white", "blue"])

function ScatterplotColor(name,frequency){
    var myColor = d3.scaleSequential().domain([0,10]).interpolator(d3.interpolatePurples);
    doc_rel = docRelevance(name);
    if (doc_rel ==2 ){
            return "#007527";
    }
     if (doc_rel==0){
            return "#d75a4a";
    }
    if (name == 'Query'){
        return "#fff102";
    }

    // return '#BDBDBD';
    return myColor(frequency*2);
}

/* Common to more than 1 view */
function SelectOnScatterplot(document){   
    StrokeCircle(document.docname)
}


function performMark(respText) {
  var arrayLength = wordsHighlight.length;
  for (var i = 0; i < arrayLength; i++) {
     respText = respText.replace(new RegExp('(\\b' + wordsHighlight[i] + '\\b)', 'gi'), highlightHtml);
  }
  if(qewordsHighlight){
    qewordsHighlight = qewordsHighlight.filter(e => !wordsHighlight.includes(e))
    var qearrayLength = qewordsHighlight.length;
      for (var i = 0; i < qearrayLength; i++) {
         respText = respText.replace(new RegExp('(\\b' + qewordsHighlight[i] + '\\b)', 'gi'), highlightQEHtml);
      }
  }
  return respText
}


function OpenDocument(name){
   if (name != "Query"){
       $.get('/getDocument?docname='+name,function(resp) {
          document.getElementById("docview").innerHTML = document.getElementById("docview").innerHTML = "<div style=\"white-space: pre-wrap;\">DocName: "+name+'\n'+performMark(resp) + "</div>";
       });
   }else{
          document.getElementById("docview").innerHTML = document.getElementById('queryText').value;
   }
}
