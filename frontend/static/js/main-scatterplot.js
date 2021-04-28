var svg;
var xScale;
var yScale;
var datascatterplot;
var colorFocus = "#17a2b8";
var colorSuggestion = "#ffc107";
var colorNotRelevant = "#d75a4a";
var colorBase = "#007527";
var colorDefault = '#BDBDBD';
var documentFeedback = {};
var previousQueryText = "";
var wordsHighlight;
var qewordsHighlight;
var docTextMap = {};
var docTitleMap = {};
var suggestionProbMap = {};
var previousSelectedVal = []
var previousDocumentKeys;
function hexToRgb(hex) {
    // Expand shorthand form (e.g. "03F") to full form (e.g. "0033FF")
    var shorthandRegex = /^#?([a-f\d])([a-f\d])([a-f\d])$/i;
    hex = hex.replace(shorthandRegex, function (m, r, g, b) {
        return r + r + g + g + b + b;
    });

    var result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
    return result ? "rgb(" + [
        parseInt(result[1], 16),
        parseInt(result[2], 16),
        parseInt(result[3], 16)
    ].join(', ') + ")" : null;
}

function addWordToQuery(term) {
    document.getElementById('queryText').value = document.getElementById('queryText').value + ' ' + term
}

function resetAll() {
    document.getElementById('queryId').value = "";
    document.getElementById('queryText').value = "";
    /*document.getElementById("queryDescription").innerHTML = "";*/
    $("#resultbody tr").remove();
    $("#gramBody tr").remove();
    documentFeedback = {};
    previousQueryText = "";
    suggestionProbMap = {};
    previousSelectedVal = []
    previousDocumentKeys = [];
    docTextMap = {};
    docTitleMap = {};
    $("#qeterm-selection").html("");
    $("#qeterm-selection").trigger("chosen:updated")
    document.getElementById('scatterplotcontainer').innerHTML = "";
    document.getElementById('docview').innerHTML = "";
    LoadScatterplot()
}

function searchResults() {
    currentQueryText = document.getElementById('queryText').value
    var selectedValues = $('#qeterm-selection').val();
    var selectedQEText = selectedValues.join(' ')
    $.get('/wordsToHighlight?words=' + selectedQEText, function (resp) {
        qewordsHighlight = JSON.parse(resp)
    });
    $.get('/wordsToHighlight?words=' + currentQueryText, function (resp) {
        wordsHighlight = JSON.parse(resp)
    });

    if (Object.keys(documentFeedback).length > 0 && JSON.stringify(previousDocumentKeys ) !== JSON.stringify(Object.keys(documentFeedback))) {
        $.get('/queryExpansionTerms?feedbackDocs=' + JSON.stringify(documentFeedback) + '&queryText=' + currentQueryText, function (resp) {
            var suggestionList = resp['expansion_terms'];
            suggestionProbMap = resp['word_weight_dict'];
                var expansionterms = document.getElementById("qeterm-selection");
                expansionterms.innerHTML = ""
            for (var i = 0; i < suggestionList.length; i++) {
                $("#qeterm-selection").append('<option selected value=\"'+suggestionList[i]+'\">'+suggestionList[i]+'</option>');
            }
            $("#qeterm-selection").trigger("chosen:updated");
        });
    }else{
                LoadScatterplot()
    }
    previousQueryText = document.getElementById('queryText').value
    previousSelectedVal = selectedValues
    previousDocumentKeys = Object.keys(documentFeedback)
}

function populateResultTable(resultlist) {
    const table = document.getElementById("resultbody");
    $("#resultbody tr").remove();
    resultlist.forEach(item => {
        let row = table.insertRow();
        let docname = row.insertCell(0);
        docname.innerHTML = item.docname;
        let text = row.insertCell(1);
        docTextMap[item.docname] = item.text;
        docTitleMap[item.docname] = item.title;
        text.innerHTML = '<a href="#documentDialog" class="openDocModal" data-toggle="modal" data-id="' + item.docname + '">' + item.title + '</a>'
        ;
    });
}

function populateWordQueryGram(resultlist) {
    const table = document.getElementById("gramBody");
    $("#gramBody tr").remove();
    for (let word_data in resultlist) {
        let row = table.insertRow();
        let grams = row.insertCell(0);
        grams.innerHTML = "<div id=\""+word_data+"\" onclick=\'plotWordGramDocuments("+word_data+")\'>"+resultlist[word_data]+"<\div>";
    }
}

$(function () {
    $(document).on('click', '.openDocModal', function () {
        var docId = $(this).data('id');
        // send an AJAX request to fetch the data
        $('#documentDialog #myModalLabel').html(docTitleMap[docId]);
        $('#documentDialog .modal-body').html('<div style=\"white-space: pre-wrap;\">' + docTextMap[docId] + '</div>');
        return false;
    });
});

function filterDocuments() {
    var checkedFilter = $('#docFilter').is(':checked')
    if (checkedFilter) {
        d3.selectAll(".dot").style("visibility", "hidden")
        var circles = d3.selectAll(".dot")._groups[0];
        for (var i = 0; i < circles.length; i++) {
            var circle_value = circles[i].__data__.matchKey
            if (circle_value > 0) {
                circles[i].setAttribute("style", "fill: " + circles[i].style.fill + "; stroke: " + circles[i].style.stroke + "; stroke-width: 1px; opactity: 1.0; visibility: visible");
            }
        }
    } else {
        d3.selectAll(".dot").style("visibility", "visible")
        d3.selectAll(".dot").style("opacity", "1.0")
    }
}

function plotWordGramDocuments(word_gram_num) {
    var checkedFilter = $('#docFilter').is(':checked')
    if ($.type(word_gram_num) == "number" ) {
        d3.selectAll(".dot").style("visibility", "hidden")
        var circles = d3.selectAll(".dot")._groups[0];
        for (var i = 0; i < circles.length; i++) {
            var circle_value = circles[i].__data__.word_group
            if (circle_value == word_gram_num || circles[i].__data__.name == 'Query') {
                circles[i].setAttribute("style", "fill: " + circles[i].style.fill + "; stroke: " + circles[i].style.stroke + "; stroke-width: 1px; opactity: 1.0; visibility: visible");
            }
        }
    } else {
        d3.selectAll(".dot").style("visibility", "visible")
        d3.selectAll(".dot").style("opacity", "1.0")
    }
}

function LoadScatterplot() {
    var queryId = document.getElementById('queryId').value
    var queryText = document.getElementById('queryText').value
    var selectedValues = $('#qeterm-selection').val();
    docTextMap = {};
    docTitleMap = {};
    $("#loader-modal").modal({
      backdrop: "static", //remove ability to close modal with click
      keyboard: false, //remove option to close with keyboard
      show: true //Display loader!
    });
    d3.request("/scatterplot")
        .header("Content-Type", "application/json")
        .post(JSON.stringify({queryId: queryId, queryText: queryText,expansionTerms: selectedValues,
            expansionMap:JSON.stringify(suggestionProbMap),feedbackDocs: JSON.stringify(documentFeedback)}),
            function (error, d) {
            try {
                //PARAMETERS AND VARIABLES
                parsedresp = JSON.parse(d.responseText);
                datascatterplot = JSON.parse(parsedresp['scatterplot']);
                resultlist = parsedresp['retrieval_results'];
                word_query_gram = parsedresp['word_gram'];
                populateResultTable(resultlist);
                populateWordQueryGram(word_query_gram)
                d3.select(".scatterplot-svg").remove();
                d3.select(".tooltip").remove();
                // add the tooltip area to the webpage
                var tooltip = d3.select("body").append("div")
                    .attr("class", "tooltip")
                    .style("opacity", 0);

                svg = d3.select("#scatterplotcontainer").append("svg"),
                    margin = {top: 20, right: 20, bottom: 30, left: 40},
                    width = $("#scatterplotcontainer").width() , // - margin.left - margin.right,
                    height = window.innerHeight - margin.top - margin.bottom /*- $("#termscontainer").height()*/,
                    g = svg.append("g").attr("transform", "translate(" + (margin.left) + "," + margin.top + ")");

                var barHeight = 30;
                var barWidth = 120;

                // setup x
                var xValue = function (d) {
                    return d.X1;
                }; // data -> value
                xScale = d3.scaleLinear().range([0, width]).nice(); // value -> display
                var xMap = function (d) {
                    return xScale(xValue(d));
                } // data -> display

                // setup y
                var yValue = function (d) {
                    return d.X2;
                }; // data -> value
                yScale = d3.scaleLinear().range([height, 0]).nice(); // value -> display
                var yMap = function (d) {
                    return yScale(yValue(d));
                }; // data -> display
                //setup r
                var rValue = function (d) {
                    return +d.matchKey + 1 * 6;
                };
                var rMap = function (d) {
                    return rValue(d);
                };

                xScale.domain([d3.min(datascatterplot, xValue) - 1, d3.max(datascatterplot, xValue) + 1]);
                yScale.domain([d3.min(datascatterplot, yValue) - 1, d3.max(datascatterplot, yValue) + 1]);

                var zoomBeh = d3.zoom()
                    .scaleExtent([.5, 20])
                    .extent([[0, 0], [width, height]])
                    .on("zoom", zoom);

                d3.select("#scatterplotcontainer")
                    .style("height", height + "px")
                    .style("width", width + "px")
                svg.attr("height", height)
                    .attr("width", width)
                    .attr("class", "scatterplot-svg")
                    .call(zoomBeh)


                svg.selectAll(".dot")
                    .data(datascatterplot)
                    .enter().append("circle")
                    // .filter(function(d) { return d.matched_keyword >= num_keyword })
                    .attr("r", rMap)
                    // .attr("r", function(d, i) { return sizeScale((Math.random() * 30)); })
                    // .style("fill-opacity", function(d, i) { return opacityScale(i); })
                    .attr("cx", xMap)
                    .attr("cy", yMap)
                    .style("fill", function (d) {
                        return ScatterplotColor(d.name, d.matchKey)
                    })
                    .style("stroke", "#000")
                    .attr("class", "dot")
                    .on("mouseover", function (d) {
                        d3.transition()
                            .duration(200)
                            .select('.tooltip')
                            .style("opacity", .9)
                        tooltip.html("<tooltiptext>" + d.name + "</tooltiptext>")
                            .style("width", (getTextWidth(d.name, "12px Arial") + 20) + "px")
                            .style("left", svg.node().getBoundingClientRect().left + "px")
                            .style("top", svg.node().getBoundingClientRect().top + "px")
                    })
                    .on("mouseout", function (d) {
                        d3.transition()
                            .duration(500)
                            .select('.tooltip')
                            .style("opacity", 0);
                    })
                    .on("click", function (d, i) {
                        d3.selectAll(".dot").style("stroke-width", "1px")
                        if (d3.select(this).style("stroke-width") == "1px") {
                            d3.select(this).style("stroke-width", "3px")
                            OpenDocument(d.name);
                            var selected_circle = d3.select(this);
                            //Removing any existing menu
                            d3.selectAll(".barmenu")
                                .remove()
                            var options_scatter = d3.select("div#menucontainer").append("svg")
                                .attr("class", "barmenu")
                                .attr("width", function () {
                                    return barWidth;
                                })
                                .attr("height", function () {
                                    return barHeight + 10
                                })
                                .attr("fill", "#d8eaff")
                            yPos = d3.event.pageY - 35;
                            d3.select("div#menucontainer")
                                .style("left", function () {
                                    if ((selected_circle.style("fill") != hexToRgb(colorFocus)) && (selected_circle.style("fill")) != hexToRgb(colorNotRelevant)) {
                                        return (d3.event.pageX - 110) + "px";
                                    } else {
                                        return (d3.event.pageX - 55) + "px";
                                    }

                                })
                                .style("top", function () {
                                    return yPos + "px"
                                })

                            options_scatter.append("rect")
                                .attr("class", "barmenu")
                                .attr("height", barHeight)
                                .attr("width", 70)
                                .attr("fill", "#a9a9a9")
                                .attr("y", 7)
                                .attr("x", 30)

                            options_scatter.append("svg:image")
                                .attr("class", "barmenu")
                                .attr("xlink:href", "../static/images/close.png")
                                .attr("width", "20px")
                                .attr("y", 0)
                                .attr("x", function () {
                                    return 90;
                                })
                                .on("click", function () {
                                    d3.selectAll(".barmenu")
                                        .remove()

                                })


                            options_scatter.append("svg:image")
                                .attr("class", "barmenu")
                                .attr("xlink:href", "../static/images/upvote.png")
                                .attr("width", "25px")
                                .attr("y", 10)
                                .attr("x", 35)
                                .on("click", function () {
                                    d3.selectAll(".barmenu")
                                        .remove()
                                    selected_circle.style("fill", hexToRgb(colorBase))
                                    documentFeedback[d.name] = 2
                                })

                            options_scatter.append("svg:image")
                                .attr("class", "barmenu")
                                .attr("xlink:href", "../static/images/downvote.png")
                                .attr("width", "25px")
                                .attr("y", 10)
                                .attr("x", 65)
                                .on("click", function () {
                                    d3.selectAll(".barmenu")
                                        .remove()
                                    selected_circle.style("fill", hexToRgb(colorNotRelevant))
                                    documentFeedback[d.name] = 0

                                })

                            // options_scatter.append("svg:image")
                            //     .attr("class", "barmenu")
                            //     .attr("xlink:href", "../static/images/clear.png")
                            //     .attr("width", "25px")
                            //     .attr("y", 10)
                            //     .attr("x", 65)
                            //     .on("click", function () {
                            //         d3.selectAll(".barmenu")
                            //             .remove()
                            //         selected_circle.style("fill", hexToRgb(colorDefault))
                            //         delete documentFeedback.d.name;
                            //
                            //     })


                        }
                    });
                // filterDocuments();
                hideLoading();
            } catch (err) {
                setTimeout(function() {
                    hideLoading();
                }, 500);
                console.log("error occured")
            }
        }).on('error', function (e) {
                setTimeout(function() {
                    hideLoading();
                }, 1000);
            console.log('Error occurred ');
    })
}

function hideLoading() {
    $("#loader-modal").modal('hide');
}

function zoom() {
    var new_xScale = d3.event.transform.rescaleX(xScale);
    var new_yScale = d3.event.transform.rescaleY(yScale);
    svg.selectAll(".dot")
        .attr('cx', function (d) {
            return new_xScale(d['X1'])
        })
        .attr('cy', function (d) {
            return new_yScale(d['X2'])
        });
}

function transform(d) {
    return "translate(" + xScale(d['X1']) + "," + yScale(d['X2']) + ")";
}

/* Functions called from control.js */

function StrokeCircle(docname) {
    var circles = d3.selectAll(".dot")._groups[0];
    d3.selectAll(".dot").style("stroke-width", "1px")
    for (var i = 0; i < circles.length; i++) {
        if (circles[i].__data__.name == docname) {
            var doc = circles[i];
            break;
        }
    }
    doc.setAttribute("style", "fill: " + doc.style.fill + "; stroke: " + doc.style.stroke + "; stroke-width: 3px;");
}

function UpdateScatterplotColors(name) {
    if (name != "null") {
        var circles = d3.selectAll(".dot")._groups[0];

        for (var i = 0; i < circles.length; i++) {
            if (circles[i].__data__.name == name) {
                circles[i].setAttribute("style", "fill: " + ScatterplotColor(circles[i].__data__.name) + "; stroke: " + circles[i].style.stroke + "; stroke-width: 1px; opactity: 1.0; visibility: " + circles[i].style.visibility)
                break;
            }
        }
    } else {

        var circles = d3.selectAll(".dot")._groups[0];

        for (var i = 0; i < circles.length; i++) {
            circles[i].setAttribute("style", "fill: " + ScatterplotColor(circles[i].__data__.name) + "; stroke: " + circles[i].style.stroke + "; stroke-width: 1px; opactity: 1.0; visibility: " + circles[i].style.visibility)
        }
    }

}
