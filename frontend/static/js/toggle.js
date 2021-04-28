function toggletab(evt, tabName) {
  var i, tabcontent, tablinks;
  tabcontent = document.getElementsByClassName("tabcontent");
  for (i = 0; i < tabcontent.length; i++) {
    tabcontent[i].style.display = "none";
  }
  tablinks = document.getElementsByClassName("tablinks");
  for (i = 0; i < tablinks.length; i++) {
    tablinks[i].className = tablinks[i].className.replace(" active", "");
  }
  document.getElementById(tabName).style.display = "block";
  evt.currentTarget.className += " active";
}


function toggleDocAndList() {
    var barColor = "#FFFFFF";
    $("#rankedListContainer")[0].style.visibility = 'hidden';
    $("#documentContentContainer")[0].style.visibility = 'visible';


    $(".btn-group").on("change", function(e){

      var target = e.target.getAttribute("id");
      if (target == "documentContentToggle"){
        if (($("#documentContentContainer")[0].style.visibility == 'hidden') && ($("#documentContentToggle")[0].disabled == false)){

            d3.selectAll(".barmenu")
                .remove()
            d3.selectAll(".bar-rect")
                    .style("fill", barColor)
            $("#rankedListContainer")[0].style.visibility = 'hidden';
            $("#documentContentContainer")[0].style.visibility = 'visible';
            $("#documentContentContainer")[0].style.width = '30%';
            $("#documentContentContainer")[0].style.height = '100%';


        }else if ($("#documentContentContainer")[0].style.visibility == 'visible'){
            $("#documentContentContainer")[0].style.visibility = 'hidden';
        }
      }else if (target == "rankedListToggle"){
        if (($("#rankedListContainer")[0].style.visibility == 'hidden')&& ($("#rankedListToggle")[0].disabled == false)){
        d3.selectAll(".barmenu")
                .remove()
            d3.selectAll(".bartitles-rect")
                .style("fill", barColor)

            $("#documentContentContainer")[0].style.visibility = 'hidden';
            $("#rankedListContainer")[0].style.visibility = 'visible';
            $("#rankedListContainer")[0].style.width = '30%';
            $("#rankedListContainer")[0].style.height = '100%';

        }else if ($("#rankedListContainer")[0].style.visibility == 'visible'){
            $("#rankedListContainer")[0].style.visibility = 'hidden';
        }

      }
});
}