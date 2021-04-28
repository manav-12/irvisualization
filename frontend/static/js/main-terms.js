function getTextWidth(text, font) {
        // if given, use cached canvas for better performance
        // else, create new canvas
        var canvas = document.createElement("canvas");
        var context = canvas.getContext("2d");
        context.font = font;
        var metrics = context.measureText(text);
        return metrics.width;
}