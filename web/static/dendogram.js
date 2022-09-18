// https://codepen.io/hnq90/pen/YvoxMJ
// Interpolates two [r,g,b] colors and returns an [r,g,b] of the result
// Taken from the awesome ROT.js roguelike dev library at
// https://github.com/ondras/rot.js
var _interpolateColor = function (color1, color2, factor) {
    if (arguments.length < 3) {
        factor = 0.5;
    }
    var result = color1.slice();
    for (var i = 0; i < 3; i++) {
        result[i] = Math.round(result[i] + factor * (color2[i] - color1[i]));
    }
    return result;
};


// https://www.d3-graph-gallery.com/graph/dendrogram_basic.html
function setupDendogram(cluster_id) {

    // read json data
    let endpoint = "/dendogram"
        + "/" + $(GLOBAL_DATASET_NAME).val()
        + "/" + $(GLOBAL_CLASS_IDX).val()
        + "/" + cluster_id.toString();
    d3.json(endpoint)
        .timeout(5 * 60 * 1000)
        .get(dendoCallback);
}

function dendoCallback(data) {
    // d3.json("https://raw.githubusercontent.com/holtzy/D3-graph-gallery/master/DATA/data_dendrogram.json", function (data) {

    console.log("setting up dendogram");
    // set the dimensions and margins of the graph
    var width = window.innerWidth;
    var height = 200;

    // append the svg object to the body of the page
    var svg = d3.select("#my_dataviz")
        .append("svg")
        .attr("width", width)
        .attr("height", height)
        .append("g")
        .attr("transform", "translate(0,40)");  // bit of margin on the top = 40

    console.log(data);

    // Create the cluster layout:
    var cluster = d3.cluster()
        .size([width, height - 100]);  // 100 is the margin I will have on the right side
    // var cluster = d3.cluster.size([width, height - 100]);

    // Give the data to this cluster layout:
    var root = d3.hierarchy(data, function (d) {
        return d.children;
    });
    cluster(root);


    // Add the links between nodes:
    svg.selectAll('path')
        .data(root.descendants().slice(1))
        .enter()
        .append('path')
        .attr("d", function (d) {
            return "M" + d.x + "," + d.y + " " + d.parent.x + "," + d.parent.y;
        })
        .style("fill", 'none')
        .attr("stroke", function (d) {
            return d.data.edge_color;
        })


    // Add a circle for each node.
    svg.selectAll("g")
        .data(root.descendants())
        .enter()
        .append("g")
        .on("click", click)
        .on("mouseover", function (d) {
            console.log(d);
            var g = d3.select(this); // The node
            // The class is used to remove the additional text later

            let purity_string = "null";
            if (d.data.purity !== null) {
                purity_string = d.data.purity.toString()
            }

            let text = "cluster_id: " + d.data.cluster_id.toString() + ", purity: " + purity_string + ", num_segments: " + d.data.num_segments.toString();
            let info = g.append('text')
                .classed('info', true)
                .attr('x', 20)
                .attr('y', 10)
                .text(text);
        })
        .on("mouseout", function () {
            // Remove the info text on mouse out.
            d3.select(this).select('text.info').remove()
        })
        .attr("transform", function (d) {
            return "translate(" + d.x + "," + d.y + ")"
        })
        .append("circle")
        .attr("r", 14)
        .style("fill", function (d) {
            return d.data.color_string;
        }).attr("stroke", function (d) {
            return d.data.stroke;
        })
        .style("stroke-width", 4)

}

// Toggle children on click.
function click(d) {
    console.log(d);
}