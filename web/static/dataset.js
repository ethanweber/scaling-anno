// some global variables
var GLOBAL_IMAGE_PREFIX = "https://wednesday.ethanweber.me/ethan/polytransform_external/ade_places_data/";
var GLOBAL_BACKEND = "https://adebackend0.ethanweber.me";
var GLOBAL_IMAGE_HEIGHT = 400;
var GLOBAL_IMAGE_WIDTH = null;
var GLOBAL_IMAGE_MAX_WIDTH = 500;
var GLOBAL_SELECTED_POLYGON = null;
var GLOBAL_SELECTED_POLYGON_DIV = null;
var GLOBAL_WORKING_POLYGON = null;

var GLOBAL_SVG = null;
var GLOBAL_DATASET_NAME = null;
var GLOBAL_CLASS_IDX = null;
var GLOBAL_ANNOTATION_ID = null;
var GLOBAL_IMAGE_ID = null;

// TODO(ethan): add these functions to utils
// function zoomAndDrawPolygons(bbox, polygons) {
//     // remove existing polygons
//     $("polygon").remove();
//     let x = bbox[0];
//     let y = bbox[1];
//     let width = bbox[2];
//     let height = bbox[3];
//
//     // scale to desired size
//     let scalar = Math.round(GLOBAL_IMAGE_HEIGHT / height);
//     document.getElementById("my_image_container").style.width = Math.round(width * scalar).toString() + "px";
//     document.getElementById("my_image_container").style.height = Math.round(height * scalar).toString() + "px";
//     document.getElementById("my_image").style.transformOrigin = "top left";
//     let position = Math.round((-x * scalar)).toString() + "px, " + Math.round((-y * scalar)).toString() + "px";
//     document.getElementById("my_image").style.transform = "translate(" + position + ")" + " scale(" + scalar.toString() + ")";
//
//
//     for (let i = 0; i < polygons.length; i++) {
//         let polygon = document.createElementNS("http://www.w3.org/2000/svg", "polygon");
//         GLOBAL_SVG.appendChild(polygon);
//         for (let j = 0; j < polygons[i].length; j++) {
//             let point = GLOBAL_SVG.createSVGPoint();
//             // TODO(ethan): better handle this hard-coded dimension
//             point.x = (polygons[i][j][0] / 513.0) * (width * scalar);
//             point.y = (polygons[i][j][1] / 513.0) * (height * scalar);
//             polygon.points.appendItem(point);
//         }
//     }
// }


$(document).ready(function () {
    // set some global variables after document is ready
    GLOBAL_SVG = document.getElementById('theSVG');
    GLOBAL_DATASET_NAME = document.getElementById('dataset_name');
    GLOBAL_CLASS_IDX = document.getElementById('class_idx');
    GLOBAL_ANNOTATION_ID = document.getElementById('annotation_id');
    GLOBAL_IMAGE_ID = document.getElementById('image_id');

    // show the annotation
    showAnnotation();
});

async function showAnnotation() {
    let dataset = $(GLOBAL_DATASET_NAME).val();
    let class_idx = $(GLOBAL_CLASS_IDX).val();
    let annotation_id = $(GLOBAL_ANNOTATION_ID).val();
    let annotation_data = await getAnnotationData(dataset, class_idx, annotation_id);
    console.log(annotation_data);

    // show the image id
    let image_id = annotation_data["data"]["image_id"];
    document.getElementById("image_id").value = image_id;

    let image_filename = annotation_data["data"]["image_filename"];
    let image_url = GLOBAL_IMAGE_PREFIX + image_filename;
    let bbox = annotation_data["data"]["bbox"];
    let detPolygons = annotation_data["data"]["polygons"]; // TODO(ethan): maybe show ground truth if it exists
    let detExample = getZoomedAndCroppedAnnotationHTML(bbox, detPolygons, image_url);
    document.getElementById("AnnotationID").innerHTML = "";
    document.getElementById("AnnotationID").appendChild(detExample);
    console.log(detExample);
}

function addPolygonFromPoints(points, color) {
    let polygon = document.createElementNS("http://www.w3.org/2000/svg", "polygon");
    polygon.style.fill = color;
    polygon.style.stroke = color;
    GLOBAL_SVG.appendChild(polygon);
    for (let j = 0; j < points.length; j++) {
        let point = GLOBAL_SVG.createSVGPoint();
        // TODO(ethan): better handle this hard-coded dimension
        point.x = points[j][0];
        point.y = points[j][1];
        polygon.points.appendItem(point);
    }
}

function drawSegmentations(segmentations, color) {
    console.log(segmentations);
    let class_idx = $(GLOBAL_CLASS_IDX).val();
    let endpoint = getRevisedGlobalBackend(GLOBAL_BACKEND, class_idx.toString()) + "/rle_to_polygon";
    for (let i = 0; i < segmentations.length; i++) {
        let seg = segmentations[i].segmentation;
        console.log(seg);
        let req = {};
        req["rle"] = seg;
        $.ajax({
            url: endpoint,
            type: "POST",
            data: JSON.stringify(req),
            contentType: "application/json",
            dataType: "json"
        }).done(function (data) {
            console.log(data);
            let polygons = data.data;
            for (let j = 0; j < polygons.length; j++) {
                addPolygonFromPoints(polygons[j], color);
            }
        });
    }
}

function showFullImage() {
    console.log("showing full image");
    $("#QueryResult").html("");
    // make a request to get the annotation data
    let class_idx = $(GLOBAL_CLASS_IDX).val();
    let endpoint = getRevisedGlobalBackend(GLOBAL_BACKEND, class_idx.toString()) + "/images"
        + "/" + $(GLOBAL_DATASET_NAME).val()
        + "/" + $(GLOBAL_CLASS_IDX).val()
        + "/" + $(GLOBAL_IMAGE_ID).val();

    $.get(endpoint,
        function (data, textStatus, jqXHR) {
            console.log(data);
            $("polygon").remove();

            document.getElementById("my_image_container").style.width = null;
            document.getElementById("my_image_container").style.height = null;
            document.getElementById("my_image").style.transformOrigin = null;
            document.getElementById("my_image").style.transform = null;

            // draw the image
            let image_filename = data.data["image_filename"];
            let image_url = GLOBAL_IMAGE_PREFIX + image_filename;
            document.getElementById("my_image").src = image_url;

            // draw segmentations
            console.log("gt");
            drawSegmentations(data.data["gt_segmentations"], "red");

            console.log("detections");
            drawSegmentations(data.data["segmentations"], "green");
        });
}

