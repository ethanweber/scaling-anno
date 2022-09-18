// some global variables
var GLOBAL_IMAGE_PREFIX = "https://wednesday.ethanweber.me/ethan/polytransform_external/ade_places_data/";
var GLOBAL_BACKEND = "https://adebackend0.ethanweber.me";
var GLOBAL_IMAGE_HEIGHT = 100;
var GLOBAL_IMAGE_WIDTH = 100;
var NUMBER_SHOWN_PER_CLUSTER = 20;

var GLOBAL_SVG = null;
var GLOBAL_DATASET_NAME = null;
var GLOBAL_CLASS_IDX = null;
var GLOBAL_CLUSTER_ID = null;
var GLOBAL_CHILDREN = null;
var GLOBAL_PARENT = null;


$(document).ready(function () {
    // set some global variables after document is ready
    GLOBAL_SVG = document.getElementById('theSVG');
    GLOBAL_DATASET_NAME = document.getElementById('dataset_name');
    GLOBAL_CLASS_IDX = document.getElementById('class_idx');
    GLOBAL_CLUSTER_ID = document.getElementById('cluster_id');
});

// open the image in the dataset viewer tab
function clickCallback(segmentID) {
    console.log("clicked me");
    // console.log(segmentID);


    let url = "/dataset"
        + "?dataset_name=" + $(GLOBAL_DATASET_NAME).val()
        + "&class_idx=" + $(GLOBAL_CLASS_IDX).val()
        + "&annotation_id=" + segmentID.toString();

    window.open(url, '_blank');
}

// Draw the segment ids in the holder div.
function drawSegmentIds(segmentIds, holderDiv) {
    let dataset_name = $(GLOBAL_DATASET_NAME).val();
    let class_idx = $(GLOBAL_CLASS_IDX).val();
    for (let i = 0; i < segmentIds.length; i++) {
        drawAnnotationInDiv(segmentIds[i], holderDiv, segmentIds[i].toString(), dataset_name, class_idx, clickCallback);
    }
}

// Handle keypresses for quick navigation of interface.
document.onkeydown = function (e) {
    switch (e.keyCode) {
        case 37:
            // left
            if (GLOBAL_CHILDREN != null) {
                setPanelsWithClusters(GLOBAL_CHILDREN[0]);
            }
            break;
        case 38:
            // up
            if (GLOBAL_PARENT != null) {
                setPanelsWithClusters(GLOBAL_PARENT);
            }
            break;
        case 39:
            // right
            if (GLOBAL_CHILDREN != null) {
                setPanelsWithClusters(GLOBAL_CHILDREN[1]);
            }
            break;
    }
};

$(document).ready(function () {
    // run function to update with current data
    updateClusterInfo();
});

// On button press.
function updateClusterInfo() {
    let cluster_id = $(GLOBAL_CLUSTER_ID).val();
    setPanelsWithClusters(cluster_id);
}

// Clear everything.
function clearEverything() {

    // stop outgoing requests
    window.stop();

    // clear dendogram
    $("#my_dataviz").html("");

    // clear the divs
    $("#topInfo").html("");
    $("#topInfo").html("");

    $("#leftChild").html("");
    $("#rightChild").html("");

    $("#leftInfo").html("");
    $("#rightInfo").html("");

}


async function setPanelsWithClusters(cluster_id) {
    console.log("let's show some images!");

    clearEverything();


    // start the dendogram
    setupDendogram(cluster_id);

    $(GLOBAL_CLUSTER_ID).val(cluster_id);

    GLOBAL_PARENT = await getParentFromCluster(cluster_id);

    console.log("GLOBAL_PARENT");
    console.log(GLOBAL_PARENT);

    GLOBAL_CHILDREN = await getChildrenFromCluster(cluster_id);
    if (GLOBAL_CHILDREN.length == 0) {
        GLOBAL_CHILDREN = null;
        return;
    }
    let [leftSegmentIds, rightSegmentIds] = await Promise.all([
        getSegmentIds(GLOBAL_CHILDREN[0]),
        getSegmentIds(GLOBAL_CHILDREN[1])
    ]);

    // console.log("leftSegmentIds");
    // console.log(leftSegmentIds);

    let [leftSize, rightSize] = await Promise.all([
        getSize(GLOBAL_CHILDREN[0]),
        getSize(GLOBAL_CHILDREN[1])
    ]);

    // get purites for cluster ids
    // responses are [purity, num responses]
    let [currentPurityResponse, leftChildPurityResponse, rightChildPurityResponse] = await Promise.all([
        getPurityFromCluster(cluster_id),
        getPurityFromCluster(GLOBAL_CHILDREN[0]),
        getPurityFromCluster(GLOBAL_CHILDREN[1])
    ]);
    // // TODO(ethan): note that this will be NaN in the case of having a response of None (null)
    // currentPurityResponse[0] = getRoundedNumber(currentPurityResponse[0], 2);
    // leftChildPurityResponse[0] = getRoundedNumber(leftChildPurityResponse[0], 2);
    // rightChildPurityResponse[0] = getRoundedNumber(rightChildPurityResponse[0], 2);

    let [currentQuality, leftChildQuality, rightChildQuality] = await Promise.all([
        getQualityFromCluster(cluster_id),
        getQualityFromCluster(GLOBAL_CHILDREN[0]),
        getQualityFromCluster(GLOBAL_CHILDREN[1])
    ]);
    currentQuality = getRoundedNumber(currentQuality, 2);
    leftChildQuality = getRoundedNumber(leftChildQuality, 2);
    rightChildQuality = getRoundedNumber(rightChildQuality, 2);


    // set some info about the clusters
    $("#topInfo").append("<span>Cluster ID: " + cluster_id.toString() + "</span>");
    $("#topInfo").append("<span>Num leaves: " + (leftSize + rightSize).toString() + "</span>");
    $("#topInfo").append("<span>Quality estimate: " + currentQuality.toString() + "</span>");
    $("#topInfo").append("<span>Num questions: " + currentPurityResponse[1].toString() + "</span>");

    $("#leftInfo").append("<span>Cluster ID: " + GLOBAL_CHILDREN[0].toString() + "</span>");
    $("#leftInfo").append("<span>Num leaves: " + leftSize.toString() + "</span>");
    $("#leftInfo").append("<span>Quality estimate: " + leftChildQuality.toString() + "</span>");
    $("#leftInfo").append("<span>Num questions: " + leftChildPurityResponse[1].toString() + "</span>");

    $("#rightInfo").append("<span>Cluster ID: " + GLOBAL_CHILDREN[1].toString() + "</span>");
    $("#rightInfo").append("<span>Num leaves: " + rightSize.toString() + "</span>");
    $("#rightInfo").append("<span>Quality estimate: " + rightChildQuality.toString() + "</span>");
    $("#rightInfo").append("<span>Num questions: " + rightChildPurityResponse[1].toString() + "</span>");

    // only show some images in the cluster
    leftSegmentIds = getRandomSubarray(leftSegmentIds, NUMBER_SHOWN_PER_CLUSTER);
    rightSegmentIds = getRandomSubarray(rightSegmentIds, NUMBER_SHOWN_PER_CLUSTER);

    drawSegmentIds(leftSegmentIds, "leftChild");
    drawSegmentIds(rightSegmentIds, "rightChild");
}
