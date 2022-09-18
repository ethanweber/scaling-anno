// utilities for drawing polygons, editing, hitting the server, etc.

// Get segmentation ids of a cluster.
async function getSegmentIds(cluster_id) {
    return new Promise(function (resolve, reject) {
        let endpoint = "/segment_ids"
            + "/" + $(GLOBAL_DATASET_NAME).val()
            + "/" + $(GLOBAL_CLASS_IDX).val()
            + "/" + cluster_id.toString();
        $.get(endpoint,
            function (data, textStatus, jqXHR) {
                resolve(data);
            });
    });
}

// Get the total size a cluster.
async function getSize(cluster_id) {
    return new Promise(function (resolve, reject) {
        let endpoint = "/size"
            + "/" + $(GLOBAL_DATASET_NAME).val()
            + "/" + $(GLOBAL_CLASS_IDX).val()
            + "/" + cluster_id.toString();
        $.get(endpoint,
            function (data, textStatus, jqXHR) {
                resolve(data);
            });
    });
}

// Get the parent cluster.
async function getParentFromCluster(cluster_id) {
    return new Promise(function (resolve, reject) {
        let endpoint = "/parent"
            + "/" + $(GLOBAL_DATASET_NAME).val()
            + "/" + $(GLOBAL_CLASS_IDX).val()
            + "/" + cluster_id.toString();
        $.get(endpoint,
            function (data, textStatus, jqXHR) {
                resolve(data);
            });
    });
}

// Get the children cluster ids.
async function getChildrenFromCluster(cluster_id) {
    return new Promise(function (resolve, reject) {
        let endpoint = "/children"
            + "/" + $(GLOBAL_DATASET_NAME).val()
            + "/" + $(GLOBAL_CLASS_IDX).val()
            + "/" + cluster_id.toString();
        $.get(endpoint,
            function (data, textStatus, jqXHR) {
                resolve(data);
            });
    });
}

// Get purity from cluster id.
async function getPurityFromCluster(cluster_id) {
    return new Promise(function (resolve, reject) {
        let endpoint = "/purity"
            + "/" + $(GLOBAL_DATASET_NAME).val()
            + "/" + $(GLOBAL_CLASS_IDX).val()
            + "/" + cluster_id.toString()
            + "/" + "6"; // TODO(ethan): make this a parameter, to select the round number of purity
        $.get(endpoint,
            function (data, textStatus, jqXHR) {
                resolve(data);
            });
    });
}

// Get quality from cluster id.
async function getQualityFromCluster(cluster_id) {
    return new Promise(function (resolve, reject) {
        let endpoint = "/quality"
            + "/" + $(GLOBAL_DATASET_NAME).val()
            + "/" + $(GLOBAL_CLASS_IDX).val()
            + "/" + cluster_id.toString()
            + "/" + "6"; // TODO(ethan): make this a parameter, to select the round number of purity
        $.get(endpoint,
            function (data, textStatus, jqXHR) {
                resolve(data);
            });
    });
}

// Get the config for a HIT.
async function getConfigFromConfigTypeAndConfigName(config_type, config_name) {
    return new Promise(function (resolve, reject) {
        console.log(config_name);
        let endpoint = "/hits"
            // + "/" + config_type
            + "/" + GLOBAL_ROUND_NUMBER
            + "/" + config_name;
        $.get(endpoint,
            function (data, textStatus, jqXHR) {
                resolve(data);
            });
    });
}

// Get the config for a HIT.
async function getResultsFromConfigTypeAndConfigName(config_type, config_name) {
    return new Promise(function (resolve, reject) {
        console.log(config_name);
        let endpoint = "/get_responses"
            // + "/" + config_type
            + "/" + GLOBAL_ROUND_NUMBER
            + "/" + config_name;
        $.get(endpoint,
            function (data, textStatus, jqXHR) {
                resolve(data);
            });
    });
}

// Get the pickn choices.
async function getPicknChoicesFromConfigName(config_name) {
    return new Promise(function (resolve, reject) {
        console.log(config_name);
        let endpoint = "/hits"
            + "/" + "pickn_choices"
            + "/" + GLOBAL_ROUND_NUMBER
            + "/" + config_name;
        $.get(endpoint,
            function (data, textStatus, jqXHR) {
                resolve(data);
            });
    });
}

// Get polygon from RLE.
async function getPolygonsFromRLE(rle) {
    return new Promise(function (resolve, reject) {
        let endpoint = GLOBAL_BACKEND + "/rle_to_polygon";
        let req = {};
        req["rle"] = rle;
        $.ajax({
            url: endpoint,
            type: "POST",
            data: JSON.stringify(req),
            contentType: "application/json",
            dataType: "json"
        }).done(function (data) {
            let polygons = data.data;
            resolve(polygons);
        });
    });
}

// Get IoU from polygons.
async function getIoUFromPolygons(polygons1, polygons2, height, width) {
    return new Promise(function (resolve, reject) {
        let endpoint = GLOBAL_BACKEND + "/iou_from_polygons";
        let req = {};
        req["polygons1"] = polygons1;
        req["polygons2"] = polygons2;
        req["height"] = height;
        req["width"] = width;
        $.ajax({
            url: endpoint,
            type: "POST",
            data: JSON.stringify(req),
            contentType: "application/json",
            dataType: "json"
        }).done(function (data) {
            let iou = data.data;
            resolve(iou);
        });
    });
}

// Get IoU from RLE encodings.
async function getIoUFromRLEs(rle1, rle2) {
    let height = rle1["size"][0];
    let width = rle1["size"][1];
    let [polygons1, polygons2] = await Promise.all([
        getPolygonsFromRLE(rle1),
        getPolygonsFromRLE(rle2)
    ]);
    let iou = await getIoUFromPolygons(polygons1, polygons2, height, width);
    return iou;
}

// Return the HTML with the nicely formatted, zoomed in, annotation. Polygons on top. :)
function getZoomedAndCroppedAnnotationHTML(bbox, polygons, image_url) {

    // pull out the bounding box data
    let x = bbox[0];
    let y = bbox[1];
    let width = bbox[2];
    let height = bbox[3];

    let container = document.createElement('div');
    container.setAttribute("class", "img-overlay-wrap");
    let image = document.createElement('img');
    let svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    container.appendChild(image);
    container.appendChild(svg);

    // set the image url
    image.src = image_url;

    // scale to desired size
    let y_scalar = GLOBAL_IMAGE_HEIGHT / height;
    let x_scalar = y_scalar;
    if (GLOBAL_IMAGE_WIDTH != null) {
        x_scalar = GLOBAL_IMAGE_WIDTH / width;
    } else if (GLOBAL_IMAGE_MAX_WIDTH != null) {
        if (width * x_scalar > GLOBAL_IMAGE_MAX_WIDTH) {
            x_scalar = GLOBAL_IMAGE_MAX_WIDTH / width;
            y_scalar = x_scalar;
        }
    }

    container.style.width = Math.round(width * x_scalar).toString() + "px";
    container.style.height = Math.round(height * y_scalar).toString() + "px";
    image.style.transformOrigin = "top left";
    let position = Math.round((-x * x_scalar)).toString() + "px, " + Math.round((-y * y_scalar)).toString() + "px";
    image.style.transform = "translate(" + position + ")" + " scale(" + x_scalar.toString() + "," + y_scalar.toString() + ")";

    for (let i = 0; i < polygons.length; i++) {
        let polygon = document.createElementNS("http://www.w3.org/2000/svg", "polygon");
        svg.appendChild(polygon);
        for (let j = 0; j < polygons[i].length; j++) {
            let point = svg.createSVGPoint();
            point.x = (polygons[i][j][0] - x) * x_scalar;
            point.y = (polygons[i][j][1] - y) * y_scalar;
            polygon.points.appendItem(point);
        }
    }
    let outercontainer = document.createElement('div');
    outercontainer.appendChild(container);
    return outercontainer;
}

// Draw the annotation (polygon) in the specified divID.
// holderID is where the new annotation will be appended (added).
// newDivID is the newly created id of the annotation.
function drawAnnotationInDiv(annotation_id, holderID, newDivID, dataset, class_idx, onclickCallback = null) {
    // make a request to get the annotation data
    let endpoint = getRevisedGlobalBackend(GLOBAL_BACKEND, class_idx.toString())
        + "/annotations"
        + "/" + dataset
        + "/" + class_idx.toString()
        + "/" + annotation_id.toString()
    $.get(endpoint,
        function (data, textStatus, jqXHR) {
            let image_filename = data.data["image_filename"];
            let image_url = GLOBAL_IMAGE_PREFIX + image_filename;
            let bbox = data.data["bbox"];
            let polygons = data.data["polygons"];
            let annotationHTML = getZoomedAndCroppedAnnotationHTML(bbox, polygons, image_url);
            annotationHTML.setAttribute("id", newDivID);
            annotationHTML.setAttribute("class", "TreeDiv");
            annotationHTML.addEventListener('click', function () {
                if (onclickCallback != null) {
                    onclickCallback(newDivID);
                }
            });
            document.getElementById(holderID).appendChild(annotationHTML);
        });
}


// Get the annotation as a div.
async function getAnnotationAsDiv(dataset, class_idx, annotation_id) {
    return new Promise(function (resolve, reject) {
        // make a request to get the annotation data
        let endpoint = getRevisedGlobalBackend(GLOBAL_BACKEND, class_idx.toString())
            + "/annotations"
            + "/" + dataset
            + "/" + class_idx.toString()
            + "/" + annotation_id.toString();
        // console.log(endpoint);
        $.get(endpoint,
            function (data, textStatus, jqXHR) {
                let image_filename = data.data["image_filename"];
                let image_url = GLOBAL_IMAGE_PREFIX + image_filename;
                let bbox = data.data["bbox"];
                let polygons = data.data["polygons"];
                // let polygons = await getPolygonsFromRLE(data.data["segmentation"]["segmentation"]);
                let annotationHTML = getZoomedAndCroppedAnnotationHTML(bbox, polygons, image_url);
                resolve(annotationHTML);
            });
    });
}

// Get the pickn annotation as a div.
async function getPicknAnnotationAsDiv(dataset, class_idx, annotation_id, choices) {
    return new Promise(function (resolve, reject) {
        // make a request to get the annotation data
        let endpoint = getRevisedGlobalBackend(GLOBAL_BACKEND, class_idx.toString())
            + "/annotations"
            + "/" + dataset
            + "/" + class_idx.toString()
            + "/" + annotation_id.toString();
        $.get(endpoint,
            function (data, textStatus, jqXHR) {
                let image_filename = data.data["image_filename"];
                let image_url = GLOBAL_IMAGE_PREFIX + image_filename;
                let bbox = data.data["bbox"];
                // let polygons = data.data["polygons"];
                let annotationHTML = document.createElement('div');
                annotationHTML.setAttribute("class", "PickNRow");
                for (let j = 0; j < choices.length; j++) {
                    let rowItem = document.createElement('div');
                    rowItem.setAttribute("class", "PickNRowItem");
                    let choice_polygons = choices[j]["polygons"];
                    rowItem.innerHTML = getZoomedAndCroppedAnnotationHTML(bbox, choice_polygons, image_url).innerHTML;

                    rowItem.innerHTML += "<div>" + j.toString() + "</div>"
                    // outercontainer.innerHTML += new_example.innerHTML;
                    annotationHTML.appendChild(rowItem);
                }
                // also add a button for no choices
                let buttonDiv = document.createElement('div');
                buttonDiv.innerHTML = document.getElementById("picknNoneButtonTemplate").innerHTML;
                annotationHTML.appendChild(buttonDiv);


                resolve(annotationHTML);
            });
    });
}

// Get annotation data.
async function getAnnotationData(dataset, class_idx, annotation_id) {
    return new Promise(function (resolve, reject) {
        // make a request to get the annotation data
        let endpoint = getRevisedGlobalBackend(GLOBAL_BACKEND, class_idx.toString())
            + "/annotations"
            + "/" + dataset
            + "/" + class_idx.toString()
            + "/" + annotation_id.toString();
        $.get(endpoint,
            function (data, textStatus, jqXHR) {
                resolve(data);
            });
    });
}

// Get a random subarray of a list.
// https://stackoverflow.com/questions/11935175/sampling-a-random-subset-from-an-array
function getRandomSubarray(arr, size) {
    let shuffled = arr.slice(0), i = arr.length, temp, index;
    while (i--) {
        index = Math.floor((i + 1) * Math.random());
        temp = shuffled[index];
        shuffled[index] = shuffled[i];
        shuffled[i] = temp;
    }
    return shuffled.slice(0, size);
}

function getClassNameFromClassIdx(class_idx) {
    return GLOBAL_CLASS_IDX_TO_CLASS_NAME[class_idx];
}

function getRoundedIoU(iou) {
    return Math.round(iou * 100) / 100;
}

function getHTMLCopyOfElementID(element_id) {
    let tempdiv = document.createElement("div");
    tempdiv.appendChild(document.getElementById(element_id));
    return tempdiv;
}

function deleteAllElementsOfTagName(tagName) {
    let htmlCollection = document.getElementsByTagName(tagName);
    let arr = Array.from(htmlCollection);
    arr.map(el => el.parentNode.removeChild(el));
}

/* Randomize array in-place using Durstenfeld shuffle algorithm */

// https://stackoverflow.com/questions/2450954/how-to-randomize-shuffle-a-javascript-array
function shuffleArray(array) {
    for (let i = array.length - 1; i > 0; i--) {
        let j = Math.floor(Math.random() * (i + 1));
        let temp = array[i];
        array[i] = array[j];
        array[j] = temp;
    }
}

function getRevisedGlobalBackend(global_backend, class_idx) {
    let class_idx_as_int = parseInt(class_idx);
    let bucket = Math.floor(class_idx_as_int / 20);
    // NOTE: this trick was used to load balance when using many machines
    let revised_global_backend = global_backend.replace("adebackend0", "adebackend" + bucket.toString());
    // let revised_global_backend = global_backend
    // console.log(revised_global_backend)
    return revised_global_backend;
}

// https://www.jacklmoore.com/notes/rounding-in-javascript/
function getRoundedNumber(value, decimals) {
    return Number(Math.round(value + 'e' + decimals) + 'e-' + decimals);
}
