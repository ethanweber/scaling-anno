# annotate

# TODO(ethan): use .json instead 
```yaml
TYPE: <binary/pickN/fixPolygon>
  QUALIFICATION_TEST:
    NUM_IMAGES: 10
HIT:
  QUALITY_CONTROL:
    NUM_IMAGES: 5
  DATASET: "places.json"
  CLASS: 9
  ANNOTATION_IDS: [100,39,80,2]
```
# components

- server.py - used for the main task and HITs
- backend.py - used for return images/annotations/data/etc. very quickly
