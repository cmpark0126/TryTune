# Object Detection Pipeline Example

Make sub images from original image using object detection and cropping pipeline.
Cropping can be placed into trytune by auto gen function in the future.
However, now we just use built-in module of trytune here.

## Prepare Image to detect object

```bash
# From [`Penn-Fudan Database for Pedestrian Detection and Segmentation`](https://www.cis.upenn.edu/~jshi/ped_html/)
$ wget https://www.cis.upenn.edu/~jshi/ped_html/PennFudanPed.zip
$ unzip PennFudanPed.zip
# Used at Torchvision Example
$ cp PennFudanPed/PNGImages/FudanPed00054.png FudanPed00054.png
```

### (Optional) Test Triton Python Backend code work well
```bash
# Currently, we are on /obj_dtc_pipeline directory
$ vi ./model.py
# Comment out `import triton_python_backend_utils as pb_utils`
# Uncomment from `# To simulate triton python backend code without installing triton` to end of file
$ python model.py --input ./FudanPed00054.png --output ./result.png
```

## Launch TryTune & Triton Inference Server with `model.py`

TODO

## Execute `client.py` to send request and obtain response from trytune

TODO (response can be converted to multiple images)
TODO (add pytest example also)
