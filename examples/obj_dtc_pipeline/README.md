# Object Detection Pipeline Example

## Prepare Image to detect object

```bash
# From [`Penn-Fudan Database for Pedestrian Detection and Segmentation`](https://www.cis.upenn.edu/~jshi/ped_html/)
$ wget https://www.cis.upenn.edu/~jshi/ped_html/PennFudanPed.zip
$ unzip PennFudanPed.zip
# Used at Torchvision Example
$ cp PennFudanPed/PNGImages/FudanPed00054.png FudanPed00054.png
# (optional) Validate python backend code
$ python models.py --input ./FudanPed00054.png --output ./result.png
```
