# Run this script at the top of the project, where cityscapes is expected to contain original the cityscapes dataset
mkdir ../cityscapes-converted
mogrify -resize 256x256! -path ../cityscapes-converted/leftImg8bit/val/frankfurt/ ../cityscapes/leftImg8bit/val/frankfurt/*.png
mogrify -resize 256x256! -path ../cityscapes-converted/leftImg8bit/val/lindau/ ../cityscapes/leftImg8bit/val/lindau/*.png
mogrify -resize 256x256! -path ../cityscapes-converted/leftImg8bit/val/munster/ ../cityscapes/leftImg8bit/val/munster/*.png
