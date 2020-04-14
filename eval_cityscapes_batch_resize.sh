#run this script at the top of the project 
mkdir datasets/cityscapes_test
convert "datasets/leftImg8bit/val/frankfurt_test/*.png" -set filename:base "%[basename]" -resize 256x256! "datasets/cityscapes_test/%[filename:base].png"
