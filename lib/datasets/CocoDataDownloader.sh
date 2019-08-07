# get COCO dataset
mkdir coco/
cd coco/

mkdir coco/images
mkdir coco/annotations

wget http://images.cocodataset.org/zips/train2017.zip
wget http://images.cocodataset.org/zips/val2017.zip
wget http://images.cocodataset.org/zips/test2017.zip
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip

unzip annotations_trainval2017.zip -d coco/annotations/

unzip train2017.zip -d coco/images
unzip val2017.zip -d coco/images
unzip test2017.zip -d coco/images

rm -f annotations_trainval2017.zip
rm -f train2017.zip
rm -f val2017.zip
rm -f test2017.zip
