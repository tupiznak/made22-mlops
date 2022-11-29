cd docker/base || exit
docker build -t made22-mlops-hw3-base:1.0 .
cd ../generate || exit
docker build -t made22-mlops-hw3-generate:1.0 .
cd ../train || exit
docker build -t made22-mlops-hw3-train:1.0 .
cd ../predict || exit
docker build -t made22-mlops-hw3-predict:1.0 .
