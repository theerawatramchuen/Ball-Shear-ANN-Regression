## Reference web 
https://machinelearningmastery.com/a-practical-guide-to-deploying-machine-learning-models/?fbclid=IwY2xjawGRGGtleHRuA2FlbQIxMAABHbCoCgWAAbc8P6bja4oBr99FIy-kloGQRiOwAH2YbBlUASZTcEnZpQ0AgA_aem_buaLfZ-zW4X46sutHyPxYw&sfnsn=mo
### Build docker image
docker build -t testuph-api .
### Run docker image
docker run -d -p 80:80 testuph-api
