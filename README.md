#this is for AWS EC@ ubuntu server 18.04
#connect to ubuntu server

$ curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
$ sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable"
$ sudo apt-get update
$ sudo apt-get install -y docker-ce
$ sudo usermod -aG docker ${USER}
$ sudo apt-get install git
$ git clone https://github.com/CheeHau86/barcodeDetector.git && cd barcodeDetector
$ docker build -t barcode-docker:0.0.1 .
$ docker run -p 8080:8080 barcode-docker:0.0.1
