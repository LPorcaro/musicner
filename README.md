# Recognizing Musical Entities in User-generated Content

We present a novel method for detecting musical entities from user-generated content, modelling linguistic features with statistical models and extracting contextual information from a radio schedule.  We analyzed tweets related to a classical music radio station, integrating its schedule to connect users' messages to tracks broadcasted. 

This repository contains code to reproduce the results of our [arXiv paper](https://arxiv.org/abs/1904.00648).

#### Reference:
> Lorenzo Porcaro, Horacio Saggion (2019). Recognizing Musical Entities in User-generated Content. Paper presented at the International Conference on Computational Linguistics and Intelligent Text Processing (CICLing) 2019, University of La Rochelle, La Rochelle, 7-13 April.


## Reproduce our results

#### Installation:
Create a python 2.7 (sorry!) virtual environment and install dependencies `pip install -r src/requirements.txt`

#### Update config file:
Update the file `etc/config.yaml`, insert your consumer key, consumer secret, access token, access secret from the Twitter API. More info about the API: https://developer.twitter.com/

#### Import data:
To receive the data for reproducing the experiment, please contact `lorenzo.porcaro at gmail.com`. Once received, go to the data [README](https://github.com/LPorcaro/musicner/tree/master/data) page for more info 
