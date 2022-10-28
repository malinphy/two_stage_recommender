# Two stage recommender
Overview :<br/>
Tensorflow/Keras implementation of two stage recommender system
![image](https://user-images.githubusercontent.com/55249305/198729391-48e8c510-6f1c-446a-8967-282baae3d06e.png)

<!-- </br> -->
During the learning process, each question and answer were vectorized using the same Universal sentence encoder.<br/> Additional negative sample generated for each question and answer pair. Questions, answers and negative sentences were trained using siamese-BERT structure with triplet ranking loss. 
Questions, answers and negative sentences were treated as anchor, positive and negative terms respectively. Purpose of this method is  minimizing the distance between anchor and positive pairs, while maximizing the distance between anchor and negative pairs.
<!-- <br/> -->After training, questionas and corresponding answers were vectorized using trained sentence encoder. 
<!-- <br/> -->
Vector values of each answer, loaded in vector similarity search library (scaNN).
<!-- <br/> -->
In prediction process, test question is vectorized using trained sentence encoder and using the vector similarity search, closest answer will be returned. User, does not have query exactly same question but another question can be used with the same or close meaning. Trained sentence encoder will vectorize the question and generate the close values to exact question in vector space. Sample process can be seen in example.

Data :<br/>
----

ELI 5 (from reddit forum explain like I am 5) Dataset : 
https://facebookresearch.github.io/ELI5/download.html
<br/>
for simplicity dataset hugginface datasets :
https://huggingface.co/datasets/eli5

File Description :
----
- data_loader.py : loads the dataset and splits questions and answers
- HelperFunctions.py : Data preparation for model training
- model.py : nueralnetwork model written with tensoflow/keras
- negative_maker.py : generation of random negative samples
- train.py : training file
- prediction.py : prediction file for deployment purpose
- sent_bert_mnr_cli5_v2.ipynb : notebook that is suitable for colab.





EVALUATION
----------
```
              |@1    |@3    |@5    |@10   |@20   |
|-------------|------|------|------|------|------|
|precision@K  |0.2376|0.1210|0.0924|0.0557|0.0321|
|-------------|------|------|------|------|------|
|recall@K     |0.2376|0.3899|0.4618|0.5572|0.6421|   
```
EXAMPLE
----------

TEST QUESTION : What's the difference between a bush, a shrub, and a tree?
<br />
TEST ANSWER : Shrubs and trees are both specifically *woody* plants with stems that survive throughput the winter. A tree has a clear central trunk whereas a shrub has multiple stems rising from the ground.'Bush' is a more general term for any plant with multiple stems rising from the ground, and that can be either woody or what's called herbaceous, herbaceous plants are ones where the stems die back completely or substantially in the winter leaving the plant with just its roots and new stems grow next spring.
<br />
ALTERNATIVE QUESTION: 'Is there any difference between a bush, a shrub, and a tree?'

