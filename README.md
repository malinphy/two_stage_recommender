# Two stage recommender
Overview :<br/>
Tensorflow/Keras implementation of two stage recommender system
![image](https://user-images.githubusercontent.com/55249305/198822998-be0cdb8d-a18b-4040-b91a-05c5ebbedab9.png)


<!-- </br> -->
Two- stage recommendation systems are commonly used in industy
because of scalability. In general, these systems composed one
retrieval part which reduce number of candidate items from all
possible items in a very large dataset and a ranking part or 
reranker to fine tune and sort as end recommendations.

Our system composed two neural networks which serve as retrieval
stage and ranking stage.

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

