# Two Stage Recommender System:
Overview :
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
Retrieval part is treated as a next item prediction problem with
softmax probability. Purpose of this part learning user and item
embeddings as a function of user's history. Item embeddings were
generated from the weights of the softmax layer. User embeddings
were extracted from the dense layer before the softmax layer.
Dot product of user and item embeddings have been used to determine
top-N recommendations. From this point of view, final step of 
retrieval stage can be considered as non-linear matrix factorization
method. 

During the serving process, item embeddings stored in vector 
similarity search library (ScaNN). User embeddings generated through 
the retrieval model treated as query embeddings. By using the ScaNN, 
the closest vectors to the query embeddings can be extracted via 
dot product. 

Ranking part designed quite similar to the retrieval part. 
At this part, outputs of the retrieval part used as inputs
and most of the features come from the item properties. As final
layer of the ranking stage, sigmoid function were employed and 
output of the sigmoid function considered as probabilities of
the items. According to calculated probabilites, items were 
sorted and served to user.

Total arcihecture of the system can be seen in picture. 
For detailed information following study can be investigated (https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/45530.pdf)

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

