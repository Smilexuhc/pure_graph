# Experiments

## Setting:

**Dataset:**

1. Flickr

2. Reddit

3. PPI

4. PPI-large

5. yelp will be added soon

**Sampler:**

1. Random Walker sampler

2. Random Node sampler

3. Cluster

**GCN:**

1. Sage

2. GAT

## Flickr

batch_size=6000

sage num_hidden=256

gat num_heads=4 num_hidden=256



| train_type | eval_type | gcn_type | sampler | loss_norm | Best f1-micro | Best epoch | subgraph(mean) |
| :--------: | :-------: | :------: | :-----: | :-------: | :-----------: | :--------: | :------------: |
|    full    |   full    |   sage   |    -    |     -     |    0.5249     |    195     | (89250,899756) |
|    full    |   full    |   gat    |    -    |     -     |       -       |     -      |       -        |
|   sample   |   full    |   sage   |  rw-my  |   True    |    0.5163     |     12     | (14560,86300)  |
|   sample   |   full    |   sage   |  rw-my  |   False   |    0.5164     |     10     | (14500,86000)  |
|   sample   |   full    |   gat    |  rw-my  |   True    |    0.5130     |    107     | (14500,86000)  |
|   sample   |   full    |   gat    |  rw-my  |   False   |    0.4977     |     96     | (14500,86000)  |
|   sample   |  sample   |   sage   |  rw-my  |   True    |    0.5041     |     18     | (14500,86000)  |
|   sample   |  sample   |   sage   |  rw-my  |   False   |    0.5122     |     10     | (14500,86000)  |
|   sample   |  sample   |   gat    |  rw-my  |   True    |    0.5188     |     59     | (14500,86000)  |
|   sample   |  sample   |   gat    |  rw-my  |   False   |    0.5168     |     69     | (14500,86000)  |
|   sample   |   full    |   sage   |  rn-my  |   True    |    0.5039     |     24     | (17850,36000)  |
|   sample   |   full    |   sage   |  rn-my  |   False   |    0.5083     |     34     | (17850,36000)  |
|   sample   |   full    |   gat    |  rn-my  |   True    |    0.4813     |    119     | (17850,36000)  |
|   sample   |  sample   |   sage   |  rn-my  |   True    |    0.4783     |     30     | (17850,36000)  |
|   sample   |  sample   |   sage   |  rn-my  |   False   |    0.4887     |     38     | (17850,36000)  |
|   sample   |   full    |   sage   | cluster |     -     |    0.5163     |     41     | (17850,97000)  |
|   sample   |  sample   |   sage   | cluster |     -     |    0.5034     |     40     | (17850,97000)  |
|   sample   |   full    |   gat    | cluster |     -     |  **0.4662**   |    117     | (17850,97000)  |
|   sample   |  sample   |   gat    | cluster |     -     |  **0.4712**   |     79     | (17850,97000)  |

## reddit

batch_size=3000

num_heads=1

num_epochs=800


| train_type | eval_type | gcn_type | sampler | loss_norm | Best f1-micro | Best epoch |   subgraph(mean)    |
| :--------: | :-------: | :------: | :-----: | :-------: | ------------- | ---------- | :-----------------: |
|    full    |   full    |    -     |    -    |     -     | oom           | -          |  (232965,11606919)  |
|   sample   |  sample   |   sage   |  rw-my  |   True    | 0.9534        | 779        |    (8680,850000)    |
|   sample   |  sample   |   sage   |  rw-my  |   False   | 0.9714        | 743        |    (8680,850000)    |
|   sample   |  sample   |   gat    |  rw-my  |   True    | 0.9221        | 100        |    (8680,850000)    |
|   sample   |  sample   |   gat    |  rw-my  |   False   | 0.9495        | 84         |    (8680,850000)    |
|   sample   |  sample   |   sage   | cluster |     -     | 0.9381        | 176        | **（3100,185000）** |
|   sample   |  sample   |   gat    | cluster |     -     | 0.9370        | -          |    （2329, - ）     |
|   sample   |  sample   |   sage   | cluster |           | 0.9411        | -          |     （4660, -）     |
|   sample   |  sample   |   sage   | cluster |           | 0.9434        | -          |   (9320, 1100000)   |





## ppi

num_heads=4

num_epochs=1500


| train_type | eval_type | gcn_type | sampler | loss_norm | Best f1-micro | Best epoch | subgraph(mean)  |
| :--------: | :-------: | :------: | :-----: | :-------: | ------------- | ---------- | :-------------: |
|    full    |   full    |   sage   |    -    |     -     | 0.8129        | 1498       | (14755,225,270) |
|   sample   |   full    |   sage   |  rw-my  |   True    | 0.8717        | 1499       |  (8350,260000)  |
|   sample   |   full    |   sage   |  rw-my  |   False   | 0.8711        | 1482       |  (8350,260000)  |
|   sample   |   full    |   gat    |  rw-my  |   True    | **0.5610**    | 1470       |  (8350,260000)  |
|   sample   |   full    |   gat    |  rw-my  |   False   | **0.5492**    | 1468       |  (8350,260000)  |
|   sample   |  sample   |   sage   |  rw-my  |   True    | 0.8552        | 1468       |  (8350,260000)  |
|   sample   |  sample   |   sage   |  rw-my  |   False   | 0.8557        | 1489       |  (8350,260000)  |
|   sample   |   full    |   sage   |  rn-my  |   True    | 0.7222        | 1492       |  (8350,260000)  |
|   sample   |   full    |   sage   |  rn-my  |   False   | 0.8089        | 1484       |  (8350,260000)  |
|   sample   |  sample   |   sage   |  rn-my  |   True    | 0.8205        | 1489       |  (8350,260000)  |
|   sample   |  sample   |   sage   |  rn-my  |   False   | 0.8125        | 1498       |  (8350,260000)  |

## ppi-large 

num_epochs=800

| train_type | eval_type | gcn_type | sampler | loss_norm | Best f1-micro | Best epoch | subgraph(mean) |
| :--------: | :-------: | :------: | :-----: | :-------: | :-----------: | :--------: | :------------: |
|    full    |   full    |   sage   |    -    |     -     |    0.8771     |    799     | (56944,818716) |
|   sample   |   full    |   sage   |  rw-my  |   True    |    0.8946     |    794     | (13450,277000) |
|   sample   |   full    |   sage   |  rw-my  |   False   |    0.8942     |    793     | (13450,277000) |
|   sample   |  sample   |   sage   |  rw-my  |   True    |    0.8671     |    775     | (13450,277000) |
|   sample   |  sample   |   sage   |  rw-my  |   False   |    0.8732     |    771     | (13450,277000) |
|   sample   |   full    |   sage   | cluster |     -     |    0.9036     |    793     | (14236,270000) |
|   sample   |  sample   |   sage   | cluster |     -     |    0.8892     |    724     | (14236,270000) |





## Aggr type

dataset=flickr, sampler=rw-my

| aggr_type | train_type | eval_type | norm  | Best f1-micro | Best epoch |
| --------- | ---------- | --------- | ----- | ------------- | ---------- |
| max       | full       | full      | -     | 0.5202        | 118        |
| max       | sample     | full      | True  | **0.4371**    | 1          |
| max       | sample     | full      | False | 0.5249        | 25         |
| add       | full       | full      | -     | **0.4658**    | 46         |
| add       | sample     | full      | True  | **0.4544**    | 116        |
| add       | sample     | full      | False | **0.4680**    | 119        |
| mean      | full       | full      | -     | 0.5211        | 114        |
| mean      | sample     | full      | True  | **0.4611**    | 5          |
| mean      | sample     | full      | False | 0.5179        | 10         |



dataset=ppi, sampler=rw-my

| aggr_type | train_type | eval_type | norm  | Best f1-micro | Best epoch |
| --------- | ---------- | --------- | ----- | ------------- | ---------- |
| max       | full       | full      | -     | 0.8964        | 1494       |
| max       | sample     | full      | True  | **0.3984**    | 61         |
| max       | sample     | full      | False | 0.9401        | 1398       |
| add       | full       | full      | -     | **0.7120**    | 1169       |
| add       | sample     | full      | True  | **0.4882**    | 1491       |
| add       | sample     | full      | False | 0.8502        | 1491       |
| mean      | full       | full      | -     | 0.8126        | 1497       |
| mean      | sample     | full      | True  | **0.3686**    | 0          |
| mean      | sample     | full      | False | 0.8707        | 1482       |

2329 0.9370

4660 0.9411

9320 0.9434

## Conclusion

1. full evaluate > sampling evaluate
2. random walk sampler/ cluster sampler > random node sampler 
3. No normalization > use normalization, when using sampling evaluating
4. No normalization $\approx $ use normalization, when using full evaluating
5. cluster sampler < rw sampler when sub graph size equals
6. sub graph size matters