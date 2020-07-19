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


|  dataset  | sampler | eval |     f1-micro     |    subgraph     |
| :-------: | :-----: | :--: | :--------------: | :-------------: |
|  flickr   |  rw-my  | full |      0.5163      |  (14500,86000)  |
|  flickr   | cluster | full |      0.5034      |  (17850,97000)  |
|  flickr   |   gec   | full |      0.4740      |  (11000,23400)  |
|  flickr   | gec+rw  | full |      0.5120      |        (23200,142700)        |
|  reddit   |  rw-my  | eval |      0.9534      |  (8680,850000)  |
|  reddit   | cluster | eval |      0.9434      | (9320, 1100000) |
|  reddit   |   gec   | eval |      0.9242      |       (3000,700000)       |
|    ppi    |  rw-my  | full |  0.9264  |  (3800,89550)  |
|    ppi    | cluster | full |      -      | (14755, 450540) |
|    ppi    |   gec   | full | **0.9472** | (3688,112500) |
| ppi-large |  rw-my  | full |      0.8946      | (13450,277000)  |
| ppi-large | cluster | full |      0.9036      | (14236,270000)  |
| ppi-large |   gec   | full |    **0.9423**    |  (5700,160000)  |
| yelp | gec | eval | 0.6257 | (3000,30000) |
| yelp | rw-my | eval | 0.6331 | (2500,12500) |

