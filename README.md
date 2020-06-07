Test method of sampling graph and gcn model on pure graph task


Example of running  
Run graphsaint model on flicker dataset with sampling and loss normalization.

```shell script
python main.py --dataset='flickr' --train_sample=1 --loss_norm=1
```

Dataset support:
1. Flickr
2. Reddit
3. Yelp
4. PPI


TODO:
1. Tensorboard for visualization
2. More sampler and more gcn net.
3. Rewrite dataset