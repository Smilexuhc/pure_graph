Test method of sampling graph and gcn model on pure graph task


Example of running  
Run graphsaint model on flicker dataset with sampling and loss normalization.

```shell script
CUDA_VISIBLE_DEVICES=1 python main.py --dataset='reddit' --eval_sample=1 --batch_size=3000
```

Dataset support:
1. Flickr
2. Reddit
3. Yelp(exist error)
4. PPI(exist error)

Sample support:
1. Graphsaint random walker sampler.
2. Random node sampler.
3. 


TODO 06-07:
1. Fix bias in sampling evaluation. Done
2. Tensorboard for visualization.
3. More sampler and more gcn net.
4. Write own dataset loader for custom data.