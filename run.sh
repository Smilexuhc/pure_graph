for i in {0,1}
do
CUDA_VISIBLE_DEVICES=1 python main.py --batch_size=3000 --dataset='reddit' --loss_norm=i --eval_sample=1
done

for i in {}
CUDA_VISIBLE_DEVICES=3 python main.py --batch_size=6000 --loss_norm=1 --eval_sample=0