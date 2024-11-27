# ETTm2

if [ ! -d "./logs" ]; then
    mkdir ./logs
fi
if [ ! -d "./logs/TCM_log" ]; then
    mkdir ./logs/TCM_log
fi
if [ ! -d "./logs/TCM_log/TCM_$(date +%Y%m%d)" ]; then
    mkdir ./logs/TCM_log/TCM_$(date +%Y%m%d)
fi


model=TCM


for seq_len in  96
do
for e_layers in 2
do
for pred_len in  96
do
python -u run.py \
      --is_training 1 \
      --data_path ETTm2.csv \
      --data ETTm2 \
      --features M \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in 7 \
      --batch_size 32 \
      --learning_rate 0.001 \
      --dropout 0.3 \
      --model $model \
      --e_layers $e_layers \
      --itr 1 >logs/TCM_log/TCM_$(date +%Y%m%d)/TCM_ETTm2'_'$seq_len'_'$pred_len'_'$(date +%T).log

done 
done 
done 