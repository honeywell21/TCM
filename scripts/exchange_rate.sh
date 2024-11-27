# exchange_rate

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
for e_layers in 1
do
for pred_len in  96
do
python -u run.py \
      --is_training 1 \
      --data_path exchange_rate.csv \
      --data custom \
      --features M \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in 8 \
      --learning_rate 0.0005 \
      --dropout 0.5 \
      --model $model \
      --e_layers $e_layers \
      --itr 1 >logs/TCM_log/TCM_$(date +%Y%m%d)/TCM_exchange'_'$seq_len'_'$pred_len'_'$(date +%T).log

done 
done 
done 