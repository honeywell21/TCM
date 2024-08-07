# ili

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

for seq_len in  36
do
for e_layers in 2
do
for pred_len in 24 36 48 60
do
 python -u run.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path national_illness.csv \
  --model $model \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --label_len 18 \
  --pred_len $pred_len \
  --e_layers $e_layers \
  --learning_rate 0.01 \
  --enc_in 7 \
  --itr 1 >logs/TCM_log/TCM_$(date +%Y%m%d)/TCM_illness'_'$seq_len'_'$pred_len'_'$(date +%T).log
 done
 done
 done
