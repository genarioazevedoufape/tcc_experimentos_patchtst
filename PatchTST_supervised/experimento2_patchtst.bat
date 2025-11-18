@echo off
echo.
echo ================================================
echo  EXPERIMENTO 2 - SÓ PETR4 COMO COVARIÁVEL
echo ================================================

python run_longExp.py ^
    --is_training 1 ^
    --model_id Exp2_somente_PETR4 ^
    --model PatchTST ^
    --data custom ^
    --root_path ./data/custom/ ^
    --data_path PRIO3_somente_PETR4.csv ^
    --features M ^
    --target Fech_PRIO3 ^
    --seq_len 30 ^
    --pred_len 1 ^
    --enc_in 2 ^
    --patch_len 16 ^
    --stride 8 ^
    --revin 1 ^
    --subtract_last 1 ^
    --decomposition 0 ^
    --kernel_size 3 ^
    --d_model 128 ^
    --n_heads 8 ^
    --e_layers 2 ^
    --train_epochs 20 ^
    --batch_size 32 ^
    --learning_rate 0.0005 ^
    --n_runs 10 ^
    --do_walkforward ^
    --walkf_exog PETR4 ^
    --out_dir resultados_exp2_somente_PETR4/ ^
    --use_gpu False

echo.
echo ================================================
echo EXPERIMENTO 2 CONCLUÍDO!
echo Resultados em: resultados_exp2_somente_PETR4/
echo ================================================
pause