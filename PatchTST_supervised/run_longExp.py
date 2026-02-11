import argparse
import os
import torch
from exp.exp_main import Exp_Main
import random
import numpy as np
import shutil
import time
import pandas as pd
from scipy import stats

def parse_args():
    parser = argparse.ArgumentParser(
        description='PatchTST - extended runner with walk-forward and multi-run (tempo + IC95% + relatório)'
    )

    # random seed
    parser.add_argument('--random_seed', type=int, default=2021)

    # basic config
    parser.add_argument('--is_training', type=int, required=True, default=1)
    parser.add_argument('--model_id', type=str, required=True, default='test')
    parser.add_argument('--model', type=str, required=True, default='PatchTST')

    # data loader
    parser.add_argument('--data', type=str, required=True, default='custom')
    parser.add_argument('--root_path', type=str, default='./data/custom/')
    parser.add_argument('--data_path', type=str, default='PRIO3_PATCHTST.csv')
    parser.add_argument('--features', type=str, default='M')
    parser.add_argument('--target', type=str, default='Fech_PRIO3')
    parser.add_argument('--freq', type=str, default='b')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=30)
    parser.add_argument('--label_len', type=int, default=0)
    parser.add_argument('--pred_len', type=int, default=1)

    # PatchTST specific
    parser.add_argument('--patch_len', type=int, default=16)
    parser.add_argument('--stride', type=int, default=8)
    parser.add_argument('--padding_patch', type=str, default='end')
    parser.add_argument('--revin', type=int, default=1)
    parser.add_argument('--affine', type=int, default=0)
    parser.add_argument('--subtract_last', type=int, default=1)
    parser.add_argument('--fc_dropout', type=float, default=0.05)
    parser.add_argument('--head_dropout', type=float, default=0.0)
    parser.add_argument('--individual', type=int, default=0)

    # decomposition
    parser.add_argument('--decomposition', type=int, default=0)
    parser.add_argument('--decomposition_kernel', type=int, default=0)
    parser.add_argument('--kernel_size', type=int, default=3)

    # model architecture
    parser.add_argument('--enc_in', type=int, default=3, help='número total de variáveis no CSV (inclui target)')
    parser.add_argument('--dec_in', type=int, default=3)  
    parser.add_argument('--c_out', type=int, default=1)
    parser.add_argument('--d_model', type=int, default=128)
    parser.add_argument('--n_heads', type=int, default=8)
    parser.add_argument('--e_layers', type=int, default=2)
    parser.add_argument('--d_layers', type=int, default=1)
    parser.add_argument('--d_ff', type=int, default=256)
    parser.add_argument('--factor', type=int, default=1)
    parser.add_argument('--distil', action='store_false', default=True)
    parser.add_argument('--dropout', type=float, default=0.05)
    parser.add_argument('--embed', type=str, default='timeF')

    # optimization
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--train_epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--learning_rate', type=float, default=0.0005)
    parser.add_argument('--lradj', type=str, default='type3')
    parser.add_argument('--use_amp', type=int, default=0)
    parser.add_argument('--pct_start', type=float, default=0.3)
    parser.add_argument('--des', type=str, default='exp')
    parser.add_argument('--loss', type=str, default='mse')

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--use_multi_gpu', action='store_true', default=False)
    parser.add_argument('--device_ids', type=int, nargs='+', default=[0])

    # NEW FLAGS
    parser.add_argument('--n_runs', type=int, default=10)
    parser.add_argument('--out_dir', type=str, default='resultados_exp4/')
    parser.add_argument('--do_walkforward', action='store_true')
    parser.add_argument('--walkf_exog', type=str, default='BRENT,PETR4')
    parser.add_argument('--do_predict', action='store_true')

    # BASELINE PARAMETRIZÁVEL (para os 4 experimentos)
    parser.add_argument('--baseline_mse', type=float, default=None,
                        help='MSE do baseline LSTM para comparação (deixe vazio para não comparar)')
    parser.add_argument('--baseline_name', type=str, default='LSTM (Silva Júnior, 2024)',
                        help='Nome exibido no relatório para o baseline')

    return parser.parse_args()

def ic95(values: np.ndarray):
    """Retorna (mean, margin, (low, high)) do IC 95% com t de Student."""
    n = len(values)
    if n < 2:
        mean = float(np.mean(values))
        return mean, float('nan'), (float('nan'), float('nan'))
    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1))
    se = std / np.sqrt(n)
    t = stats.t.ppf(0.975, df=n - 1)
    margin = float(t * se)
    return mean, margin, (mean - margin, mean + margin)

def detect_experiment_title(data_path: str):
    data_name = os.path.basename(data_path).upper()
    if "UNIVARIADO" in data_name:
        return "PATCHTST UNIVARIADO (SEM EXÓGENAS)", "Nenhuma"
    if "SOMENTE_PETR4" in data_name:
        return "PATCHTST COM PETR4 (APENAS COVARIÁVEL SETORIAL)", "PETR4"
    if "SOMENTE_BRNT" in data_name or "BRENT" in data_name:
        return "PATCHTST COM BRENT (APENAS COVARIÁVEL INTERNACIONAL)", "BRENT"
    return "PATCHTST COMPLETO (BRENT + PETR4)", "BRENT + PETR4"

if __name__ == '__main__':
    args = parse_args()
    args.dec_in = args.enc_in

    # seeds base
    fix_seed = args.random_seed
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    os.makedirs(args.out_dir, exist_ok=True)
    metrics_all_runs = []
    execution_times = []

    Exp = Exp_Main

 
    for run_idx in range(args.n_runs):
        start_time = time.time()

        cur_seed = args.random_seed + run_idx
        random.seed(cur_seed)
        torch.manual_seed(cur_seed)
        np.random.seed(cur_seed)

        setting = '{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(
            args.model_id, args.model, args.data, args.features, args.seq_len, args.label_len, args.pred_len,
            args.d_model, args.n_heads, args.e_layers, args.d_layers, args.d_ff, args.factor,
            args.embed, args.distil, args.des, run_idx
        )

        print(f"\n======= RUN {run_idx+1}/{args.n_runs} : {setting} (seed={cur_seed}) =======")
        exp = Exp(args)

        # TRAIN
        if args.is_training:
            print(f'>>>>>>> start training : {setting} >>>>>>>>>>>>>>>>>>>>>>>')
            exp.train(setting)

        # TEST (repo behavior)
        print(f'>>>>>>> testing : {setting} <<<<<<<<<<<<<<<<<<<<<<<<<<<')
        try:
            exp.test(setting)
        except Exception as e:
            print("Warning: exp.test failed:", e)

        # copiar checkpoints
        ckpt_src = os.path.join(args.checkpoints, setting)
        run_out_dir = os.path.join(args.out_dir, setting)
        os.makedirs(run_out_dir, exist_ok=True)

        if os.path.exists(ckpt_src):
            try:
                for f in os.listdir(ckpt_src):
                    s = os.path.join(ckpt_src, f)
                    d = os.path.join(run_out_dir, f)
                    if os.path.isfile(s):
                        shutil.copy2(s, d)
            except Exception as e:
                print("Warning copying checkpoints:", e)

        # WALK-FORWARD (métricas oficiais)
        metrics_row_this_run = None
        if args.do_walkforward:
            futr_exog_list = [s.strip() for s in args.walkf_exog.split(',')] if args.walkf_exog else None
            if hasattr(exp, 'walk_forward_and_save'):
                try:
                    print("Running walk-forward and saving results...")
                    wf_pred_csv, wf_metrics_csv = exp.walk_forward_and_save(
                        setting, os.path.join(run_out_dir, "walkforward"), futr_exog_list=futr_exog_list
                    )

                    dfm = pd.read_csv(wf_metrics_csv)
                    metrics_row_this_run = dfm.to_dict(orient='records')[0]
                    metrics_row_this_run.update({"run": run_idx + 1, "seed": cur_seed, "setting": setting})
                    metrics_all_runs.append(metrics_row_this_run)
                except Exception as e:
                    print("Walk-forward failed:", e)
            else:
                print("Exp_Main.walk_forward_and_save not implemented. Skipping walk-forward.")

        # fim da run: tempo
        duration_min = (time.time() - start_time) / 60.0
        execution_times.append(duration_min)
        print(f"Tempo da execução {run_idx+1}: {duration_min:.2f} minutos")

        # salvar tempo dentro da linha de métricas desta run (se existir)
        if metrics_row_this_run is not None:
            metrics_row_this_run["time_min"] = duration_min

        # Limpeza
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

    # =============================================
    # RELATÓRIO FINAL (fora do loop)
    # =============================================
    print("\n" + "="*80)
    print("GERANDO RELATÓRIO FINAL OFICIAL PARA O TCC...")
    print("="*80)

    if len(metrics_all_runs) == 0:
        print("Nenhuma métrica encontrada. Rode com --do_walkforward para gerar o relatório oficial.")
        print("="*80)
        print("\nTodas as execuções concluídas.")
        raise SystemExit(0)

    df_all = pd.DataFrame(metrics_all_runs)

    # IC95%
    mape_mean, mape_marg, mape_ic = ic95(df_all["MAPE"].to_numpy())
    mae_mean,  mae_marg,  mae_ic  = ic95(df_all["MAE"].to_numpy())
    mse_mean,  mse_marg,  mse_ic  = ic95(df_all["MSE"].to_numpy())

    tempo_total = float(np.sum(execution_times))

    # melhor/pior por seed
    best_i = df_all["MAPE"].idxmin()
    worst_i = df_all["MAPE"].idxmax()
    best_seed = int(df_all.loc[best_i, "seed"])
    worst_seed = int(df_all.loc[worst_i, "seed"])
    best_mape = float(df_all.loc[best_i, "MAPE"])
    worst_mape = float(df_all.loc[worst_i, "MAPE"])

    # detectar título do experimento
    config_title, exog_name = detect_experiment_title(args.data_path)

    seeds_ini = args.random_seed
    seeds_fim = args.random_seed + args.n_runs - 1

    # comparação baseline (opcional)
    baseline_block = ""
    if args.baseline_mse is not None and args.baseline_mse > 0:
        reduc_mse_pct = (args.baseline_mse - mse_mean) / args.baseline_mse * 100.0
        baseline_block = f"""
COMPARAÇÃO COM O BASELINE:
{'-'*60}
Baseline              : {args.baseline_name}
MSE Baseline          : {args.baseline_mse:.6f}
MSE PatchTST          : {mse_mean:.6f}
Variação do MSE       : {reduc_mse_pct:.1f}%  (positivo = melhora)
"""
    else:
        baseline_block = f"""
COMPARAÇÃO COM O BASELINE:
{'-'*60}
Baseline              : (não informado)  → use --baseline_mse para comparar
"""

    relatorio_texto = f"""RELATÓRIO FINAL COMPLETO - {config_title}
{'='*75}

CONFIGURAÇÃO DO EXPERIMENTO:
{'-'*50}
Modelo                : PatchTST (Nie et al., 2023)
Dataset               : {os.path.basename(args.data_path)}
Variável alvo         : {args.target}
Covariável(is)        : {exog_name}
Janela de entrada     : {args.seq_len} dias úteis
Horizonte de previsão : {args.pred_len} dia(s) à frente
Validação             : Walk-forward com retreinamento contínuo
Número de execuções   : {args.n_runs} (seeds {seeds_ini}–{seeds_fim})
RevIN                 : {'Ativado' if args.revin else 'Desativado'}
Subtract_last         : {'Ativado' if args.subtract_last else 'Desativado'}
Patch_len / Stride    : {args.patch_len} / {args.stride}

MÉTRICAS CONSOLIDADAS ({args.n_runs} execuções independentes):
{'-'*60}
{'Métrica':<8} {'Média':>12} {'± Margem':>12} {'IC 95%':>28}
{'-'*60}
{'MSE':<8} {mse_mean:12.6f} {mse_marg:12.6f}  [{mse_ic[0]:.6f}, {mse_ic[1]:.6f}]
{'MAE':<8} {mae_mean:12.4f} {mae_marg:12.4f}  [{mae_ic[0]:.4f}, {mae_ic[1]:.4f}]
{'MAPE':<8} {mape_mean:11.3f}% {mape_marg:11.3f}%  [{mape_ic[0]:.3f}%, {mape_ic[1]:.3f}%]
{'-'*60}

DESEMPENHO:
Melhor MAPE (seed)    : {best_mape:.3f}% (seed {best_seed})
Pior MAPE (seed)      : {worst_mape:.3f}% (seed {worst_seed})
Desvio padrão MAPE    : {df_all['MAPE'].std(ddof=1):.3f}%

TEMPO DE EXECUÇÃO:
Tempo médio por run   : {np.mean(execution_times):.2f} minutos
Tempo total           : {tempo_total:.1f} minutos
{baseline_block}
Relatório gerado em {time.strftime('%d/%m/%Y às %H:%M')}
"""

    relatorio_path = os.path.join(args.out_dir, "RELATORIO_FINAL_OFICIAL.txt")
    with open(relatorio_path, "w", encoding="utf-8") as f:
        f.write(relatorio_texto.strip())

    print(relatorio_texto)
    print(f"\nRelatório salvo em: {relatorio_path}")
    print("\nTodas as execuções concluídas com sucesso!")
