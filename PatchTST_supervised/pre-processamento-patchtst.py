import os
import pandas as pd

# =============================================================================
# CAMINHOS CORRIGIDOS 
# =============================================================================
CAMINHO_RAIZ = os.path.dirname(os.path.abspath(__file__))        # pasta atual
CAMINHO_ENTRADA = os.path.join(CAMINHO_RAIZ, "data")             # ← seus CSVs brutos
CAMINHO_SAIDA   = os.path.join(CAMINHO_RAIZ, "data", "custom")   # ← onde PatchTST espera
os.makedirs(CAMINHO_SAIDA, exist_ok=True)

print(f"Procurando dados em: {CAMINHO_ENTRADA}")
print(f"Salvando datasets em: {CAMINHO_SAIDA}\n")

# =============================================================================
# FUNÇÕES DE CARREGAMENTO (corrigidas e testadas)
# =============================================================================
def carregar_yahoo(caminho, nome_coluna):
    if not os.path.exists(caminho):
        raise FileNotFoundError(f"Arquivo não encontrado: {caminho}")
    df = pd.read_csv(caminho)
    if "Date" not in df.columns:
        raise ValueError(f"Coluna 'Date' não encontrada em {caminho}")
    df = df.rename(columns={"Date": "date", "Close": nome_coluna})
    df["date"] = pd.to_datetime(df["date"])
    df = df[["date", nome_coluna]].dropna()
    df = df.sort_values("date").reset_index(drop=True)
    print(f"   {os.path.basename(caminho)} → {len(df)} linhas")
    return df

def carregar_brent(caminho):
    if not os.path.exists(caminho):
        raise FileNotFoundError(f"Arquivo BRENT não encontrado: {caminho}")
    # Formato brasileiro com header errado
    cols = ["Last", "Fech_BRENT", "Open", "High", "Low", "Vol", "Change%"]
    df = pd.read_csv(caminho, encoding='latin-1', usecols=[0,1], names=["Data", "Fech_BRENT"], header=0)
    
    df["Fech_BRENT"] = df["Fech_BRENT"].astype(str).str.replace('.', '', regex=False).str.replace(',', '.', regex=False)
    df["Fech_BRENT"] = pd.to_numeric(df["Fech_BRENT"], errors='coerce')
    df["Data"] = pd.to_datetime(df["Data"], dayfirst=True, errors='coerce')
    df = df.rename(columns={"Data": "date"}).dropna()
    df = df.sort_values("date").reset_index(drop=True)
    print(f"   BRNT.csv → {len(df)} linhas")
    return df

# =============================================================================
# CARREGAR OS ARQUIVOS
# =============================================================================
print("Carregando arquivos brutos...")

petr4_path = os.path.join(CAMINHO_ENTRADA, "PETR4.SA.csv")
prio3_path = os.path.join(CAMINHO_ENTRADA, "PRIO3.SA.csv")
brent_path = os.path.join(CAMINHO_ENTRADA, "BRNT.csv")

petr4 = carregar_yahoo(petr4_path, "Fech_PETR4")
prio3 = carregar_yahoo(prio3_path, "Fech_PRIO3")
brent = carregar_brent(brent_path)

# =============================================================================
# GERAR OS 4 DATASETS
# =============================================================================
print("\nGerando datasets...")

# 1. Completo (PETR4 + BRENT)
df_full = prio3.merge(petr4, on="date").merge(brent, on="date")
df_full = df_full[["date", "Fech_PETR4", "Fech_BRENT", "Fech_PRIO3"]]
df_full.to_csv(os.path.join(CAMINHO_SAIDA, "PRIO3_com_PETR4_BRENT.csv"), index=False)
print(f"1. Completo (3 vars): {len(df_full)} linhas → PRIO3_com_PETR4_BRENT.csv")

# 2. Só PETR4
df_petr4 = prio3.merge(petr4, on="date")
df_petr4 = df_petr4[["date", "Fech_PETR4", "Fech_PRIO3"]]
df_petr4.to_csv(os.path.join(CAMINHO_SAIDA, "PRIO3_somente_PETR4.csv"), index=False)
print(f"2. Só PETR4: {len(df_petr4)} linhas → PRIO3_somente_PETR4.csv")

# 3. Só BRENT
df_brent = prio3.merge(brent, on="date")
df_brent = df_brent[["date", "Fech_BRENT", "Fech_PRIO3"]]
df_brent.to_csv(os.path.join(CAMINHO_SAIDA, "PRIO3_somente_BRNT.csv"), index=False)
print(f"3. Só BRENT: {len(df_brent)} linhas → PRIO3_somente_BRNT.csv")

# 4. Univariado
df_uni = prio3.copy()
df_uni = df_uni[["date", "Fech_PRIO3"]]
df_uni.to_csv(os.path.join(CAMINHO_SAIDA, "PRIO3_univariado.csv"), index=False)
print(f"4. Univariado: {len(df_uni)} linhas → PRIO3_univariado.csv")

# =============================================================================
# FINAL
# =============================================================================
print("\nTODOS OS 4 DATASETS GERADOS COM SUCESSO!")
print("="*80)
