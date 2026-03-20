import os
import csv
import pickle
import numpy as np
import pandas as pd
import selfies as sf
from sklearn.model_selection import train_test_split
import re

# =========================
# Configuración de E/S
# =========================
output_dir = 'data/very_big_molecules_selfies_2'
local_files = [
    # "/LUSTRE/users/pvaras/nanoGPT/data/very_big_molecules_selfies_2/cross_docked_jblasser/merged_pocket_smiles.csv"
    "/LUSTRE/users/pvaras/sequence_gpt/data/very_big_molecules_selfies_2/cross_docked_jblasser/crossdocked_filtered_with_pocket_smiles.csv"

]
os.makedirs(output_dir, exist_ok=True)

# meta
META_FILENAME = 'meta_chembl_db_aa_2_proto4.pkl'
# Modo prueba (solo N filas). Pon None para procesar todo.
SAMPLE_FIRST_N = None

# Longitud máxima total (incluye <SOS>, <EOS> y delimitadores)
MAX_LEN = 156

# =========================
# Mapeo de aminoácidos
# =========================
AA_ONE_TO_TOKEN = {
    "A": "<AA_ALA>", "C": "<AA_CYS>", "D": "<AA_ASP>", "E": "<AA_GLU>",
    "F": "<AA_PHE>", "G": "<AA_GLY>", "H": "<AA_HIS>", "I": "<AA_ILE>",
    "K": "<AA_LYS>", "L": "<AA_LEU>", "M": "<AA_MET>", "N": "<AA_ASN>",
    "P": "<AA_PRO>", "Q": "<AA_GLN>", "R": "<AA_ARG>", "S": "<AA_SER>",
    "T": "<AA_THR>", "V": "<AA_VAL>", "W": "<AA_TRP>", "Y": "<AA_TYR>",
}

# =========================
# Tokenizador combinado
# =========================

def keep_before_dot(smiles: str) -> str:
    if not isinstance(smiles, str):
        return smiles
    return smiles.split('.', 1)[0].strip()

class CharTokenizerPocketLigand:
    def __init__(self, data, meta_path=None):
        print('Construyendo tokenizador POCKET+LIGAND...')
        if meta_path and os.path.exists(meta_path):
            self.load_meta(meta_path)
            # Asegura especiales + AA y extiende vocab con nuevas SELFIES
            self._ensure_special_tokens(
                ['<PAD>', '<SOS>', '<EOS>',
                 '<LIGAND>', '</LIGAND>',
                 '<POCKET>', '</POCKET>',
                 '<UNK>'] + list(AA_ONE_TO_TOKEN.values())
            )
            self.token2id = {v: k for k, v in self.id2token.items()}
            # self.extend_vocab(data['selfies'])
            print(f"Vocab tras cargar+extender: {self.vocab_size}")
        else:
            self.id2token = self.build_tokenizer(data['selfies'])
            self.add_special_tokens()
            self._ensure_special_tokens(list(AA_ONE_TO_TOKEN.values()) + ['<UNK>'])
            self.token2id = {v: k for k, v in self.id2token.items()}
            (f"Vocab creado de cero: {self.vocab_size}")

    def add_special_tokens(self):
        len_tokens = len(self.id2token)
        special_tokens = ['<PAD>', '<SOS>', '<EOS>', '<UNK>', '<LIGAND>', '</LIGAND>', '<POCKET>', '</POCKET>']
        for i, token in enumerate(special_tokens, start=len_tokens):
            self.id2token[i] = token

    def _ensure_special_tokens(self, specials):
        next_id = (max(self.id2token) + 1) if len(self.id2token) > 0 else 0
        for tok in specials:
            if tok not in self.id2token.values():
                self.id2token[next_id] = tok
                next_id += 1

    def build_tokenizer(self, selfies_series):
        tokens = set()
        for sf_str in selfies_series:
            if pd.notna(sf_str):
                tokens.update(self.split_selfies(sf_str))
        return {i: token for i, token in enumerate(sorted(tokens))}

    def extend_vocab(self, selfies_series):
        nuevos = set()
        for sf_str in selfies_series:
            if pd.notna(sf_str):
                nuevos.update(self.split_selfies(sf_str))
        nuevos = sorted(t for t in nuevos if t not in self.token2id)
        if not nuevos:
            print("No hay tokens SELFIES nuevos que añadir.")
            return
        next_id = max(self.id2token) + 1
        for tok in nuevos:
            self.id2token[next_id] = tok
            next_id += 1
        self.token2id = {v: k for k, v in self.id2token.items()}
        print(f"Añadidos {len(nuevos)} tokens SELFIES nuevos al vocabulario.")

    # ✅ split SELFIES corregido: solo extrae bloques [ ... ]
    def split_selfies(self, selfies_str):
        if not isinstance(selfies_str, str):
            return []
        return re.findall(r'\[[^\]]+\]', selfies_str)

    @property
    def vocab_size(self):
        return len(self.id2token)

    # ------- Núcleo y filtro por longitud (simple) -------
    def core_length(self, pocket_str, selfies_str):
        """Longitud del cuerpo: nº AAs + nº tokens SELFIES."""
        return len(pocket_str) + len(self.split_selfies(selfies_str))

    def filter_valid_pairs(self, df_pairs, max_sequence_length):
        # Reservamos 6 posiciones: <SOS>, <EOS>, <POCKET>, </POCKET>, <LIGAND>, </LIGAND>
        max_core = max_sequence_length - 6
        print(f"Filtrando pares con core > {max_core} (max_len={max_sequence_length})...")
        keep = []
        for _, row in df_pairs.iterrows():
            core_len = self.core_length(row['pocket'], row['selfies'])
            keep.append(core_len <= max_core)
        filtered = df_pairs[pd.Series(keep, index=df_pairs.index)]
        print(f"Pares restantes: {len(filtered)} (de {len(df_pairs)})")
        return filtered

    # ------- Tokenización (sin truncado, solo padding) -------
    def tokenize_pair(self, pocket_str, selfies_str, padding=True, max_length=156):
        pocket_tokens = ["<POCKET>"] + [AA_ONE_TO_TOKEN.get(aa, "<UNK>") for aa in pocket_str] + ["</POCKET>"]
        ligand_tokens = ["<LIGAND>"] + self.split_selfies(selfies_str) + ["</LIGAND>"]
        seq_tokens = ["<SOS>"] + pocket_tokens + ligand_tokens + ["<EOS>"]

        enc = [self.token2id.get(t, self.token2id['<UNK>']) for t in seq_tokens]

        if padding and len(enc) < max_length:
            enc += [self.token2id['<PAD>']] * (max_length - len(enc))

        # Si algo supera max_length, es un error de filtrado previo (no truncamos aquí)
        if len(enc) > max_length:
            raise ValueError(f"Secuencia > max_length ({len(enc)} > {max_length}). Revisa el filtrado.")

        return enc

    def load_meta(self, meta_path):
        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)
        self.id2token = meta['itos']
        self.token2id = meta['stoi']
        print(f"Tokenizador cargado desde {meta_path}.")

# =========================
# Utilidades de guardado
# =========================
def save_to_bin_file(data, filename):
    data = np.array(data, dtype=np.uint16)
    with open(filename, 'ab') as f:
        data.tofile(f)
    print(f'Datos guardados en {filename}')

def save_meta(tokenizer, meta_path):
    meta = {
        'vocab_size': tokenizer.vocab_size,
        'itos': tokenizer.id2token,
        'stoi': tokenizer.token2id,
        'special_tokens': {
            'PAD':'<PAD>', 'SOS':'<SOS>', 'EOS':'<EOS>',
            'LIGAND_L':'<LIGAND>', 'LIGAND_R':'</LIGAND>',
            'POCKET_L':'<POCKET>', 'POCKET_R':'</POCKET>',
            'UNK':'<UNK>'
        },
        'aa_tokens': list(AA_ONE_TO_TOKEN.values())
    }
    with open(meta_path, 'wb') as f:
        pickle.dump(meta, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f'Metadatos guardados en {meta_path}')

# =========================
# Preprocesado específico
# =========================
def split_pocket_smiles(df):
    """Devuelve DataFrame con columnas: pocket, smiles."""
    pockets, smiles = [], []
    for entry in df["pocket_smiles"].dropna().tolist():
        if "_" in entry:
            p, s = entry.split("_", 1)
            pockets.append(p.strip())
            smiles.append(s.strip())
    return pd.DataFrame({"pocket": pockets, "smiles": smiles})

def smiles_to_selfies(smiles_list):
    print("Convirtiendo SMILES a SELFIES (usando fragmento antes del '.')...")
    valid_smiles = []
    selfies = []
    for smiles in smiles_list:
        try:
            if pd.notna(smiles):
                smi_main = keep_before_dot(smiles.strip())
                # si se quedó vacío, salta
                if not smi_main:
                    continue
                selfie = sf.encoder(smi_main)
                valid_smiles.append(smi_main)
                selfies.append(selfie)
        except Exception as e:
            print(f"Error al convertir SMILES '{smiles}': {e}")
    # quitar duplicados tras el recorte por '.'
    df = pd.DataFrame({'smiles': valid_smiles, 'selfies': selfies}) #Esta linea va a haber que cambiarla (me esta quitando smiles repetidos en pockets diferentes lo que es util)
    print(f"Conversión completada. Registros: {len(df)}")
    return df

# =========================
# Pipeline de lote
# =========================
def process_and_save_batch(df_pairs, batch_num, tokenizer, max_length=MAX_LEN):
    # Filtra por longitud del cuerpo (core)
    filtered = tokenizer.filter_valid_pairs(df_pairs, max_sequence_length=max_length)

    # Split train/val sobre PAREJAS
    train_df, val_df = train_test_split(filtered, test_size=0.1, random_state=42)

    # Tokenizar (padding dentro; sin truncado)
    train_ids = [tokenizer.tokenize_pair(p, s, padding=True, max_length=max_length)
                 for p, s in zip(train_df['pocket'], train_df['selfies'])]
    val_ids   = [tokenizer.tokenize_pair(p, s, padding=True, max_length=max_length)
                 for p, s in zip(val_df['pocket'], val_df['selfies'])]

    # Guardar binarios concatenados
    save_to_bin_file([it for seq in train_ids for it in seq], os.path.join(output_dir, f'train_1_ligand_{batch_num}.bin'))
    save_to_bin_file([it for seq in val_ids   for it in seq], os.path.join(output_dir, f'val_1_ligand_{batch_num}.bin'))

# =========================
# MAIN
# =========================
def main():
    meta_path = os.path.join(output_dir, META_FILENAME)
    batch_num = 1

    for file_path in local_files:
        print(f"\nProcesando archivo: {file_path}")

        try:
            # Lee CSV (ajusta sep si fuese necesario)
            df = pd.read_csv(file_path)

            # Modo prueba
            if SAMPLE_FIRST_N is not None:
                df = df.head(SAMPLE_FIRST_N)

            if "pocket_smiles" not in df.columns:
                print("❌ No se encontró la columna 'pocket_smiles'.")
                continue
            pairs = split_pocket_smiles(df)
            if pairs.empty:
                print("❌ No hay pares válidos pocket+smiles.")
                continue

            # Convertir SMILES → SELFIES (y alinear por índice)
            data_sm = smiles_to_selfies(pairs["smiles"].tolist())
            if data_sm.empty:
                print("❌ No se pudieron convertir SMILES válidos.")
                continue
            pairs = pairs.loc[data_sm.index].copy()
            pairs["selfies"] = data_sm["selfies"].values

            # Construir/actualizar tokenizador
            tokenizer = CharTokenizerPocketLigand(
                pairs,
                meta_path=meta_path if os.path.exists(meta_path) else None
            )

            # 🧩 Mostrar los 3 primeros ejemplos
            print("\n=== Vista previa de los 3 primeros pares tokenizados ===")
            for i in range(min(3, len(pairs))):
                pocket = pairs.iloc[i]["pocket"]
                smiles = pairs.iloc[i]["smiles"]
                selfies = pairs.iloc[i]["selfies"]

                print(f"\nEjemplo {i+1}:")
                print(f"POCKET: {pocket}")
                print(f"SMILES: {smiles}")
                print(f"SELFIES: {selfies}")

                # Tokenizar el par
                encoded = tokenizer.tokenize_pair(pocket, selfies, padding=True, max_length=MAX_LEN)
                decoded = [tokenizer.id2token[idx] for idx in encoded if tokenizer.id2token[idx] != "<PAD>"]

                print(f"Tokens ({len(decoded)}): {' '.join(decoded[:200])}{' ...' if len(decoded) > 200 else ''}")
                print(f"IDs ({len(encoded)}): {encoded[:200]}{' ...' if len(encoded) > 200 else ''}")
                print(f"Longitud total (con PAD): {len(encoded)}")

            print("\n✅ Vista previa completada.\n")

            # Guardar lote
            process_and_save_batch(pairs, batch_num, tokenizer, max_length=MAX_LEN)
            # Guardar meta actualizado
            # save_meta(tokenizer, meta_path)

            print(f"✅ Terminado {file_path}")
            batch_num += 1

        except Exception as e:
            print(f"⚠️ Error procesando {file_path}: {e}")

if __name__ == "__main__":
    main()
