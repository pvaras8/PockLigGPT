import os
import pickle
import numpy as np
import pandas as pd
import requests
import selfies as sf
from sklearn.model_selection import train_test_split
import csv
import re



# Directorio de salida
output_dir = 'data/very_big_molecules_selfies_2'
# Archivos locales a procesar
local_files = [
    "/LUSTRE/users/pvaras/sequence_gpt/data/very_big_molecules_selfies_2/chembl/chembl1_filtered.csv",  # <-- reemplaza con la ruta real
    "/LUSTRE/users/pvaras/sequence_gpt/data/very_big_molecules_selfies_2/chembl/chembl2_filtered.csv"
    # "/LUSTRE/users/pvaras/sequence_gpt/data/very_big_molecules_selfies_2/bindingdb/BindingDB_All.tsv"
    # "/LUSTRE/users/pvaras/sequence_gpt/data/very_big_molecules_selfies/full_dataset_EE_morfeus2.csv"
]

os.makedirs(output_dir, exist_ok=True)

# --- util mínima: quedarse con lo anterior al primer '.' ---
def keep_before_dot(smiles: str) -> str:
    if not isinstance(smiles, str):
        return smiles
    return smiles.split('.', 1)[0].strip()

# Directorio de salida
output_dir = 'data/very_big_molecules_selfies_2'
os.makedirs(output_dir, exist_ok=True)

class CharTokenizerSelfies:
    def __init__(self, data, meta_path=None):
        if meta_path and os.path.exists(meta_path):
            self.load_meta(meta_path)
            print('cargando vocablario')

            # Asegura que existan los tokens especiales necesarios
            self._ensure_special_tokens(['<PAD>', '<SOS>', '<EOS>', '<LIGAND>', '</LIGAND>', '<UNK>'])
            self.token2id = {v: k for k, v in self.id2token.items()}
            # Si el meta es fijo, puedes comentar la siguiente línea:
            # self.extend_vocab(data['selfies'])
        else:
            self.id2token = self.build_tokenizer(data['selfies'])
            self.add_special_tokens()
            self.token2id = {v: k for k, v in self.id2token.items()}

    def _ensure_special_tokens(self, specials):
        next_id = (max(self.id2token) + 1) if len(self.id2token) > 0 else 0
        for tok in specials:
            if tok not in self.id2token.values():
                self.id2token[next_id] = tok
                next_id += 1

    def add_special_tokens(self):
        len_tokens = len(self.id2token)
        special_tokens = ['<PAD>', '<SOS>', '<EOS>', '<LIGAND>', '</LIGAND>', '<UNK>']
        for i, token in enumerate(special_tokens, start=len_tokens):
            self.id2token[i] = token

    def build_tokenizer(self, selfies_series):
        tokens = set()
        for selfies_str in selfies_series:
            if pd.notna(selfies_str):
                tokens.update(self.split_selfies(selfies_str))
        return {i: token for i, token in enumerate(sorted(tokens))}
    
    def extend_vocab(self, selfies_series):
        nuevos = set()
        for sf_str in selfies_series:
            if pd.notna(sf_str):
                nuevos.update(self.split_selfies(sf_str))
        nuevos = sorted(t for t in nuevos if t not in self.token2id)
        if not nuevos:
            print("No hay tokens nuevos que añadir.")
            return
        next_id = max(self.id2token) + 1
        for tok in nuevos:
            self.id2token[next_id] = tok
            next_id += 1
        print(f"Añadidos {len(nuevos)} tokens nuevos al vocabulario.")
        self.token2id = {v: k for k, v in self.id2token.items()}

    # ✅ split SELFIES corregido: solo extrae bloques [ ... ]
    def split_selfies(self, selfies_str):
        if not isinstance(selfies_str, str):
            return []
        return re.findall(r'\[[^\]]+\]', selfies_str)

    @property
    def vocab_size(self):
        return len(self.id2token)

    def tokenize(self, selfies_str, padding=True, max_length=156):
        toks = self.split_selfies(selfies_str)
        enc = [self.token2id['<SOS>'], self.token2id['<LIGAND>']]
        enc += [self.token2id.get(t, self.token2id['<UNK>']) for t in toks]
        enc += [self.token2id['</LIGAND>'], self.token2id['<EOS>']]
        if padding and len(enc) < max_length:
            enc += [self.token2id['<PAD>']] * (max_length - len(enc))
        return enc

    def decode(self, token_ids):
        skip = {'<PAD>', '<SOS>', '<EOS>', '<LIGAND>', '</LIGAND>', '<UNK>'}
        return ''.join([self.id2token[i] for i in token_ids if self.id2token[i] not in skip])

    def filter_valid_selfies(self, data, max_sequence_length):
        print(f"Filtrando moléculas SELFIES con longitud mayor a {max_sequence_length} tokens...")
        filtered_data = data[data['selfies'].apply(lambda x: len(self.split_selfies(x)) <= max_sequence_length)]
        print(f"Moleculas restantes: {len(filtered_data)}")
        return filtered_data

    def load_meta(self, meta_path):
        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)
        self.id2token = meta['itos']
        self.token2id = meta['stoi']
        print(f"Tokenizador cargado desde {meta_path}.")

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
            'PAD':'<PAD>', 'SOS':'<SOS>', 'UNK':'<UNK>', 'EOS':'<EOS>',
            'LIGAND_L':'<LIGAND>', 'LIGAND_R':'</LIGAND>'
        }
    }
    with open(meta_path, 'wb') as f:
        pickle.dump(meta, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f'Metadatos guardados en {meta_path}')

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
    df = pd.DataFrame({'smiles': valid_smiles, 'selfies': selfies}).drop_duplicates()
    print(f"Conversión completada. Registros: {len(df)}")
    return df

def process_and_save_batch(data, batch_num, tokenizer, max_length=152):
    filtered_data = tokenizer.filter_valid_selfies(data, max_sequence_length=max_length)
    train_data, val_data = train_test_split(filtered_data['selfies'], test_size=0.1, random_state=42)
    train_ids = [tokenizer.tokenize(selfies) for selfies in train_data]
    val_ids = [tokenizer.tokenize(selfies) for selfies in val_data]
    save_to_bin_file([item for sublist in train_ids for item in sublist],
                     os.path.join(output_dir, f'train_{batch_num}_chembl.bin'))
    save_to_bin_file([item for sublist in val_ids for item in sublist],
                     os.path.join(output_dir, f'val_{batch_num}_chembl.bin'))

# def main():
#     meta_path  = os.path.join(output_dir, 'meta_chembl_db_aa_2_proto4.pkl')
#     batch_num  = 1

#     for file_path in local_files:
#         print(f"Procesando archivo: {file_path}")

#         try:
#             df = pd.read_csv(
#                     file_path,
#                     sep=",",
#                     engine="python",
#                     error_bad_lines=False,   # salta las filas “rotas”
#                     warn_bad_lines=False     # y no muestra avisos
#             )
#             # df = pd.read_csv(
#             #     file_path,
#             #     sep="\t",            # ← tabulador en vez de ‘;’
#             #     engine="c",          # el motor C es más rápido; ‘python’ si tuvieras problemas
#             #     low_memory=False,    # evita troceado automático de tipos
#             #     error_bad_lines=False,   # salta las filas “rotas”
#             #     warn_bad_lines=False     # y no muestra avisos
#             # )
#             # smiles_list = df["Ligand SMILES"].dropna().tolist()
#             smiles_list = df["__smiles_clean__"].dropna().tolist()

#             print('Longitud inicial:', len(smiles_list))        # con coma

#             if not smiles_list:
#                 print(f"❌ No hubo SMILES en columna AF de {file_path}")
#                 continue

#             smiles_list = list(set(smiles_list))          # quita duplicados
#             print('Longitud inicial 2:', len(smiles_list))        # con coma

#             data        = smiles_to_selfies(smiles_list)  # tu función de siempre

#             if data.empty:
#                 print(f"❌ No se pudieron convertir SMILES válidos en {file_path}")
#                 continue

#             tokenizer = CharTokenizerSelfies(
#                 data,
#                 meta_path=meta_path if os.path.exists(meta_path) else None
#             )

#             process_and_save_batch(data, batch_num, tokenizer)  # genera train_/val_
#             # save_meta(tokenizer, meta_path)

#             print(f"✅ Terminado {file_path}")
#             batch_num += 1

#         except Exception as e:
#             print(f"⚠️ Error procesando {file_path}: {e}")

def main():
    meta_path  = os.path.join(output_dir, 'meta_chembl_db_aa_2_proto4.pkl')
    batch_num  = 1

    # 🔹 Ejemplos de prueba (con y sin puntos)
    test_smiles = [
        "Cc1ccccc1",   # benceno
        "Cc1ccccc1.Cl",   # benceno con Cl- separado
        "CSc1ccc(C)cc1.O=C(O)O",  # molécula + ácido oxálico
        "Clc1ccc(Cn2cc[n+](Cc3ccc(Cl)cc3)c2)cc1.[Cl-]",  # con contraión
    ]

    print("=== TEST MANUAL DE SMILES ===")
    print("Entrada:")
    for s in test_smiles:
        print("  ", s)

    # aplica la conversión (usa keep_before_dot dentro)
    data = smiles_to_selfies(test_smiles)

    print("\nSalida (después de keep_before_dot + SELFIES):")
    for smi, sfie in zip(data["smiles"], data["selfies"]):
        print(f"  {smi:50s} -> {sfie}")

    # 🔹 Si quieres, también probamos tokenización
    tokenizer = CharTokenizerSelfies(data, meta_path=meta_path)
    for smi, sfie in zip(data["smiles"], data["selfies"]):
        toks = tokenizer.split_selfies(sfie)
        print(f"\nTokens para {smi}:")
        print(toks)
        ids = tokenizer.tokenize(sfie, padding=False)
        print("Token IDs:", ids)


if __name__ == "__main__":
    main()