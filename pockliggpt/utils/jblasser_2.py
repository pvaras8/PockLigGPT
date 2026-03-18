#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import urllib.request
import numpy as np
import pandas as pd
import requests
from tqdm import tqdm
from Bio.PDB import MMCIFParser, is_aa
import argparse

# ==============================
# ARGUMENTOS DE ENTRADA
# ==============================
parser = argparse.ArgumentParser()
parser.add_argument("--input", type=str, default="binding_affinity_pchembl_gt7.csv",
                    help="CSV de entrada")
parser.add_argument("--out", type=str, default=None,
                    help="CSV de salida")
parser.add_argument("--start", type=int, default=100000,
                    help="Fila inicial (incluida, 0-based)")
parser.add_argument("--end", type=int, default=None,
                    help="Fila final (excluida)")
parser.add_argument("--flush-every", type=int, default=25000,
                    help="Cada cuántas filas volcar resultados al CSV")
args = parser.parse_args()

INPUT_CSV    = args.input
START        = args.start
END          = args.end
FLUSH_EVERY  = args.flush_every
OUT_CSV      = args.out or f"jblasser_affinity_with_pockets_gt7.part_{START}_{END or 'end'}.csv"

RADIUS_ANG   = 10.0
MIN_IDENT    = 0.90
MIN_COV      = 0.80
PDB_DIR      = "./pdb_cif"

# ==============================
# 0) Leer dataset CSV
# ==============================
print(f"Leyendo CSV: {INPUT_CSV}")
df_full = pd.read_csv(INPUT_CSV)
print(f"Total filas en CSV: {len(df_full)}")

df_iter = df_full.iloc[START:END].reset_index(drop=True)
print(f"Procesando filas {START} hasta {END or len(df_full)} (exclusivo).")
print(f"Salida incremental: {OUT_CSV}, flush cada {FLUSH_EVERY} filas.")

# ==============================
# 1) Utilidades (RCSB, mmCIF)
# ==============================
RCSB_SEQ_URL = "https://search.rcsb.org/rcsbsearch/v2/query"

def rcsb_sequence_search(seq, identity_cutoff=0.9, evalue_cutoff=1e-5, max_results=3):
    payload = {
        "query": {
            "type": "terminal",
            "service": "sequence",
            "parameters": {
                "evalue_cutoff": evalue_cutoff,
                "identity_cutoff": identity_cutoff,
                "sequence_type": "protein",
                "target": "pdb_protein_sequence",
                "value": seq
            }
        },
        "request_options": {"scoring_strategy": "sequence"},
        "return_type": "polymer_entity"
    }
    r = requests.post(RCSB_SEQ_URL, json=payload, timeout=60)
    r.raise_for_status()
    data = r.json()

    hits = []
    for i, item in enumerate(data.get("result_set", [])):
        if i >= max_results:
            break
        struct_id = item.get("identifier")
        if not struct_id:
            continue
        pdb_id, entity_id = struct_id.split("_", 1) if "_" in struct_id else (struct_id, None)
        ident, cov, e_val = None, None, item.get("score", None)
        try:
            svc = item["services"][0]["nodes"][0]["service_response"]["payload"]
            ident = float(svc.get("sequence_identity")) if svc.get("sequence_identity") else None
            cov   = float(svc.get("coverage")) if svc.get("coverage") else None
        except Exception:
            pass
        hits.append({"pdb_id": pdb_id.upper(), "entity_id": entity_id,
                     "identity": ident, "coverage": cov, "evalue": e_val})
    return hits

def download_mmcif(pdb_id, out_dir=PDB_DIR):
    os.makedirs(out_dir, exist_ok=True)
    pdb_id = pdb_id.upper()
    url = f"https://files.rcsb.org/download/{pdb_id}.cif"
    dst = os.path.join(out_dir, f"{pdb_id}.cif")
    if not os.path.exists(dst):
        urllib.request.urlretrieve(url, dst)
    return dst

# ==============================
# 2) Ligandos y pocket
# ==============================
EXCLUDE_LIG = set("""
HOH WAT DOD NA  K  CL  MG  CA  MN  ZN  CU  CD  CO  FE  SO4 PO4 NO3
GOL EDO PEG PGE PGO TRS MES HEP HEPES BME DMS ACT ACE CIT MSE SEP TPO PTR
""".split())

THREE_TO_ONE = {'ALA':'A','CYS':'C','ASP':'D','GLU':'E','PHE':'F','GLY':'G','HIS':'H',
                'ILE':'I','LYS':'K','LEU':'L','MET':'M','ASN':'N','PRO':'P','GLN':'Q',
                'ARG':'R','SER':'S','THR':'T','VAL':'V','TRP':'W','TYR':'Y'}

def atom_is_heavy(atom):
    return atom.element not in ("H","D")

def list_ligands_by_entity(structure):
    ligs = []
    for model in structure:
        for chain in model:
            for res in chain:
                hetflag, _, _ = res.get_id()
                if hetflag.strip() == "" or is_aa(res, standard=True):
                    continue
                resname = res.get_resname().strip()
                if resname in EXCLUDE_LIG:
                    continue
                heavy = sum(1 for a in res.get_atoms() if atom_is_heavy(a))
                if heavy < 6 or heavy > 80:
                    continue
                ligs.append((model.get_id(), chain.get_id(), res, resname, heavy))
    return ligs

def ligand_center(lig_res):
    xyz = [a.get_coord() for a in lig_res.get_atoms() if atom_is_heavy(a)]
    return np.mean(xyz, axis=0) if len(xyz) else None

def pocket_from_ligand(structure, chain_id, lig_res, radius=RADIUS_ANG, contact_cut=None):
    lig_atoms = [a for a in lig_res.get_atoms() if atom_is_heavy(a)]
    if not lig_atoms:
        return []
    if contact_cut is None:
        center = ligand_center(lig_res)
    target_chain = None
    for model in structure:
        for ch in model:
            if ch.get_id() == chain_id:
                target_chain = ch
                break
    if target_chain is None:
        target_chain = next(iter(next(iter(structure)).get_chains()))
    pocket = []
    for res in target_chain:
        if not is_aa(res, standard=True):
            continue
        heavy_atoms = [a for a in res.get_atoms() if atom_is_heavy(a)]
        if not heavy_atoms:
            continue
        if contact_cut is not None:
            dmin = min(np.linalg.norm(a.get_coord()-b.get_coord()) for a in heavy_atoms for b in lig_atoms)
            if dmin <= contact_cut:
                pocket.append(res)
        else:
            dmin = min(np.linalg.norm(a.get_coord()-center) for a in heavy_atoms)
            if dmin <= radius:
                pocket.append(res)
    return pocket

def residues_to_sequences(res_list):
    seq3 = [r.get_resname().strip() for r in res_list]
    seq1 = "".join(THREE_TO_ONE.get(x, 'X') for x in seq3)
    idx_pdb = [r.get_id()[1] for r in res_list]
    return seq3, seq1, idx_pdb

# ==============================
# 3) Escritura incremental
# ==============================
def append_rows_safely(rows, out_csv):
    if not rows:
        return
    df_chunk = pd.DataFrame(rows)
    file_exists = os.path.exists(out_csv)
    df_chunk.to_csv(out_csv, mode="a", header=not file_exists, index=False)
    rows.clear()

# ==============================
# 4) Loop principal
# ==============================
rows_buffer = []
processed = 0

for _, rec in tqdm(df_iter.iterrows(), total=len(df_iter)):
    seq = rec["seq"]
    smi = rec["smiles"]
    rid = rec.get("id", str(_))

    try:
        hits = rcsb_sequence_search(seq, identity_cutoff=MIN_IDENT, evalue_cutoff=1e-5, max_results=3)
        if not hits:
            continue
    except Exception as e:
        print(f"[WARN] id={rid} fallo en búsqueda: {e}")
        continue

    hit = None
    for h in hits:
        cov_ok = (h["coverage"] is None) or (h["coverage"] >= MIN_COV)
        if cov_ok:
            hit = h; break
    if hit is None:
        hit = hits[0]
    pdb_id = hit["pdb_id"].upper()

    try:
        cif_path = download_mmcif(pdb_id, out_dir=PDB_DIR)
        parser = MMCIFParser(QUIET=True)
        structure = parser.get_structure(pdb_id, cif_path)
    except Exception as e:
        print(f"[WARN] id={rid} pdb={pdb_id} fallo al descargar/parsear: {e}")
        continue

    ligs = list_ligands_by_entity(structure)
    if not ligs:
        continue
    model_idx, chain_id, lig_res, lig_name, heavy = max(ligs, key=lambda x: x[4])

    pocket_res = pocket_from_ligand(structure, chain_id, lig_res, radius=RADIUS_ANG)
    if not pocket_res:
        continue

    seq3, seq1, idx_pdb = residues_to_sequences(pocket_res)

    rows_buffer.append({
        "id": rid,
        "pdb_id": pdb_id,
        "chain_id": chain_id,
        "ligand_name": lig_name,
        "ligand_heavy_atoms": heavy,
        "pocket_seq_3L": " ".join(seq3),
        "pocket_seq_1L": seq1,
        "ligand_smiles": smi,
        "pChEMBL": rec.get("neg_log10_affinity_M", None)
    })
    processed += 1

    if processed % FLUSH_EVERY == 0:
        append_rows_safely(rows_buffer, OUT_CSV)

append_rows_safely(rows_buffer, OUT_CSV)

print(f"✅ Proceso terminado. CSV incremental en: {OUT_CSV}")
