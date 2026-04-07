import json
import math
import os
import sys
from multiprocessing import Pool, cpu_count
from typing import List, Tuple, Optional

import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from meeko import MoleculePreparation
from vina import Vina


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        cfg = json.load(f)

    required = [
        "filename_of_receptor",
        "center_x",
        "center_y",
        "center_z",
        "size_x",
        "size_y",
        "size_z",
        "final_folder",
    ]
    missing = [k for k in required if k not in cfg]
    if missing:
        raise ValueError(f"Faltan claves en vars.json: {missing}")

    return cfg


def read_smiles_file(smiles_path: str) -> List[str]:
    smiles = []
    with open(smiles_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            smi = line.split()[0]
            smiles.append(smi)
    return smiles


def smiles_to_3d_mol(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("SMILES inválido")

    mol = Chem.AddHs(mol)

    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    status = AllChem.EmbedMolecule(mol, params)
    if status != 0:
        raise RuntimeError("No se pudo generar conformero 3D")

    try:
        mmff_props = AllChem.MMFFGetMoleculeProperties(mol)
        if mmff_props is not None:
            AllChem.MMFFOptimizeMolecule(mol)
        else:
            AllChem.UFFOptimizeMolecule(mol)
    except Exception:
        try:
            AllChem.UFFOptimizeMolecule(mol)
        except Exception:
            pass

    return mol


def mol_to_pdbqt_string(mol) -> str:
    preparator = MoleculePreparation()
    preparator.prepare(mol)

    if hasattr(preparator, "write_pdbqt_string"):
        return preparator.write_pdbqt_string()

    raise RuntimeError(
        "Tu versión de Meeko no soporta write_pdbqt_string(). "
        "Actualiza Meeko o adapta este bloque."
    )


def dock_smiles(
    smiles: str,
    receptor_pdbqt: str,
    center: List[float],
    box_size: List[float],
    exhaustiveness: int,
    n_poses: int,
    vina_cpu: int,
) -> float:
    mol = smiles_to_3d_mol(smiles)
    ligand_pdbqt = mol_to_pdbqt_string(mol)

    v = Vina(sf_name="vina", cpu=vina_cpu)
    v.set_receptor(receptor_pdbqt)
    v.set_ligand_from_string(ligand_pdbqt)
    v.compute_vina_maps(center=center, box_size=box_size)
    v.dock(exhaustiveness=exhaustiveness, n_poses=n_poses)

    energies = v.energies()
    if energies is None or len(energies) == 0:
        raise RuntimeError("Vina no devolvió energías")

    return float(energies[0][0])


WORKER_CFG = {}


def init_worker(cfg: dict):
    global WORKER_CFG
    WORKER_CFG = cfg


def worker(task: Tuple[int, str]):
    idx, smiles = task

    try:
        score = dock_smiles(
            smiles=smiles,
            receptor_pdbqt=WORKER_CFG["filename_of_receptor"],
            center=WORKER_CFG["center"],
            box_size=WORKER_CFG["box_size"],
            exhaustiveness=WORKER_CFG["exhaustiveness"],
            n_poses=WORKER_CFG["n_poses"],
            vina_cpu=WORKER_CFG["vina_cpu_per_job"],
        )
        return {
            "ID": str(idx + 1),
            "SMILES": smiles,
            "Docking": score,
            "Status": "OK",
        }
    except Exception as e:
        return {
            "ID": str(idx + 1),
            "SMILES": smiles,
            "Docking": WORKER_CFG["fallback_score"],
            "Status": f"ERROR: {str(e)}",
        }


def main():
    if len(sys.argv) != 4:
        raise SystemExit(
            "Uso: python reward_vina.py <source_compound_file> <vars.json> <epoch>"
        )

    source_compound_file = os.path.abspath(sys.argv[1])
    vars_json = os.path.abspath(sys.argv[2])
    epoch = str(sys.argv[3])

    cfg = load_config(vars_json)

    if not os.path.exists(source_compound_file):
        raise FileNotFoundError(f"No se encontró el archivo de SMILES: {source_compound_file}")

    receptor_pdbqt = os.path.abspath(cfg["filename_of_receptor"])
    if not os.path.exists(receptor_pdbqt):
        raise FileNotFoundError(f"No se encontró el receptor PDBQT: {receptor_pdbqt}")

    initial_smiles_list = read_smiles_file(source_compound_file)
    if len(initial_smiles_list) == 0:
        raise ValueError("El archivo de entrada no contiene SMILES")

    center = [
        float(cfg["center_x"]),
        float(cfg["center_y"]),
        float(cfg["center_z"]),
    ]
    box_size = [
        float(cfg["size_x"]),
        float(cfg["size_y"]),
        float(cfg["size_z"]),
    ]

    n_workers = int(cfg.get("num_processors", max(1, cpu_count() - 1)))
    if n_workers == -1:
        n_workers = max(1, cpu_count() - 1)

    n_workers = max(1, n_workers)

    vina_cpu_per_job = int(cfg.get("vina_cpu_per_job", 1))
    exhaustiveness = int(cfg.get("exhaustiveness", 8))
    n_poses = int(cfg.get("n_poses", 1))
    fallback_score = float(cfg.get("fallback_score", -6.0))
    save_every = int(cfg.get("save_every", 25))

    final_folder = os.path.abspath(cfg["final_folder"])
    os.makedirs(final_folder, exist_ok=True)

    intermediate_csv_path = os.path.join(
        final_folder, f"docking_results_{epoch}_temp.csv"
    )

    partial_csv_path = os.path.join(
        final_folder, f"docking_results_{epoch}_partial.csv"
    )

    worker_cfg = {
        "filename_of_receptor": receptor_pdbqt,
        "center": center,
        "box_size": box_size,
        "exhaustiveness": exhaustiveness,
        "n_poses": n_poses,
        "vina_cpu_per_job": vina_cpu_per_job,
        "fallback_score": fallback_score,
    }

    tasks = list(enumerate(initial_smiles_list))

    print("---------- 1. Prediccion Docking ----------")
    print(f"Moléculas: {len(initial_smiles_list)}")
    print(f"Workers: {n_workers}")
    print(f"Vina CPU/job: {vina_cpu_per_job}")
    print(f"Receptor: {receptor_pdbqt}")
    print(f"Centro: {center}")
    print(f"Box: {box_size}")

    results = []
    done = 0

    with Pool(processes=n_workers, initializer=init_worker, initargs=(worker_cfg,)) as pool:
        for result in pool.imap_unordered(worker, tasks):
            results.append(result)
            done += 1

            if result["Status"] == "OK":
                print(f"[{done}/{len(tasks)}] OK     {result['Docking']:.3f}   {result['SMILES'][:60]}")
            else:
                print(f"[{done}/{len(tasks)}] ERROR  {result['SMILES'][:60]} -> {result['Status']}")

            if done % save_every == 0:
                df_partial = pd.DataFrame(results)
                df_partial["ID_int"] = df_partial["ID"].astype(int)
                df_partial = df_partial.sort_values("ID_int").drop(columns=["ID_int"])
                df_partial.to_csv(partial_csv_path, index=False)
                print(f"Guardado parcial: {partial_csv_path}")

    df = pd.DataFrame(results)
    df["ID_int"] = df["ID"].astype(int)
    df = df.sort_values("ID_int").drop(columns=["ID_int"])

    df_out = df[["ID", "SMILES", "Docking"]].copy()
    df_out.to_csv(intermediate_csv_path, index=False)

    print(f"Resultados intermedios guardados en: {intermediate_csv_path}")

    debug_csv_path = os.path.join(final_folder, f"docking_results_{epoch}_debug.csv")
    df.to_csv(debug_csv_path, index=False)
    print(f"Resultados debug guardados en: {debug_csv_path}")


if __name__ == "__main__":
    main()
