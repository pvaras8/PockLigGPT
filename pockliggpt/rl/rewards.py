import os
import subprocess
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd
import selfies as sf
from numpy import exp
from rdkit import Chem
from rdkit.Chem import Descriptors


# =========================
# helpers químicos
# =========================

def calculate_logp(smiles: str) -> float:
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        return float(Descriptors.MolLogP(mol))
    return 0.0


def calculate_weight(smiles: str) -> float:
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        return float(Descriptors.MolWt(mol))
    return 0.0


# =========================
# funciones de reward
# =========================

def penalize_logP(logP, sigma=1.0):
    val = np.exp(-((logP - 2) ** 2) / (2 * sigma**2))
    if isinstance(logP, (int, float)):
        if logP < 0 or logP > 5:
            return 0.0
        return val
    val = np.where((logP < 0) | (logP > 5), 0.0, val)
    return val


def sigmoid_pen_docking(x, min_x=-12.0, max_x=-6.0, min_sig_in=-6, max_sig_in=6):
    sig = lambda y: 1 / (1 + exp(-y))
    input_map = lambda y: (y - min_x) / (max_x - min_x) * (max_sig_in - min_sig_in) + min_sig_in
    return sig(input_map(x))


def sigmoid_pen_pchembl(x, min_x=5.0, max_x=7.5, min_sig_in=-6, max_sig_in=6):
    sig = lambda y: 1 / (1 + exp(-y))
    input_map = lambda y: (y - min_x) / (max_x - min_x) * (max_sig_in - min_sig_in) + min_sig_in
    return sig(input_map(x))


def sigmoid_pen_reward(x, min_x=300.0, max_x=500, min_sig_in=-6, max_sig_in=6):
    sig = lambda y: 1 / (1 + exp(-y))
    input_map = lambda y: (y - min_x) / (max_x - min_x) * (max_sig_in - min_sig_in) + min_sig_in
    return sig(input_map(x))


def sigmoid_pen_delta(x, ref, min_x=-3.0, max_x=3.0, min_sig_in=-6, max_sig_in=6):
    delta = x - ref
    z = ((delta - min_x) / (max_x - min_x)) * (max_sig_in - min_sig_in) + min_sig_in
    return z


def combine_docking_only(d):
    return 1 - sigmoid_pen_docking(d)


def combine_docking_logP(docking, logp):
    def docking_component(d, d_min=-10.0, d_max=-4.0, s_min=-6.0, s_max=6.0, power=1.3):
        z = (d - d_min) / (d_max - d_min) * (s_max - s_min) + s_min
        base = 1.0 - (1.0 / (1.0 + np.exp(-z)))
        return np.clip(base, 0.0, 1.0) ** power

    def logp_component(lp, mu=2.5, sigma=1.0):
        return np.exp(-((lp - mu) ** 2) / (2.0 * sigma**2))

    d_comp = docking_component(docking)
    lp_comp = logp_component(logp)

    w_lp = 0.5
    bias = 0.2
    eps_floor = 0.02

    mod = (bias + w_lp * lp_comp) / (bias + w_lp)
    reward = np.clip(d_comp * mod, eps_floor, 1.0)
    return float(reward)


def combine_docking_logP_mw(docking, logp, mw):
    def sigmoid(x, center, width):
        return 1.0 / (1.0 + np.exp(-(x - center) / width))

    def docking_component(d, d_min=-12.0, d_max=-6.0, s_min=-6.0, s_max=6.0, power=1.3):
        z = (d - d_min) / (d_max - d_min) * (s_max - s_min) + s_min
        base = 1.0 - (1.0 / (1.0 + np.exp(-z)))
        return np.clip(base, 0, 1) ** power

    def logp_component(lp, mu=2.5, sigma=1.0):
        return np.exp(-((lp - mu) ** 2) / (2.0 * sigma**2))

    def mw_component(mw, center=500.0, width=60.0):
        sig = sigmoid(mw, center=center, width=width)
        return 1.0 - sig

    d_comp = docking_component(docking)
    lp_comp = logp_component(logp)
    mw_comp = mw_component(mw)

    w_lp, w_mw, bias = 0.4, 0.4, 0.3
    eps_floor = 0.02

    mod = (bias + w_lp * lp_comp + w_mw * mw_comp) / (bias + w_lp + w_mw)
    reward = np.clip(d_comp * mod, eps_floor, 1.0)
    return float(reward)


def combine_two_docking_specificity(d1, d2, k=25.0, tau=0.20, eps_floor=0.02):
    s1 = 1.0 - sigmoid_pen_docking(d1)
    s2 = 1.0 - sigmoid_pen_docking(d2)

    core = np.clip(s1 * (1.0 - s2), 0.0, 1.0)
    gate = 1.0 / (1.0 + np.exp(-k * ((s1 - s2) - tau)))

    r0 = np.clip(core * gate, 0.0, 1.0)
    r = eps_floor + (1.0 - eps_floor) * r0

    if isinstance(d1, (int, float)) and isinstance(d2, (int, float)):
        return float(r)
    return r


def combine_two_docking_specific_2(d1, d2, ref1, ref2, eps_floor=0.02):
    sig = lambda y: 1 / (1 + exp(-y))
    z1 = sigmoid_pen_delta(d1, ref1)
    z2 = sigmoid_pen_delta(d2, ref2)
    delta = z1 - z2

    r_spec = 1 - sig(delta)
    r_target = 1 - sigmoid_pen_docking(d1)

    r0 = r_spec * r_target
    r = eps_floor + (1 - eps_floor) * r0
    return float(r)


def combine_two_docking_specific_new(d1, d2, ref1, ref2, eps_floor=0.02):
    sig = lambda y: 1 / (1 + exp(-y))
    z1 = sigmoid_pen_delta(d1, ref1)
    z2 = sigmoid_pen_delta(d2, ref2)
    delta = z1 - z2

    gap = d2 - d1
    tau_gap = 1.0

    if (d2 < ref2) and (gap < tau_gap):
        return float(eps_floor)

    r_spec = 1 - sig(delta)
    r_target = 1 - sigmoid_pen_docking(d1)

    r0 = r_spec * r_target
    r = eps_floor + (1 - eps_floor) * r0
    return float(r)


COMBINERS: Dict[str, Callable] = {
    "docking_only": combine_docking_only,
    "docking_logp": combine_docking_logP,
    "docking_logp_mw": combine_docking_logP_mw,
    "two_docking_specificity": combine_two_docking_specificity,
    "two_docking_specific_2": combine_two_docking_specific_2,
    "two_docking_specific_new": combine_two_docking_specific_new,
    "mw_only": sigmoid_pen_reward,
}


class RewardRunner:
    def __init__(self, cfg):
        self.cfg = cfg
        self.reward_cfg = cfg["reward"]

        script = self.reward_cfg.get("script")
        script_2 = self.reward_cfg.get("script_2")
        vars_file = self.reward_cfg.get("vars_file")
        vars_file_2 = self.reward_cfg.get("vars_file_2")

        self.reward_script = os.path.abspath(script) if script else None
        self.reward_script_2 = os.path.abspath(script_2) if script_2 else None
        self.vars_file = os.path.abspath(vars_file) if vars_file else None
        self.vars_file_2 = os.path.abspath(vars_file_2) if vars_file_2 else None

        self.smiles_output_file = os.path.abspath(self.reward_cfg["smiles_output_file"])
        self.output_prefix = self.reward_cfg.get("output_prefix", "reward_results")
        self.combiner_name = self.reward_cfg["combiner"]

        if self.combiner_name not in COMBINERS:
            raise ValueError(f"Combiner desconocido: {self.combiner_name}")

        self.combiner = COMBINERS[self.combiner_name]

    def _decode_selfies(self, molecules_selfies: List[str]) -> List[Optional[str]]:
        molecules_smiles = []
        for selfie in molecules_selfies:
            try:
                smiles = sf.decoder(selfie)
                if smiles is None or not isinstance(smiles, str) or len(smiles.strip()) == 0:
                    smiles = None
                molecules_smiles.append(smiles)
            except Exception:
                molecules_smiles.append(None)
        return molecules_smiles

    def _prepare_smiles_file(self, smiles_list: List[str]) -> None:
        with open(self.smiles_output_file, "w") as f:
            for smi in smiles_list:
                f.write(smi + "\n")

    def _run_reward_script(
        self,
        script_path: str,
        vars_file: Optional[str],
        epoch: int,
    ) -> pd.DataFrame:
        if vars_file is None:
            raise ValueError(f"vars_file es obligatorio para script {script_path}")

        subprocess.run(
            ["python", script_path, self.smiles_output_file, vars_file, str(epoch)],
            check=True,
        )

        output_file_temp = os.path.join(
            os.path.dirname(self.smiles_output_file),
            f"{self.output_prefix}_{epoch}_temp.csv",
        )

        if not os.path.exists(output_file_temp):
            raise FileNotFoundError(f"No se encontró el archivo temporal: {output_file_temp}")

        df = pd.read_csv(output_file_temp)
        return df

    def _run_reward_script_with_suffix(
        self,
        script_path: str,
        vars_file: Optional[str],
        epoch: int,
        suffix: str,
    ) -> pd.DataFrame:
        if vars_file is None:
            raise ValueError(f"vars_file es obligatorio para script {script_path}")

        subprocess.run(
            ["python", script_path, self.smiles_output_file, vars_file, str(epoch)],
            check=True,
        )

        output_file_temp = os.path.join(
            os.path.dirname(self.smiles_output_file),
            f"{self.output_prefix}_{suffix}_{epoch}_temp.csv",
        )

        if not os.path.exists(output_file_temp):
            raise FileNotFoundError(f"No se encontró el archivo temporal: {output_file_temp}")

        df = pd.read_csv(output_file_temp)
        return df

    def _build_base_df(self, smiles_list: List[str]) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "input_idx": np.arange(len(smiles_list)),
                "SMILES": smiles_list,
            }
        )

    def _assign_single_docking_by_position(
        self,
        base_df: pd.DataFrame,
        df_docking: pd.DataFrame,
    ) -> pd.DataFrame:
        if "Docking" not in df_docking.columns:
            raise ValueError("El CSV de docking no contiene columna 'Docking'")

        if len(df_docking) != len(base_df):
            raise RuntimeError(
                f"Docking devolvió {len(df_docking)} filas, esperaba {len(base_df)}"
            )

        df_temp = base_df.copy()
        df_temp["Docking"] = df_docking["Docking"].to_numpy()

        if "SMILES" in df_docking.columns:
            returned_smiles = df_docking["SMILES"].astype(str).tolist()
            expected_smiles = df_temp["SMILES"].astype(str).tolist()
            if returned_smiles != expected_smiles:
                raise RuntimeError(
                    "El docking no devolvió los SMILES en el mismo orden que la entrada"
                )

        return df_temp

    def _assign_double_docking_by_position(
        self,
        base_df: pd.DataFrame,
        df_d1: pd.DataFrame,
        df_d2: pd.DataFrame,
    ) -> pd.DataFrame:
        if "Docking" not in df_d1.columns:
            raise ValueError("El CSV del docking 1 no contiene columna 'Docking'")
        if "Docking" not in df_d2.columns:
            raise ValueError("El CSV del docking 2 no contiene columna 'Docking'")

        if len(df_d1) != len(base_df):
            raise RuntimeError(
                f"Docking 1 devolvió {len(df_d1)} filas, esperaba {len(base_df)}"
            )
        if len(df_d2) != len(base_df):
            raise RuntimeError(
                f"Docking 2 devolvió {len(df_d2)} filas, esperaba {len(base_df)}"
            )

        df_temp = base_df.copy()
        df_temp["Docking_1"] = df_d1["Docking"].to_numpy()
        df_temp["Docking_2"] = df_d2["Docking"].to_numpy()

        if "SMILES" in df_d1.columns:
            returned_smiles_1 = df_d1["SMILES"].astype(str).tolist()
            expected_smiles = df_temp["SMILES"].astype(str).tolist()
            if returned_smiles_1 != expected_smiles:
                raise RuntimeError(
                    "Docking 1 no devolvió los SMILES en el mismo orden que la entrada"
                )

        if "SMILES" in df_d2.columns:
            returned_smiles_2 = df_d2["SMILES"].astype(str).tolist()
            expected_smiles = df_temp["SMILES"].astype(str).tolist()
            if returned_smiles_2 != expected_smiles:
                raise RuntimeError(
                    "Docking 2 no devolvió los SMILES en el mismo orden que la entrada"
                )

        return df_temp

    def _apply_combiner(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.combiner_name == "docking_only":
            if "Docking" not in df.columns:
                raise ValueError("Falta columna 'Docking' para combiner docking_only")
            df["Fitness"] = df["Docking"].apply(self.combiner)

        elif self.combiner_name == "docking_logp":
            if "Docking" not in df.columns:
                raise ValueError("Falta columna 'Docking' para combiner docking_logp")
            df["LogP"] = df["SMILES"].apply(calculate_logp)
            df["Fitness"] = df.apply(
                lambda row: self.combiner(row["Docking"], row["LogP"]),
                axis=1,
            )

        elif self.combiner_name == "docking_logp_mw":
            if "Docking" not in df.columns:
                raise ValueError("Falta columna 'Docking' para combiner docking_logp_mw")
            df["LogP"] = df["SMILES"].apply(calculate_logp)
            df["MW"] = df["SMILES"].apply(calculate_weight)
            df["Fitness"] = df.apply(
                lambda row: self.combiner(row["Docking"], row["LogP"], row["MW"]),
                axis=1,
            )

        elif self.combiner_name == "mw_only":
            df["MW"] = df["SMILES"].apply(calculate_weight)
            df["Fitness"] = df["MW"].apply(self.combiner)

        elif self.combiner_name == "two_docking_specificity":
            required = ["Docking_1", "Docking_2"]
            for col in required:
                if col not in df.columns:
                    raise ValueError(f"Falta columna '{col}' para combiner {self.combiner_name}")
            df["Fitness"] = df.apply(
                lambda row: self.combiner(row["Docking_1"], row["Docking_2"]),
                axis=1,
            )

        elif self.combiner_name in {"two_docking_specific_2", "two_docking_specific_new"}:
            required = ["Docking_1", "Docking_2"]
            for col in required:
                if col not in df.columns:
                    raise ValueError(f"Falta columna '{col}' para combiner {self.combiner_name}")

            ref1 = self.reward_cfg.get("ref1")
            ref2 = self.reward_cfg.get("ref2")
            if ref1 is None or ref2 is None:
                raise ValueError(
                    f"Los combiners '{self.combiner_name}' requieren reward.ref1 y reward.ref2"
                )

            df["Fitness"] = df.apply(
                lambda row: self.combiner(row["Docking_1"], row["Docking_2"], ref1, ref2),
                axis=1,
            )

        else:
            raise NotImplementedError(f"Combiner '{self.combiner_name}' no implementado")

        return df

    def _save_epoch_results(self, df_temp: pd.DataFrame, epoch: int) -> None:
        global_output_file = os.path.join(
            os.path.dirname(self.smiles_output_file),
            f"{self.output_prefix}_{epoch}.csv",
        )

        if os.path.exists(global_output_file):
            df_global = pd.read_csv(global_output_file)
            df_final = pd.concat([df_global, df_temp], ignore_index=True)
        else:
            df_final = df_temp

        df_final.to_csv(global_output_file, index=False)

    def _compute_rewards_df(self, smiles_list: List[str], epoch: int) -> pd.DataFrame:
        base_df = self._build_base_df(smiles_list)

        if self.combiner_name == "mw_only":
            df_temp = base_df.copy()

        elif self.combiner_name in {"docking_only", "docking_logp", "docking_logp_mw"}:
            if self.reward_script is None:
                raise ValueError(
                    f"reward.script es obligatorio para combiner '{self.combiner_name}'"
                )

            print(f"Llamando a {self.reward_script} para docking en época {epoch}...")
            df_docking = self._run_reward_script(
                script_path=self.reward_script,
                vars_file=self.vars_file,
                epoch=epoch,
            )

            df_temp = self._assign_single_docking_by_position(base_df, df_docking)

        elif self.combiner_name in {
            "two_docking_specificity",
            "two_docking_specific_2",
            "two_docking_specific_new",
        }:
            if self.reward_script is None or self.reward_script_2 is None:
                raise ValueError(
                    f"Los combiners '{self.combiner_name}' requieren reward.script y reward.script_2"
                )

            print(f"Llamando a {self.reward_script} para docking 1 en época {epoch}...")
            df_d1 = self._run_reward_script_with_suffix(
                script_path=self.reward_script,
                vars_file=self.vars_file,
                epoch=epoch,
                suffix="1",
            )

            print(f"Llamando a {self.reward_script_2} para docking 2 en época {epoch}...")
            df_d2 = self._run_reward_script_with_suffix(
                script_path=self.reward_script_2,
                vars_file=self.vars_file_2,
                epoch=epoch,
                suffix="2",
            )

            df_temp = self._assign_double_docking_by_position(base_df, df_d1, df_d2)

        else:
            raise NotImplementedError(f"Combiner '{self.combiner_name}' no soportado")

        df_temp = self._apply_combiner(df_temp)
        return df_temp

    def __call__(self, molecules_selfies: List[str], epoch: int) -> List[float]:
        molecules_smiles = self._decode_selfies(molecules_selfies)

        if len(molecules_smiles) == 0:
            return []

        valid_pairs = [
            (idx, smi) for idx, smi in enumerate(molecules_smiles) if smi is not None
        ]

        if len(valid_pairs) == 0:
            return [0.0] * len(molecules_selfies)

        valid_indices = [idx for idx, _ in valid_pairs]
        valid_smiles = [smi for _, smi in valid_pairs]

        self._prepare_smiles_file(valid_smiles)

        try:
            df_temp = self._compute_rewards_df(valid_smiles, epoch)
            self._save_epoch_results(df_temp, epoch)

            valid_rewards = (
                df_temp.sort_values("input_idx")["Fitness"]
                .fillna(0.0)
                .astype(float)
                .tolist()
            )

            if len(valid_rewards) != len(valid_smiles):
                raise RuntimeError(
                    f"Longitud de rewards válidas incorrecta: {len(valid_rewards)} vs {len(valid_smiles)}"
                )

            rewards = [0.0] * len(molecules_selfies)
            for original_idx, reward in zip(valid_indices, valid_rewards):
                rewards[original_idx] = float(reward)

            return rewards

        except Exception as e:
            print(f"Error durante reward_fn: {e}")
            return [0.0] * len(molecules_selfies)