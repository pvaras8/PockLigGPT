import numpy as np
from numpy import exp

# --- Sigmoid functions for different normalizations ---

def penalize_logP(logP, sigma=1.0):
    """
    Función gaussiana centrada en 2, forzada a 0 para logP < 0 o logP > 5.
    
    logP  : valor(es) donde evaluar la función (float o np.array).
    sigma : parámetro que controla la anchura de la campana gaussiana.
            A mayor sigma, más 'lenta' la caída fuera de 2.
    
    Retorna:
      - Un float si logP es escalar.
      - Un array si logP es un array de NumPy.
    """
    # Calculamos la gaussiana pura
    val = np.exp(-((logP - 2)**2)/(2 * sigma**2))
    
    # Si es un número escalar (float o int), forzamos a 0 fuera de [0, 5]
    if isinstance(logP, (int, float)):
        if logP < 0 or logP > 5:
            return 0.0
        return val
    
    # Si es un array de NumPy, forzamos a 0 todo lo que esté fuera de [0, 5]
    val = np.where((logP < 0) | (logP > 5), 0.0, val)
    return val

def sigmoid_pen_docking(x, min_x=-12.0, max_x=-6.0, min_sig_in=-6, max_sig_in=6):
    """
    Sigmoid function to normalize docking scores.
    Maps `x` to the range [0, 1].
    """
    sig = lambda y: 1 / (1 + exp(-y))
    input_map = lambda y: (y - min_x) / (max_x - min_x) * (max_sig_in - min_sig_in) + min_sig_in
    return sig(input_map(x))

def sigmoid_pen_pchembl(x, min_x=5.0, max_x=7.5, min_sig_in=-6, max_sig_in=6):
    """
    Sigmoid function to normalize pChEMBL values.
    Maps `x` to the range [0, 1].
    """
    sig = lambda y: 1 / (1 + exp(-y))
    input_map = lambda y: (y - min_x) / (max_x - min_x) * (max_sig_in - min_sig_in) + min_sig_in
    return sig(input_map(x))

def sigmoid_pen_reward(x, min_x=300.0, max_x=500, min_sig_in=-6, max_sig_in=6):
    """
    Sigmoid function for reward weighting.
    Maps `x` to the range [0, 1].
    """
    sig = lambda y: 1 / (1 + exp(-y))
    input_map = lambda y: (y - min_x) / (max_x - min_x) * (max_sig_in - min_sig_in) + min_sig_in
    return sig(input_map(x))

def sigmoid_pen_delta(x, ref, min_x=-3.0, max_x=3.0,
                      min_sig_in=-6, max_sig_in=6):
    delta = x - ref
    z = ((delta - min_x) / (max_x - min_x)) * (max_sig_in - min_sig_in) + min_sig_in
    return z

# --- Fitness functions ---

def combine_docking_pchembl(d, p, alpha=0.5, beta=0.5):
    """
    Combines docking and pChEMBL into a fitness score in the range [0, 1].
    Alpha and beta control the relative importance of each metric.
    """
    R_docking = 1 - sigmoid_pen_docking(d)  # Normalize docking
    R_pchembl = sigmoid_pen_pchembl(p)      # Normalize pChEMBL

    return alpha * R_docking + beta * R_pchembl  # Weighted combination

def combine_pchembl_only(p):
    """
    Computes fitness score based only on pChEMBL.
    """
    return sigmoid_pen_pchembl(p, min_x=5.5, max_x=7.5)  # Adjusted range

def reward_weight(p):
    """
    Computes fitness score based only on reward values.
    """
    return sigmoid_pen_reward(p)

def combine_docking_only(d):
    """
    Computes fitness score based only on docking.
    """
    return 1 - sigmoid_pen_docking(d)  # Inverted so lower docking is better

def combine_two_docking(d1, d2, alpha=0.5, beta=0.5):
    """
    Combines two docking scores into a single fitness value.
    Alpha and beta control the relative importance of each docking score.
    """
    R_docking_1 = 1 - sigmoid_pen_docking(d1)  # Normalize first docking
    R_docking_2 = 1 - sigmoid_pen_docking(d2)  # Normalize second docking

    return alpha * R_docking_1 + beta * R_docking_2  # Weighted combination

def combine_three_docking(d1, d2, d3, alpha=0.33, beta=0.33, gamma=0.34):
    """
    Combines three docking scores into a single fitness value.
    Alpha, beta, and gamma control the relative importance of each docking score.
    The weights should ideally sum to 1.
    """
    R_docking_1 = 1 - sigmoid_pen_docking(d1)  # Normalize first docking
    R_docking_2 = 1 - sigmoid_pen_docking(d2)  # Normalize second docking
    R_docking_3 = 1 - sigmoid_pen_docking(d3)  # Normalize third docking

    return alpha * R_docking_1 + beta * R_docking_2 + gamma * R_docking_3  # Weighted combination

# def combine_docking_logP(docking, logp, alpha=0.5, beta=0.5):
#     """
#     Combina el docking y el logP como fitness.
#     Se favorecen docking bajos y logP en el rango [1.5, 2.5].
#     """
#     R_docking = 1 - sigmoid_pen_docking(docking)  # Docking más negativo = mejor
#     R_logp = penalize_logP(logp)               # LogP óptimo ≈ 2.0

#     return alpha * R_docking + beta * R_logp

def combine_docking_scaffold(docking, scaffold, alpha=0.5, beta=0.5):
    """
    Combina la puntuación de docking con la presencia de una subestructura (scaffold).
    
    Args:
        docking (float): Valor del docking (por ejemplo, -11.4).
        scaffold (int or float): Indicador de presencia del scaffold; se espera 1 si está presente, 0 en caso contrario.
        
    Returns:
        float: Valor de fitness. Si scaffold es 1 se retorna 0.5 * (1 - sigmoid_pen_docking(docking)), de lo contrario 0.
    """
    R_docking = 1 - sigmoid_pen_docking(docking)  # Calcular el fitness por docking
    return alpha * R_docking + beta * scaffold

def norm_rotb(rotb, k_opt=4, k_hard=6):
    """1.0 si RotB ≤ k_opt; 0.0 si RotB ≥ k_hard; lineal entre ambos."""
    if rotb is None:
        return 0.0
    if rotb <= k_opt:
        return 1.0
    if rotb >= k_hard:
        return 0.0
    return 1 - (rotb - k_opt) / (k_hard - k_opt)

def combine_rotb_only(rotb, k_opt=4, k_hard=6):
    """
    Devuelve un fitness 0–1 basado exclusivamente en el nº de enlaces rotables.
    • Óptimo (1.0)   si RotB ≤ k_opt
    • Cae linealmente hasta 0.0 en k_hard
    • 0.0 si RotB ≥ k_hard o rotb == None
    """
    return norm_rotb(rotb, k_opt=k_opt, k_hard=k_hard)

# ---------------------------
# 1) FUNCIÓN DE COMBINACIÓN
# ---------------------------

def combine_docking_rotb(docking, rotb, alpha=0.7, beta=0.3,
                         k_opt=4, k_hard=6,
                         min_x=-12.0, max_x=-6.0):
    # --- 1. Docking normalizado (0 ≈ malo … 1 ≈ bueno) ---
    R_docking = 1 - sigmoid_pen_docking(docking,
                                        min_x=min_x,
                                        max_x=max_x)

    # --- 2. RotB normalizado (1.0 si ≤ k_opt, 0.0 si ≥ k_hard) ---
    R_rotb = norm_rotb(rotb, k_opt=k_opt, k_hard=k_hard)

    # --- 3. Combinación ponderada ---
    return alpha * R_docking + beta * R_rotb

def boltz(p, t=0.35, k=10, eps=0.02):
    """
    • p : probabilidad 0‑1 del modelo
    • t : umbral de binder deseado
    • k : pendiente de la sigmoide
    • eps: recompensa mínima para no perder gradiente
    """
    gate = 1.0 / (1.0 + np.exp(-k * (p - t)))
    return eps + (1.0 - eps) * gate



def combine_docking_logP_mw(docking, logp, mw):
    """
    Calcula una reward suave combinando Docking, LogP y MW.
    
    - Docking domina la reward (más negativo = mejor).
    - LogP tiene forma gaussiana con máximo en 2.5.
    - MW penaliza progresivamente conforme aumenta (sigmoide decreciente centrada en 500 Da).
    """
    # === Funciones internas ===
    def sigmoid(x, center, width):
        return 1.0 / (1.0 + np.exp(-(x - center) / width))
    
    # --- Docking ---
    def docking_component(d, d_min=-12.0, d_max=-6.0, s_min=-6.0, s_max=6.0, power=1.3):
        z = (d - d_min) / (d_max - d_min) * (s_max - s_min) + s_min
        base = 1.0 - (1.0 / (1.0 + np.exp(-z)))
        return np.clip(base, 0, 1) ** power
    
    # --- LogP ---
    def logp_component(lp, mu=2.5, sigma=1.0):
        return np.exp(-((lp - mu)**2) / (2.0 * sigma**2))
    
    # --- MW ---
    def mw_component(mw, center=500.0, width=60.0):
        sig = sigmoid(mw, center=center, width=width)
        return 1.0 - sig  # penaliza conforme sube MW
    
    # === Cálculo de cada parte ===
    d_comp  = docking_component(docking)
    lp_comp = logp_component(logp)
    mw_comp = mw_component(mw)
    
    # === Combinación final (ponderada y con bias para mantener señal) ===
    w_lp, w_mw, bias = 0.4, 0.4, 0.3
    eps_floor = 0.02
    
    mod = (bias + w_lp * lp_comp + w_mw * mw_comp) / (bias + w_lp + w_mw)
    reward = np.clip(d_comp * mod, eps_floor, 1.0)
    
    return reward

def combine_docking_logP(docking, logp):
    """
    Calcula una reward suave combinando únicamente:
    - Docking   (más negativo = mejor, domina la reward)
    - LogP      (gaussiana con máximo en 2.5, sigma estrecha para frenar logP altos)
    
    MW NO SE USA porque ya se regula solo en tu modelo.
    """
    # --- Docking ---
    def docking_component(d, d_min=-10.0, d_max=-4.0, s_min=-6.0, s_max=6.0, power=1.3):
        """
        Normaliza docking a [0,1] con sigmoide invertida.
        Elevamos a 'power' para que Docking domine la reward.
        """
        z = (d - d_min) / (d_max - d_min) * (s_max - s_min) + s_min
        base = 1.0 - (1.0 / (1.0 + np.exp(-z)))   # 1 - sigmoide → más negativo = mejor
        return np.clip(base, 0.0, 1.0) ** power

    # --- LogP ---
    def logp_component(lp, mu=2.5, sigma=1.0):
        """
        Gaussiana MUY estrecha (sigma=1) para penalizar LogP altos.
        lp=2.5 → 1.0
        lp=4.0 → ~0.12
        lp=5.0 → ~0.007
        """
        return np.exp(-((lp - mu)**2) / (2.0 * sigma**2))

    # === Cálculo de cada parte ===
    d_comp  = docking_component(docking)
    lp_comp = logp_component(logp)

    # === Combinación final (Docking domina; LogP modula) ===
    w_lp  = 0.5   # peso del LogP
    bias  = 0.2   # evita reward = 0 y mantiene señal
    eps_floor = 0.02

    # modulador = combinación suave del LogP
    mod = (bias + w_lp * lp_comp) / (bias + w_lp)

    # reward final = docking dominante * modulador de logP
    reward = np.clip(d_comp * mod, eps_floor, 1.0)

    return float(reward)

def combine_two_docking_specificity(d1, d2, k=25.0, tau=0.20, eps_floor=0.02):
    """
    Combina dos dockings para medir ESPECIFICIDAD, en rango [eps_floor, 1.0].

    - Premia: d1 muy negativo (target deseado fuerte) y d2 poco negativo (off-target débil).
    - Penaliza: no-especificidad (d2 muy negativo) y falta de margen (s1 - s2 < tau).
    - Nunca devuelve 0 (piso eps_floor) para mantener señal en PPO/RL.

    Args:
        d1 (float or np.ndarray): Docking del target deseado (más negativo = mejor).
        d2 (float or np.ndarray): Docking del off-target (más negativo = peor).
        k (float): pendiente de la sigmoide de margen (dureza).
        tau (float): margen mínimo exigido en la escala normalizada (s1 - s2).
        eps_floor (float): recompensa mínima (piso) para evitar 0.

    Returns:
        float or np.ndarray: Reward de especificidad en [eps_floor, 1.0].
    """
    # Escala “buena”: más negativo => mayor (0..1)
    s1 = 1.0 - sigmoid_pen_docking(d1)  # target deseado
    s2 = 1.0 - sigmoid_pen_docking(d2)  # off-target (alto = indeseable)

    # Núcleo: exige T1 alto y T2 bajo
    core = np.clip(s1 * (1.0 - s2), 0.0, 1.0)

    # Puerta de margen: exige diferencia s1 - s2 >= tau
    gate = 1.0 / (1.0 + np.exp(-k * ((s1 - s2) - tau)))

    # Combinación y piso
    R0 = np.clip(core * gate, 0.0, 1.0)
    R  = eps_floor + (1.0 - eps_floor) * R0

    # Devuelve float si la entrada fue escalar
    if isinstance(d1, (int, float)) and isinstance(d2, (int, float)):
        return float(R)
    return R

def combine_two_docking_specific(d1, d2, ref1, ref2):
    sig = lambda y: 1 / (1 + exp(-y))
    z1 = sigmoid_pen_delta(d1, ref1)
    z2 = sigmoid_pen_delta(d2, ref2)
    delta = z1 - z2
    reward = float(1 - sig(delta))
    return reward

def combine_two_docking_specific_2 (d1, d2, ref1, ref2, eps_floor=0.02):
    """
    Nueva fórmula:
    - R_especificidad = 1 - sig(z1 - z2)
    - R_target = 1 - sigmoid_pen_docking(d1)
    - Reward = R_especificidad * R_target  (con piso)
    """

    sig = lambda y: 1 / (1 + exp(-y))

    # 1) Especificidad relativa (tu fórmula original)
    z1 = sigmoid_pen_delta(d1, ref1)
    z2 = sigmoid_pen_delta(d2, ref2)
    delta = z1 - z2
    R_spec = 1 - sig(delta)

    # 2) Afinidad absoluta del target
    R_target = 1 - sigmoid_pen_docking(d1)

    # 3) Combinación
    R0 = R_spec * R_target
    R  = eps_floor + (1 - eps_floor) * R0

    return float(R)

def combine_two_docking_specific_new(d1, d2, ref1, ref2, eps_floor=0.02):
    """
    Tu fórmula original mejorada, con una regla sencilla añadida:

    - Si el off-target (d2) es demasiado bueno (d2 < ref2)
      Y la diferencia d2 - d1 no es suficientemente grande (gap < tau_gap),
      se pone la reward al mínimo (eps_floor).

    - En caso contrario, se usa la fórmula original:
        R_especificidad = 1 - sig(z1 - z2)
        R_target        = 1 - sigmoid_pen_docking(d1)
        Reward          = eps_floor + (1 - eps_floor) * (R_spec * R_target)
    """
    sig = lambda y: 1 / (1 + exp(-y))

    # Normalización como antes
    z1 = sigmoid_pen_delta(d1, ref1)
    z2 = sigmoid_pen_delta(d2, ref2)
    delta = z1 - z2

    # --- NUEVA REGLA SENCILLA ---
    gap = d2 - d1      # cuanto más grande, mejor target frente a off-target
    tau_gap = 1.0      # diferencia mínima que consideramos "muy buena" (kcal/mol)

    if (d2 < ref2) and (gap < tau_gap):
        # Off-target demasiado bueno y sin suficiente diferencia → no premiamos
        R_spec = 0.0
        R_target = 0.0
        R = eps_floor
        return float(R)

    # --- FÓRMULA ORIGINAL (sin cambios) ---
    R_spec = 1 - sig(delta)
    R_target = 1 - sigmoid_pen_docking(d1)

    R0 = R_spec * R_target
    R  = eps_floor + (1 - eps_floor) * R0

    return float(R)