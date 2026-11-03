"""
Modelagem e simulação de um motor de indução trifásico (rotor em gaiola).

O modelo usa a transformação para eixos dq (Park), em referencial que gira
na velocidade síncrona (ωd = ωsyn). Nesse referencial, com alimentação
senoidal equilibrada, as tensões viram constantes (CC). Convenção do
Mohan, "Advanced Electric Drives", Cap. 3 (transformação invariante em
potência, eixo d alinhado com a fase a em t = 0).

Estados: fluxos concatenados  λsd, λsq, λrd, λrq  e  ωmec

    dλsd/dt = vsd - Rs·isd + ωd·λsq
    dλsq/dt = vsq - Rs·isq - ωd·λsd
    dλrd/dt =     - Rr·ird + (ωd - ωm)·λrq
    dλrq/dt =     - Rr·irq - (ωd - ωm)·λrd

    Fluxos x correntes:  λs = Ls·is + Lm·ir ,  λr = Lm·is + Lr·ir
    Torque:   Te = (p/2)·(λsd·isq - λsq·isd)
    Mecânica: J·dωmec/dt = Te - B·ωmec - TL ,   ωm = (p/2)·ωmec (elétrica)

O modelo é NÃO LINEAR (produtos ωm·λ e λ·i). Ele é integrado
numericamente e depois linearizado no ponto de operação nominal.

Parâmetros padrão: motor de 3 HP do Mohan, Exemplo 3-1, que serve para
validar o programa (o livro dá Te nominal = 12,644 N.m e isd = 5,34 A,
isq = -3,7 A, ird = -5,5 A, irq = 0,60 A).
"""

import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
from scipy.integrate import solve_ivp

# Dados de placa e circuito equivalente por fase (Mohan, Exemplo 3-1)
V_LL = 220  # Tensão de linha (V rms)
f = 60  # Frequência (Hz)
p = 4  # Número de polos
Rs = 1.77  # Resistência do estator (ohm)
Rr = 1.34  # Resistência do rotor referida ao estator (ohm)
Xls = 5.25  # Reatância de dispersão do estator a 60 Hz (ohm)
Xlr = 4.57  # Reatância de dispersão do rotor a 60 Hz (ohm)
Xm = 139.0  # Reatância de magnetização a 60 Hz (ohm)
J = 0.025  # Momento de inércia total (kg.m²)
B = 0.0  # Atrito viscoso (N.m.s/rad)
s_nominal = 0.0172  # Escorregamento nominal

# Parâmetros da simulação
t_carga = 1.1  # Instante de aplicação da carga nominal (s)
fator_carga = 1.0  # Carga aplicada como fração do torque nominal
t_final = 4.0  # Tempo total de simulação (s)

# Grandezas derivadas
w_syn = 2 * np.pi * f  # Velocidade síncrona elétrica (rad/s)
w_syn_mec = w_syn / (p / 2)  # Velocidade síncrona mecânica (rad/s)
Lm = Xm / w_syn
Ls = Xls / w_syn + Lm
Lr = Xlr / w_syn + Lm
V_fase_pico = np.sqrt(2) * V_LL / np.sqrt(3)

# Tensões em dq (fase a com pico em t = 0, eixo d alinhado com a fase a)
vsd = np.sqrt(3 / 2) * V_fase_pico
vsq = 0.0

# Matriz que relaciona fluxos e correntes: λ = M·i  ->  i = M⁻¹·λ
M = np.array([[Ls, 0, Lm, 0],
              [0, Ls, 0, Lm],
              [Lm, 0, Lr, 0],
              [0, Lm, 0, Lr]])
M_inv = np.linalg.inv(M)


def rpm_de(w_mec):
    return w_mec * 60 / (2 * np.pi)


# =============================================================================
# 1) Regime permanente nominal (Mohan, Exemplo 3-2, Eq. 3-70)
# =============================================================================
def regime_permanente(s):
    """Correntes dq e torque em regime para um escorregamento s."""
    ws = s * w_syn
    A = np.array([[Rs, -w_syn * Ls, 0, -w_syn * Lm],
                  [w_syn * Ls, Rs, w_syn * Lm, 0],
                  [0, -ws * Lm, Rr, -ws * Lr],
                  [ws * Lm, 0, ws * Lr, Rr]])
    i_dq = np.linalg.solve(A, np.array([vsd, vsq, 0, 0]))
    lam = M @ i_dq
    Te = (p / 2) * (lam[0] * i_dq[1] - lam[1] * i_dq[0])
    return i_dq, lam, Te


i_nom, lam_nom, Te_nom = regime_permanente(s_nominal)
w_nom_mec = (1 - s_nominal) * w_syn_mec
TL_carga = fator_carga * Te_nom


# =============================================================================
# 2) Curva torque x velocidade em regime (circuito equivalente)
# =============================================================================
s_curva = np.linspace(1, 1e-4, 800)
Te_curva = np.array([regime_permanente(sk)[2] for sk in s_curva])
n_curva = rpm_de((1 - s_curva) * w_syn_mec)
Te_partida = Te_curva[0]
idx_max = np.argmax(Te_curva)
Te_max, s_Tmax = Te_curva[idx_max], s_curva[idx_max]


# =============================================================================
# 3) Simulação dinâmica: partida direta a vazio + degrau de carga
# =============================================================================
def torque_carga(t):
    return TL_carga if t >= t_carga else 0.0


def motor_inducao(t, x):
    lsd, lsq, lrd, lrq, w_mec = x
    isd, isq, ird, irq = M_inv @ x[:4]
    w_m = (p / 2) * w_mec  # velocidade elétrica do rotor
    w_dA = w_syn - w_m  # velocidade de escorregamento
    Te = (p / 2) * (lsd * isq - lsq * isd)
    return [
        vsd - Rs * isd + w_syn * lsq,
        vsq - Rs * isq - w_syn * lsd,
        -Rr * ird + w_dA * lrq,
        -Rr * irq - w_dA * lrd,
        (Te - B * w_mec - torque_carga(t)) / J,
    ]


t = np.linspace(0, t_final, 20000)
sol = solve_ivp(motor_inducao, (0, t_final), np.zeros(5), t_eval=t,
                method="LSODA", max_step=1e-4, rtol=1e-7, atol=1e-9)
lam_t = sol.y[:4]
w_mec_t = sol.y[4]
i_t = M_inv @ lam_t
Te_t = (p / 2) * (lam_t[0] * i_t[1] - lam_t[1] * i_t[0])
TL_t = np.array([torque_carga(tk) for tk in t])

# Correntes de fase (transformação inversa de Park, invariante em potência)
theta = w_syn * t
ia = np.sqrt(2 / 3) * (i_t[0] * np.cos(theta) - i_t[1] * np.sin(theta))
ib = np.sqrt(2 / 3) * (i_t[0] * np.cos(theta - 2 * np.pi / 3)
                       - i_t[1] * np.sin(theta - 2 * np.pi / 3))
ic = -ia - ib
I_rms_t = np.sqrt((i_t[0]**2 + i_t[1]**2) / 3)  # valor eficaz instantâneo

# Validação dinâmica: regime final da simulação x regime calculado
w_final = w_mec_t[-1]
s_final = 1 - w_final / w_syn_mec
i_final_rp, _, Te_final_rp = regime_permanente(s_final)


# =============================================================================
# 4) Linearização no ponto nominal (sympy) -> polos do sistema
# =============================================================================
ls_sd, ls_sq, lr_d, lr_q, wm_s, TL_s, vsd_s = sp.symbols(
    "lambda_sd lambda_sq lambda_rd lambda_rq omega_mec T_L v_sd")
x_s = sp.Matrix([ls_sd, ls_sq, lr_d, lr_q, wm_s])
i_s = sp.Matrix(M_inv) * x_s[:4, 0]
w_dA_s = w_syn - (p / 2) * wm_s
Te_s = (p / 2) * (ls_sd * i_s[1] - ls_sq * i_s[0])
f_s = sp.Matrix([
    vsd_s - Rs * i_s[0] + w_syn * ls_sq,
    vsq - Rs * i_s[1] - w_syn * ls_sd,
    -Rr * i_s[2] + w_dA_s * lr_q,
    -Rr * i_s[3] - w_dA_s * lr_d,
    (Te_s - B * wm_s - TL_s) / J,
])
ponto = dict(zip(x_s, [*lam_nom, w_nom_mec]))
ponto.update({TL_s: Te_nom, vsd_s: vsd})
A_lin = np.array(f_s.jacobian(x_s).subs(ponto), dtype=float)
B_lin = np.array(f_s.jacobian(sp.Matrix([vsd_s, TL_s])).subs(ponto), dtype=float)
polos = np.linalg.eigvals(A_lin)

# Ganho estático de velocidade: Δω/ΔTL em regime = -C·A⁻¹·B
C_w = np.array([[0, 0, 0, 0, 1]])
ganho_wTL = float(-(C_w @ np.linalg.solve(A_lin, B_lin[:, [1]]))[0, 0])


# =============================================================================
# 5) Resultados
# =============================================================================
print("=" * 66)
print(" MOTOR DE INDUÇÃO TRIFÁSICO - RESULTADOS")
print("=" * 66)
print(f"Ls = {Ls * 1e3:.2f} mH | Lr = {Lr * 1e3:.2f} mH | Lm = {Lm * 1e3:.2f} mH")
print(f"vsd = {vsd:.2f} V | vsq = {vsq:.2f} V (referencial síncrono)")
print(f"Velocidade síncrona: {rpm_de(w_syn_mec):.0f} rpm")
print(f"\nRegime nominal (s = {s_nominal * 100:.2f} %, n = {rpm_de(w_nom_mec):.0f} rpm):")
print(f"  isd = {i_nom[0]:.3f} A | isq = {i_nom[1]:.3f} A | "
      f"ird = {i_nom[2]:.3f} A | irq = {i_nom[3]:.3f} A")
print(f"  λsd = {lam_nom[0]:.4f} | λsq = {lam_nom[1]:.4f} | "
      f"λrd = {lam_nom[2]:.4f} | λrq = {lam_nom[3]:.4f} Wb-esp")
print(f"  Te nominal = {Te_nom:.3f} N.m   (Mohan, Ex. 3-1: 12,644 N.m)")
print(f"  Corrente de linha = {np.hypot(*i_nom[:2]) / np.sqrt(3):.3f} A rms")
print(f"\nCurva de regime: Te partida = {Te_partida:.2f} N.m | "
      f"Te máx = {Te_max:.2f} N.m em s = {s_Tmax * 100:.1f} % "
      f"({rpm_de((1 - s_Tmax) * w_syn_mec):.0f} rpm)")
print(f"\nSimulação dinâmica (partida direta):")
print(f"  Pico de corrente de fase: {np.max(np.abs(ia)):.2f} A "
      f"(nominal: {np.sqrt(2) * np.hypot(*i_nom[:2]) / np.sqrt(3):.2f} A pico)")
print(f"  Pico de torque: {Te_t.max():.2f} N.m")
print(f"  Velocidade a vazio: {rpm_de(w_mec_t[np.searchsorted(t, t_carga) - 1]):.1f} rpm")
print(f"  Velocidade final com carga: {rpm_de(w_final):.1f} rpm "
      f"(s = {s_final * 100:.2f} %)")
print(f"  Validação: Te final simulado = {Te_t[-1]:.3f} N.m | "
      f"regime no mesmo s = {Te_final_rp:.3f} N.m")
print(f"\nModelo linearizado no ponto nominal (5 estados):")
print(f"  Polos: {np.round(np.sort_complex(polos), 2)}")
print(f"  Ganho estático Δωmec/ΔTL = {ganho_wTL:.4f} rad/s por N.m "
      f"({rpm_de(ganho_wTL):.3f} rpm por N.m)")
print("=" * 66)

fig, ax = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle(f"Motor de indução trifásico — partida direta ({V_LL} V, {f} Hz) "
             f"e carga de {TL_carga:.2f} N.m em t = {t_carga} s")

ax[0, 0].plot(t, ia, lw=0.8, label="ia")
ax[0, 0].plot(t, ib, lw=0.8, label="ib")
ax[0, 0].plot(t, ic, lw=0.8, label="ic")
ax[0, 0].set_title("Correntes de fase do estator")
ax[0, 0].set_ylabel("i (A)")
ax[0, 0].legend(loc="upper right")

ax[0, 1].plot(t, rpm_de(w_mec_t), "tab:green")
ax[0, 1].axhline(rpm_de(w_syn_mec), color="k", ls=":", label="Síncrona")
ax[0, 1].set_title("Velocidade do rotor")
ax[0, 1].set_ylabel("n (rpm)")
ax[0, 1].legend()

ax[1, 0].plot(t, Te_t, "tab:red", lw=0.8, label="Te (eletromagnético)")
ax[1, 0].plot(t, TL_t, "k--", label="TL (carga)")
ax[1, 0].set_title("Torque")
ax[1, 0].set_ylabel("T (N.m)")
ax[1, 0].legend()

ax[1, 1].plot(n_curva, Te_curva, "tab:purple", label="Regime (circuito equivalente)")
ax[1, 1].plot(rpm_de(w_mec_t), Te_t, "tab:red", lw=0.6, alpha=0.7,
              label="Trajetória dinâmica")
ax[1, 1].plot(rpm_de(w_nom_mec), Te_nom, "ko", label="Ponto nominal")
ax[1, 1].set_title("Característica torque × velocidade")
ax[1, 1].set_xlabel("n (rpm)")
ax[1, 1].set_ylabel("Te (N.m)")
ax[1, 1].legend(fontsize=8)

for axk in ax.flat[:3]:
    axk.set_xlabel("t (s)")
for axk in ax.flat:
    axk.grid(True, alpha=0.3)

fig2, ax2 = plt.subplots(1, 2, figsize=(12, 4.5))
ax2[0].plot(t, i_t[0], label="isd")
ax2[0].plot(t, i_t[1], label="isq")
ax2[0].plot(t, I_rms_t, "k--", label="I rms")
ax2[0].set_title("Correntes do estator em dq (referencial síncrono)")
ax2[0].set_xlabel("t (s)")
ax2[0].set_ylabel("i (A)")
ax2[0].legend()
ax2[0].grid(True, alpha=0.3)

ax2[1].plot(polos.real, polos.imag, "rx", markersize=10, mew=2)
ax2[1].axhline(0, color="k", lw=0.8)
ax2[1].axvline(0, color="k", lw=0.8)
ax2[1].set_title("Polos do modelo linearizado (ponto nominal)")
ax2[1].set_xlabel("Re")
ax2[1].set_ylabel("Im")
ax2[1].grid(True, alpha=0.3)

fig.tight_layout()
fig2.tight_layout()
fig.savefig("motorInducao_simulacao.png", dpi=150)
fig2.savefig("motorInducao_dq_polos.png", dpi=150)
plt.show()
