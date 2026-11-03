"""
Modelagem e simulação de um motor CC de ímã permanente (PMDC).

No PMDC o fluxo do campo é produzido por ímãs, logo é CONSTANTE. Com isso:

    Força contraeletromotriz:  e  = Kb·ω
    Torque eletromagnético:    Te = Kt·i      (no SI, Kt = Kb = K)

Equações do modelo (Dawson et al., seção 1.7.6 / Filizadeh, seção 3.4):

    Elétrica:   La·di/dt = V - Ra·i - K·ω
    Mecânica:    J·dω/dt = K·i - B·ω - TL

O modelo é LINEAR, então dá para escrever direto em espaço de estados e
em função de transferência:

    ω(s)/V(s)  =            K
                 ---------------------------
                 (La·s + Ra)(J·s + B) + K²

Parâmetros padrão: Exemplo 3.2 do Filizadeh (PM dc machine), que serve
para conferir os resultados com o livro.
"""

import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
from scipy import signal
from scipy.integrate import solve_ivp

# Parâmetros do motor PMDC (Filizadeh, Exemplo 3.2)
Ra = 0.25  # Resistência da armadura (ohm)
La = 0.02  # Indutância da armadura (H)
K = 0.8  # Constante de f.c.e.m. = constante de torque (V.s/rad = N.m/A)
J = 2.0  # Momento de inércia (kg.m²)
B = 0.05  # Coeficiente de atrito viscoso (N.m.s/rad)
V = 48  # Tensão de alimentação (V)

# Parâmetros da simulação
TL_carga = 40  # Torque de carga aplicado (N.m)
t_carga = 6.0  # Instante de aplicação da carga (s)
t_final = 12.0  # Tempo total de simulação (s)


# =============================================================================
# 1) Modelo simbólico: espaço de estados e funções de transferência (sympy)
# =============================================================================
s = sp.symbols("s")
Ra_s, La_s, K_s, J_s, B_s = sp.symbols("R_a L_a K J B", positive=True)

# x = [i, ω],  u = [V, TL]
A_sim = sp.Matrix([[-Ra_s / La_s, -K_s / La_s],
                   [K_s / J_s, -B_s / J_s]])
B_sim = sp.Matrix([[1 / La_s, 0],
                   [0, -1 / J_s]])

# G(s) = (sI - A)^-1 B  ->  linha 0: corrente, linha 1: velocidade
G_sim = sp.simplify((s * sp.eye(2) - A_sim).inv() * B_sim)


def organizar(expr):
    """Escreve numerador e denominador agrupados em potências de s."""
    num, den = sp.fraction(sp.factor(expr))
    return sp.collect(sp.expand(num), s) / sp.collect(sp.expand(den), s)


G_wV_sim = organizar(G_sim[1, 0])  # ω(s)/V(s)
G_wTL_sim = organizar(G_sim[1, 1])  # ω(s)/TL(s)
G_iV_sim = organizar(G_sim[0, 0])  # I(s)/V(s)

valores = {Ra_s: Ra, La_s: La, K_s: K, J_s: J, B_s: B}
A_num = np.array(A_sim.subs(valores), dtype=float)
B_num = np.array(B_sim.subs(valores), dtype=float)


def para_tf(expr):
    """Converte uma expressão sympy em s para scipy.signal.TransferFunction."""
    num, den = sp.fraction(sp.together(expr.subs(valores)))
    num = [float(c) for c in sp.Poly(num, s).all_coeffs()]
    den = [float(c) for c in sp.Poly(den, s).all_coeffs()]
    return signal.TransferFunction(num, den)


G_wV = para_tf(G_wV_sim)
G_wTL = para_tf(G_wTL_sim)
G_iV = para_tf(G_iV_sim)


# =============================================================================
# 2) Análise: polos, constantes de tempo, regime permanente
# =============================================================================
polos = np.linalg.eigvals(A_num)
den = G_wV.den / G_wV.den[0]  # s² + 2ζωn·s + ωn²
wn = np.sqrt(den[2])
zeta = den[1] / (2 * wn)

tau_e = La / Ra  # constante de tempo elétrica
tau_m = J / B  # constante de tempo mecânica (sem acoplamento elétrico)
tau_em = J * Ra / (K**2 + Ra * B)  # constante de tempo eletromecânica


def regime_permanente(V, TL):
    """Resolve A·x + B·u = 0."""
    x = np.linalg.solve(A_num, -B_num @ np.array([V, TL]))
    return x[0], x[1]


i0_vazio, w0_vazio = regime_permanente(V, 0.0)
i0_carga, w0_carga = regime_permanente(V, TL_carga)
i_partida = V / Ra  # corrente máxima teórica (rotor parado, sem indutância)


# =============================================================================
# 3) Simulação no tempo (partida + degrau de carga)
# =============================================================================
def torque_carga(t):
    return TL_carga if t >= t_carga else 0.0


def motor_pm(t, x):
    u = np.array([V, torque_carga(t)])
    return A_num @ x + B_num @ u


t = np.linspace(0, t_final, 6000)
sol = solve_ivp(motor_pm, (0, t_final), [0.0, 0.0], t_eval=t, max_step=1e-3)
i = sol.y[0]
w = sol.y[1]
Te = K * i
TL_t = np.array([torque_carga(tk) for tk in t])
rpm = w * 60 / (2 * np.pi)

# Validação: mesma resposta pela superposição das funções de transferência
_, w_tfV, _ = signal.lsim(G_wV, V * np.ones_like(t), t)
_, w_tfTL, _ = signal.lsim(G_wTL, TL_t, t)
w_tf = w_tfV + w_tfTL
erro_max = np.max(np.abs(w - w_tf))


# =============================================================================
# 4) Métricas da resposta ao degrau de tensão (sem carga)
# =============================================================================
t_deg = np.linspace(0, t_carga, 6000)
_, w_deg = signal.step(G_wV, T=t_deg)
w_deg *= V
w_inf = w0_vazio
t10 = t_deg[np.argmax(w_deg >= 0.1 * w_inf)]
t90 = t_deg[np.argmax(w_deg >= 0.9 * w_inf)]
fora_faixa = np.where(np.abs(w_deg - w_inf) > 0.02 * w_inf)[0]
t_acomod = t_deg[fora_faixa[-1]] if len(fora_faixa) else 0.0
sobressinal = max(0.0, (w_deg.max() - w_inf) / w_inf * 100)


# =============================================================================
# 5) Resultados
# =============================================================================
def rpm_de(w):
    return w * 60 / (2 * np.pi)


print("=" * 64)
print(" MOTOR CC DE ÍMÃ PERMANENTE (PMDC) - RESULTADOS")
print("=" * 64)
print("Modelo em espaço de estados: dx/dt = A·x + B·u")
print("  x = [i, ω],  u = [V, TL]")
print("A ="); sp.pprint(A_sim)
print("B ="); sp.pprint(B_sim)
print(f"A numérica =\n{np.round(A_num, 4)}")
print("\nFunções de transferência simbólicas:")
print("ω(s)/V(s)  ="); sp.pprint(G_wV_sim)
print("ω(s)/TL(s) ="); sp.pprint(G_wTL_sim)
print("I(s)/V(s)  ="); sp.pprint(G_iV_sim)
print("\nNuméricas:")
print(f"ω(s)/V(s)  = {G_wV.num[0]:.4g} / (s² + {G_wV.den[1]:.4g}·s + {G_wV.den[2]:.4g})")
print(f"ω(s)/TL(s): num = {np.round(G_wTL.num, 5)},  den = {np.round(G_wTL.den, 5)}")
print(f"I(s)/V(s):  num = {np.round(G_iV.num, 5)},  den = {np.round(G_iV.den, 5)}")
print(f"\nPolos: {np.round(polos, 4)}")
print(f"ωn = {wn:.4f} rad/s | ζ = {zeta:.4f} "
      f"({'superamortecido' if zeta > 1 else 'subamortecido' if zeta < 1 else 'crítico'})")
print(f"τ elétrica = La/Ra = {tau_e:.4f} s")
print(f"τ mecânica = J/B = {tau_m:.4f} s")
print(f"τ eletromecânica = J·Ra/(K² + Ra·B) = {tau_em:.4f} s")
print(f"\nCorrente de partida teórica V/Ra: {i_partida:.1f} A "
      f"(simulada, com La: {i.max():.1f} A)")
print(f"Regime a vazio:  i0 = {i0_vazio:.3f} A | ω0 = {w0_vazio:.3f} rad/s "
      f"({rpm_de(w0_vazio):.1f} rpm)")
print(f"Regime com carga (TL = {TL_carga} N.m): i0 = {i0_carga:.3f} A | "
      f"ω0 = {w0_carga:.3f} rad/s ({rpm_de(w0_carga):.1f} rpm)")
print(f"\nResposta ao degrau de {V} V (sem carga):")
print(f"  Tempo de subida (10-90%): {t90 - t10:.4f} s")
print(f"  Tempo de acomodação (2%): {t_acomod:.4f} s")
print(f"  Sobressinal: {sobressinal:.2f} %")
print(f"\nValidação espaço de estados x função de transferência: "
      f"erro máx. = {erro_max:.2e} rad/s")
print("=" * 64)

fig, ax = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle(f"Motor PMDC — partida com V = {V} V e carga "
             f"TL = {TL_carga} N.m em t = {t_carga} s")

ax[0, 0].plot(t, i, "tab:blue")
ax[0, 0].set_title("Corrente de armadura")
ax[0, 0].set_ylabel("i (A)")

ax[0, 1].plot(t, rpm, "tab:green", label="Espaço de estados")
ax[0, 1].plot(t, w_tf * 60 / (2 * np.pi), "k--", label="Função de transferência")
ax[0, 1].set_title("Velocidade")
ax[0, 1].set_ylabel("n (rpm)")
ax[0, 1].legend()

ax[1, 0].plot(t, Te, "tab:red", label="Te (eletromagnético)")
ax[1, 0].plot(t, TL_t, "k--", label="TL (carga)")
ax[1, 0].set_title("Torque")
ax[1, 0].set_ylabel("T (N.m)")
ax[1, 0].legend()

# Característica torque x velocidade em regime: ω = (V - Ra·Te/K)/K
Te_rp = np.linspace(0, K * V / Ra, 200)
w_rp = (V - Ra * Te_rp / K) / K
ax[1, 1].plot(Te_rp, rpm_de(w_rp), "tab:purple")
ax[1, 1].plot([K * i0_vazio, K * i0_carga], [rpm_de(w0_vazio), rpm_de(w0_carga)],
              "ko", label="Pontos de operação")
ax[1, 1].set_title("Característica torque × velocidade (regime)")
ax[1, 1].set_xlabel("Te (N.m)")
ax[1, 1].set_ylabel("n (rpm)")
ax[1, 1].legend()

for axk in ax.flat[:3]:
    axk.set_xlabel("t (s)")
for axk in ax.flat:
    axk.grid(True, alpha=0.3)

# Mapa de polos
fig2, ax2 = plt.subplots(figsize=(6, 4.5))
ax2.plot(polos.real, polos.imag, "rx", markersize=12, mew=2)
for p in polos:
    ax2.annotate(f"{p.real:.3f}", (p.real, p.imag), textcoords="offset points",
                 xytext=(0, 10), ha="center")
ax2.axhline(0, color="k", lw=0.8)
ax2.axvline(0, color="k", lw=0.8)
ax2.set_title("Polos de ω(s)/V(s)")
ax2.set_xlabel("Re")
ax2.set_ylabel("Im")
ax2.grid(True, alpha=0.3)

fig.tight_layout()
fig2.tight_layout()
fig.savefig("motorCCPM_simulacao.png", dpi=150)
fig2.savefig("motorCCPM_polos.png", dpi=150)
plt.show()
