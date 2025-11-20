# Universidad del Valle de Guatemala
# Facultad de Ingeniería
# Departamento de Matemáticas
# MM2029 Ecuaciones Diferenciales I
# Proyecto Final (versión con estudios de convergencia)

import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, Tuple, List
import pandas as pd


# ============================================================================
# MÉTODOS NUMÉRICOS
# ============================================================================

def metodo_heun(f: Callable, t0: float, y0: np.ndarray, tf: float, h: float) -> Tuple[np.ndarray, np.ndarray]:
    n_steps = int((tf - t0) / h) + 1
    t = np.linspace(t0, tf, n_steps)
    y0 = np.atleast_1d(y0)
    y = np.zeros((n_steps, len(y0)))
    y[0] = y0
    for i in range(n_steps - 1):
        k1 = f(t[i], y[i])
        y_pred = y[i] + h * k1
        k2 = f(t[i+1], y_pred)
        y[i+1] = y[i] + (h / 2) * (k1 + k2)
    return t, y


def metodo_rk4(f: Callable, t0: float, y0: np.ndarray, tf: float, h: float) -> Tuple[np.ndarray, np.ndarray]:
    n_steps = int((tf - t0) / h) + 1
    t = np.linspace(t0, tf, n_steps)
    y0 = np.atleast_1d(y0)
    y = np.zeros((n_steps, len(y0)))
    y[0] = y0
    for i in range(n_steps - 1):
        k1 = f(t[i], y[i])
        k2 = f(t[i] + h/2, y[i] + (h/2) * k1)
        k3 = f(t[i] + h/2, y[i] + (h/2) * k2)
        k4 = f(t[i] + h, y[i] + h * k3)
        y[i+1] = y[i] + (h / 6) * (k1 + 2*k2 + 2*k3 + k4)
    return t, y


# ============================================================================
# DEFINICIÓN DE ECUACIONES DIFERENCIALES
# ============================================================================

# 1. ED de primer orden: y'(t) + 2y(t) = e^(-t), y(0) = 1
def ed_primer_orden(t: float, y: np.ndarray) -> np.ndarray:
    return np.array([-2*y[0] + np.exp(-t)])

def solucion_analitica_1er_orden(t: np.ndarray) -> np.ndarray:
    return np.exp(-t)


# 2. ED de segundo orden: y'' + 3y' + 2y = 0, y(0)=1, y'(0)=0
def ed_segundo_orden(t: float, y: np.ndarray) -> np.ndarray:
    return np.array([y[1], -3*y[1] - 2*y[0]])

def solucion_analitica_2do_orden(t: np.ndarray) -> np.ndarray:
    return 2*np.exp(-t) - np.exp(-2*t)


# 3. Sistema 2x2 lineal
def sistema_lineal(t: float, y: np.ndarray) -> np.ndarray:
    x, y_val = y
    return np.array([4*x + y_val, -2*x + 3*y_val])

def solucion_analitica_sistema_lineal_arr(t: np.ndarray) -> np.ndarray:
    # devuelve array de forma (len(t), 2)
    omega = np.sqrt(7) / 2.0
    expfac = np.exp(7.0 * t / 2.0)
    x = expfac * (np.cos(omega * t) + (1.0 / np.sqrt(7.0)) * np.sin(omega * t))
    y = - (4.0 / np.sqrt(7.0)) * expfac * np.sin(omega * t)
    return np.vstack((x, y)).T


# 4. Sistema 2x2 no lineal (Lotka-Volterra modificado)
def sistema_no_lineal(t: float, y: np.ndarray) -> np.ndarray:
    y1, y2 = y
    return np.array([
        y1 - y1*y2 + np.sin(np.pi * t),
        y1*y2 - y1
    ])

# No hay solución analítica cerrada implementada para el sistema no lineal


# ============================================================================
# FUNCIONES DE ANÁLISIS Y CONVERGENCIA
# ============================================================================

def calcular_error_vectorial(y_num: np.ndarray, y_anal: np.ndarray) -> float:
    """Calcula RMSE considerando todas las componentes y todos los tiempos."""
    # y_num y y_anal deben tener la misma forma (n_steps, n_comp)
    diff = y_num - y_anal
    return np.sqrt(np.mean(diff**2))


def estudio_convergencia_general(f: Callable, y0: np.ndarray, t0: float, tf: float,
                                 y_analitica_func: Callable, pasos: List[float]) -> Tuple[pd.DataFrame, dict]:
    """
    Realiza estudio de convergencia para Heun y RK4.
    Si y_analitica_func devuelve un vector por tiempo (len(t), n_comp), calculamos RMSE global.
    Para casos escalares, la función puede devolver un array 1D.
    Retorna un DataFrame con h y errores, y un diccionario con pendientes estimadas.
    """
    resultados = []
    heun_errors = []
    rk4_errors = []
    hs = []

    for h in pasos:
        # Heun
        t_h, y_h = metodo_heun(f, t0, y0, tf, h)
        y_anal_h = y_analitica_func(t_h)
        # Asegurar dimensiones
        if y_h.ndim == 1:
            y_h = y_h.reshape(-1, 1)
        if y_anal_h.ndim == 1:
            y_anal_h = y_anal_h.reshape(-1, 1)
        error_h = calcular_error_vectorial(y_h, y_anal_h)

        # RK4
        t_r, y_r = metodo_rk4(f, t0, y0, tf, h)
        y_anal_r = y_analitica_func(t_r)
        if y_r.ndim == 1:
            y_r = y_r.reshape(-1, 1)
        if y_anal_r.ndim == 1:
            y_anal_r = y_anal_r.reshape(-1, 1)
        error_r = calcular_error_vectorial(y_r, y_anal_r)

        resultados.append({'h': h, 'Heun_RMSE': error_h, 'RK4_RMSE': error_r})
        heun_errors.append(error_h)
        rk4_errors.append(error_r)
        hs.append(h)

    df = pd.DataFrame(resultados)

    # Ajuste lineal en log-log para estimar pendiente (orden de convergencia)
    logh = np.log(hs)
    log_heun = np.log(heun_errors)
    log_rk4 = np.log(rk4_errors)

    pendiente_heun, interc_heun = np.polyfit(logh, log_heun, 1)
    pendiente_rk4, interc_rk4 = np.polyfit(logh, log_rk4, 1)

    pendientes = {
        'Heun_pendiente': pendiente_heun,
        'RK4_pendiente': pendiente_rk4
    }

    return df, pendientes


def graficar_convergencia(df: pd.DataFrame, titulo: str, guardar: bool = True):
    hs = df['h'].values
    e_heun = df['Heun_RMSE'].values
    e_rk4 = df['RK4_RMSE'].values

    plt.figure(figsize=(7, 5))
    plt.loglog(hs, e_heun, 'o-', label='Heun')
    plt.loglog(hs, e_rk4, 's--', label='RK4')

    # calcular pendientes locales (ajuste global también mostrado)
    logh = np.log(hs)
    pente_heun = np.polyfit(logh, np.log(e_heun), 1)[0]
    pente_rk4 = np.polyfit(logh, np.log(e_rk4), 1)[0]

    plt.title(f'Estudio de convergencia - {titulo}\nPendientes estimadas: Heun={pente_heun:.2f}, RK4={pente_rk4:.2f}')
    plt.xlabel('h (log)')
    plt.ylabel('RMSE (log)')
    plt.grid(True, which='both', ls='--', alpha=0.4)
    plt.legend()
    if guardar:
        plt.savefig(f'Convergencia_{titulo.replace(" ", "_")}.png', dpi=300, bbox_inches='tight')
    plt.show()


# ============================================================================
# FUNCIONES DE VISUALIZACIÓN ADICIONALES (mantengo las anteriores para comparar)
# ============================================================================

def graficar_comparacion(t: np.ndarray, y_num: np.ndarray, y_analitica: np.ndarray,
                         titulo: str, metodo: str):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(t, y_analitica, 'b-', label='Analítica', linewidth=2)
    plt.plot(t, y_num, 'r--', label=f'{metodo}', linewidth=2)
    plt.xlabel('t')
    plt.ylabel('y(t)')
    plt.title(f'{titulo} - Comparación')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.subplot(1, 2, 2)
    error = np.abs(y_num - y_analitica)
    plt.plot(t, error, 'g-', linewidth=2)
    plt.xlabel('t')
    plt.ylabel('Error absoluto')
    plt.title('Error absoluto')
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig(f'{titulo.replace(" ", "_")}_{metodo}.png', dpi=300, bbox_inches='tight')
    plt.show()


def graficar_sistema(t: np.ndarray, y: np.ndarray, titulo: str, metodo: str):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(t, y[:, 0], 'b-', label='y₁(t)', linewidth=2)
    plt.plot(t, y[:, 1], 'r-', label='y₂(t)', linewidth=2)
    plt.xlabel('t')
    plt.ylabel('Valores')
    plt.title(f'{titulo} - {metodo}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.subplot(1, 2, 2)
    plt.plot(y[:, 0], y[:, 1], 'b-', linewidth=2)
    plt.plot(y[0, 0], y[0, 1], 'go', markersize=10, label='Inicio')
    plt.plot(y[-1, 0], y[-1, 1], 'ro', markersize=10, label='Final')
    plt.xlabel('y₁')
    plt.ylabel('y₂')
    plt.title('Plano fase')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{titulo.replace(" ", "_")}_{metodo}.png', dpi=300, bbox_inches='tight')
    plt.show()


# ============================================================================
# PROGRAMA PRINCIPAL
# ============================================================================

def main():
    print('='*70)
    print('PROYECTO FINAL - ECUACIONES DIFERENCIALES 1 (Convergencia)')
    print('='*70)

    # parámetros generales
    h_default = 0.1
    t0, tf = 0, 5

    # lista de tamaños de paso para estudio (ordenados de mayor a menor)
    pasos = [0.5, 0.2, 0.1, 0.05, 0.01]

    # -----------------------------
    # 1) Primer orden
    # -----------------------------
    print('\n1) EDO de primer orden')
    y0_1 = np.array([1.0])
    t_heun1, y_heun1 = metodo_heun(ed_primer_orden, t0, y0_1, tf, h_default)
    y_anal1 = solucion_analitica_1er_orden(t_heun1)
    print('Heun RMSE ejemplo:', np.sqrt(np.mean((y_heun1[:, 0] - y_anal1)**2)))

    # Estudio de convergencia
    df1, pend1 = estudio_convergencia_general(ed_primer_orden, y0_1, t0, tf, lambda tt: solucion_analitica_1er_orden(tt).reshape(-1,1), pasos)
    print('\nTabla de convergencia (ED primer orden):')
    print(df1.to_string(index=False))
    print(f"Pendientes estimadas: Heun={pend1['Heun_pendiente']:.3f}, RK4={pend1['RK4_pendiente']:.3f}")
    graficar_convergencia(df1, 'ED_Primer_Orden')

    # -----------------------------
    # 2) Segundo orden
    # -----------------------------
    print('\n2) EDO de segundo orden')
    y0_2 = np.array([1.0, 0.0])
    t_heun2, y_heun2 = metodo_heun(ed_segundo_orden, t0, y0_2, tf, h_default)
    y_anal2 = solucion_analitica_2do_orden(t_heun2)
    print('Heun RMSE ejemplo (componente y):', np.sqrt(np.mean((y_heun2[:, 0] - y_anal2)**2)))

    df2, pend2 = estudio_convergencia_general(ed_segundo_orden, y0_2, t0, tf, lambda tt: solucion_analitica_2do_orden(tt).reshape(-1,1), pasos)
    print('\nTabla de convergencia (ED segundo orden):')
    print(df2.to_string(index=False))
    print(f"Pendientes estimadas: Heun={pend2['Heun_pendiente']:.3f}, RK4={pend2['RK4_pendiente']:.3f}")
    graficar_convergencia(df2, 'ED_Segundo_Orden')

    # -----------------------------
    # 3) Sistema lineal 2x2
    # -----------------------------
    print('\n3) Sistema lineal 2x2')
    y0_3 = np.array([1.0, 0.0])
    t_heun3, y_heun3 = metodo_heun(sistema_lineal, t0, y0_3, tf, h_default)
    print(f'Heun - Valores finales: x({tf}) = {y_heun3[-1,0]:.6f}, y({tf}) = {y_heun3[-1,1]:.6f}')

    df3, pend3 = estudio_convergencia_general(sistema_lineal, y0_3, t0, tf, solucion_analitica_sistema_lineal_arr, pasos)
    print('\nTabla de convergencia (Sistema lineal):')
    print(df3.to_string(index=False))
    print(f"Pendientes estimadas: Heun={pend3['Heun_pendiente']:.3f}, RK4={pend3['RK4_pendiente']:.3f}")
    graficar_convergencia(df3, 'Sistema_Lineal')

    # -----------------------------
    # 4) Sistema no lineal 2x2
    # -----------------------------
    print('\n4) Sistema no lineal 2x2 (Lotka-Volterra modificado)')
    y0_4 = np.array([2.0, 1.0])
    tf_nl = 20
    t_heun4, y_heun4 = metodo_heun(sistema_no_lineal, t0, y0_4, tf_nl, h_default)
    print(f'Heun - Valores finales: y1({tf_nl}) = {y_heun4[-1,0]:.6f}, y2({tf_nl}) = {y_heun4[-1,1]:.6f}')

    # Para el sistema no lineal no disponemos de solución analítica cerrada, por lo que
    # para el estudio de convergencia usaremos una "solución de referencia" obtenida con
    # RK4 y paso muy pequeño (p.e. h_ref = 1e-4) — esto es una práctica común.
    print('\nConstruyendo solución de referencia (RK4 con paso muy pequeño) para el sistema no lineal...')
    h_ref = 1e-3
    t_ref, y_ref = metodo_rk4(sistema_no_lineal, t0, y0_4, tf_nl, h_ref)

    def y_ref_interpolada(t_query: np.ndarray) -> np.ndarray:
        # interpola cada componente del y_ref a los tiempos t_query
        yq = np.zeros((len(t_query), y_ref.shape[1]))
        for comp in range(y_ref.shape[1]):
            yq[:, comp] = np.interp(t_query, t_ref, y_ref[:, comp])
        return yq

    # Nuevo estudio de convergencia usando la solución de referencia
    pasos_nl = [0.5, 0.2, 0.1, 0.05, 0.02]
    resultados_nl = []
    for h in pasos_nl:
        t_h, y_h = metodo_heun(sistema_no_lineal, t0, y0_4, tf_nl, h)
        y_ref_h = y_ref_interpolada(t_h)
        err_h = calcular_error_vectorial(y_h, y_ref_h)

        t_r, y_r = metodo_rk4(sistema_no_lineal, t0, y0_4, tf_nl, h)
        y_ref_r = y_ref_interpolada(t_r)
        err_r = calcular_error_vectorial(y_r, y_ref_r)

        resultados_nl.append({'h': h, 'Heun_RMSE': err_h, 'RK4_RMSE': err_r})

    df4 = pd.DataFrame(resultados_nl)
    logh = np.log(df4['h'].values)
    pente_heun = np.polyfit(logh, np.log(df4['Heun_RMSE'].values), 1)[0]
    pente_rk4 = np.polyfit(logh, np.log(df4['RK4_RMSE'].values), 1)[0]

    print('\nTabla de convergencia (Sistema no lineal, referencia numérica):')
    print(df4.to_string(index=False))
    print(f"Pendientes estimadas (ref numérica): Heun={pente_heun:.3f}, RK4={pente_rk4:.3f}")
    graficar_convergencia(df4, 'Sistema_No_Lineal')

    print('\n' + '='*70)
    print('ANÁLISIS COMPLETADO')
    print('='*70)


if __name__ == '__main__':
    main()
