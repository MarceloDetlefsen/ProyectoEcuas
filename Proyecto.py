"""
Universidad del Valle de Guatemala
Facultad de Ingeniería
Departamento de Matemáticas
MM2029 Ecuaciones Diferenciales I
Proyecto Final

Autor: Marcelo Detlefsen, Marco Díaz, Julián Divas
Fecha: 15/11/2025

Descripción: Implementación de métodos numéricos: Heun (Euler Mejorado) y RK4 para resolver
             ecuaciones diferenciales ordinarias (EDOs) de primer y segundo orden, 
             así como sistemas de ecuaciones lineales y no lineales.
"""

import numpy as np # Biblioteca para manejo de arrays y operaciones numéricas
import matplotlib.pyplot as plt # Biblioteca para visualización de datos
from typing import Callable, Tuple, List # Tipos para anotaciones de funciones
import pandas as pd # Biblioteca para manejo de datos en tablas


# ============================================================================
# MÉTODOS NUMÉRICOS
# ============================================================================

def metodo_heun(f: Callable, t0: float, y0: np.ndarray, tf: float, h: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Implementa el método de Heun (Euler Mejorado) para resolver EDOs.
    Método predictor-corrector.
    
    Parámetros:
    -----------
    f : función que define dy/dt = f(t, y)
    t0 : tiempo inicial
    y0 : condición inicial (escalar o array)
    tf : tiempo final
    h : tamaño de paso
    
    Retorna:
    --------
    t : array de tiempos
    y : array de soluciones
    """
    n_steps = int((tf - t0) / h) + 1
    t = np.linspace(t0, tf, n_steps)
    
    # Asegurar que y0 sea un array
    y0 = np.atleast_1d(y0)
    y = np.zeros((n_steps, len(y0)))
    y[0] = y0
    
    for i in range(n_steps - 1):
        # Predictor (Euler)
        k1 = f(t[i], y[i])
        y_pred = y[i] + h * k1
        
        # Corrector (Heun)
        k2 = f(t[i+1], y_pred)
        y[i+1] = y[i] + (h / 2) * (k1 + k2)
    
    return t, y


def metodo_rk4(f: Callable, t0: float, y0: np.ndarray, tf: float, h: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Implementa el método de Runge-Kutta de 4º orden (RK4).
    
    Parámetros:
    -----------
    f : función que define dy/dt = f(t, y)
    t0 : tiempo inicial
    y0 : condición inicial (escalar o array)
    tf : tiempo final
    h : tamaño de paso
    
    Retorna:
    --------
    t : array de tiempos
    y : array de soluciones
    """
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
    """y' = -2y + e^(-t)"""
    return np.array([-2*y[0] + np.exp(-t)])

def solucion_analitica_1er_orden(t: np.ndarray) -> np.ndarray:
    """Solución analítica correcta: y(t) = e^{-t}"""
    return np.exp(-t)


# 2. ED de segundo orden: y''(t) + 3y'(t) + 2y(t) = 0, y(0) = 1, y'(0) = 0
# Se convierte en sistema: y1 = y, y2 = y'
def ed_segundo_orden(t: float, y: np.ndarray) -> np.ndarray:
    """
    y1' = y2
    y2' = -3*y2 - 2*y1
    """
    return np.array([y[1], -3*y[1] - 2*y[0]])

def solucion_analitica_2do_orden(t: np.ndarray) -> np.ndarray:
    """Solución analítica: y(t) = 2e^(-t) - e^(-2t)"""
    return 2*np.exp(-t) - np.exp(-2*t)


# 3. Sistema 2x2 lineal
def sistema_lineal(t: float, y: np.ndarray) -> np.ndarray:
    """
    x' = 4x + y
    y' = -2x + 3y
    """
    x, y_val = y
    return np.array([4*x + y_val, -2*x + 3*y_val])

def solucion_analitica_sistema_lineal(t: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Solución analítica real para el sistema con condiciones iniciales x(0)=1, y(0)=0.
    x(t) = e^{7t/2} ( cos( sqrt(7)/2 * t ) + (1/sqrt(7)) sin( sqrt(7)/2 * t ) )
    y(t) = - (4/sqrt(7)) e^{7t/2} sin( sqrt(7)/2 * t )
    """
    omega = np.sqrt(7) / 2.0
    expfac = np.exp(7.0 * t / 2.0)
    x = expfac * (np.cos(omega * t) + (1.0 / np.sqrt(7.0)) * np.sin(omega * t))
    y = - (4.0 / np.sqrt(7.0)) * expfac * np.sin(omega * t)
    return x, y


# 4. Sistema 2x2 no lineal (Lotka-Volterra modificado)
def sistema_no_lineal(t: float, y: np.ndarray) -> np.ndarray:
    """
    y1' = y1 - y1*y2 + sin(pi*t)
    y2' = y1*y2 - y1
    """
    y1, y2 = y
    return np.array([
        y1 - y1*y2 + np.sin(np.pi * t),
        y1*y2 - y1
    ])


# ============================================================================
# FUNCIONES DE ANÁLISIS
# ============================================================================

def calcular_error(y_num: np.ndarray, y_analitica: np.ndarray) -> dict:
    """Calcula diferentes métricas de error"""
    error_abs = np.abs(y_num - y_analitica)
    return {
        'max': np.max(error_abs),
        'medio': np.mean(error_abs),
        'rmse': np.sqrt(np.mean(error_abs**2))
    }


def estudio_convergencia(f: Callable, y0: np.ndarray, t0: float, tf: float, 
                        y_analitica_func: Callable, pasos: List[float]) -> pd.DataFrame:
    """Realiza estudio de convergencia variando el tamaño de paso"""
    resultados = []
    
    for h in pasos:
        # Heun
        t_heun, y_heun = metodo_heun(f, t0, y0, tf, h)
        y_anal_heun = y_analitica_func(t_heun)
        if y_heun.ndim > 1:
            y_heun_comp = y_heun[:, 0]
        else:
            y_heun_comp = y_heun
        error_heun = calcular_error(y_heun_comp, y_anal_heun)
        
        # RK4
        t_rk4, y_rk4 = metodo_rk4(f, t0, y0, tf, h)
        y_anal_rk4 = y_analitica_func(t_rk4)
        if y_rk4.ndim > 1:
            y_rk4_comp = y_rk4[:, 0]
        else:
            y_rk4_comp = y_rk4
        error_rk4 = calcular_error(y_rk4_comp, y_anal_rk4)
        
        resultados.append({
            'h': h,
            'Heun_RMSE': error_heun['rmse'],
            'RK4_RMSE': error_rk4['rmse'],
            'Heun_Max': error_heun['max'],
            'RK4_Max': error_rk4['max']
        })
    
    return pd.DataFrame(resultados)


# ============================================================================
# FUNCIONES DE VISUALIZACIÓN
# ============================================================================

def graficar_comparacion(t: np.ndarray, y_num: np.ndarray, y_analitica: np.ndarray, 
                         titulo: str, metodo: str):
    """Grafica comparación entre solución numérica y analítica"""
    plt.figure(figsize=(12, 5))
    
    # Gráfica de soluciones
    plt.subplot(1, 2, 1)
    plt.plot(t, y_analitica, 'b-', label='Analítica', linewidth=2)
    plt.plot(t, y_num, 'r--', label=f'{metodo}', linewidth=2)
    plt.xlabel('t')
    plt.ylabel('y(t)')
    plt.title(f'{titulo} - Comparación')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Gráfica de error
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
    """Grafica sistema de ecuaciones 2x2"""
    plt.figure(figsize=(12, 5))
    
    # Evolución temporal
    plt.subplot(1, 2, 1)
    plt.plot(t, y[:, 0], 'b-', label='y₁(t)', linewidth=2)
    plt.plot(t, y[:, 1], 'r-', label='y₂(t)', linewidth=2)
    plt.xlabel('t')
    plt.ylabel('Valores')
    plt.title(f'{titulo} - {metodo}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plano fase
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
    """Función principal que ejecuta todo el análisis"""
    
    print("="*70)
    print("PROYECTO FINAL - ECUACIONES DIFERENCIALES 1")
    print("Métodos: Heun (Euler Mejorado) y RK4")
    print("="*70)
    
    # Parámetros generales
    h = 0.1  # Tamaño de paso
    
    # ========================================================================
    # 1. ED DE PRIMER ORDEN
    # ========================================================================
    print("\n1. ECUACIÓN DIFERENCIAL DE PRIMER ORDEN")
    print("-" * 70)
    
    t0, tf = 0, 5
    y0_1 = np.array([1.0])
    
    # Heun
    t_heun1, y_heun1 = metodo_heun(ed_primer_orden, t0, y0_1, tf, h)
    y_anal1 = solucion_analitica_1er_orden(t_heun1)
    error_heun1 = calcular_error(y_heun1[:, 0], y_anal1)
    print(f"Heun - Error RMSE: {error_heun1['rmse']:.6e}")
    
    # RK4
    t_rk4_1, y_rk4_1 = metodo_rk4(ed_primer_orden, t0, y0_1, tf, h)
    error_rk4_1 = calcular_error(y_rk4_1[:, 0], y_anal1)
    print(f"RK4  - Error RMSE: {error_rk4_1['rmse']:.6e}")
    
    # Visualización
    graficar_comparacion(t_heun1, y_heun1[:, 0], y_anal1, 
                        "ED_Primer_Orden", "Heun")
    graficar_comparacion(t_rk4_1, y_rk4_1[:, 0], y_anal1, 
                        "ED_Primer_Orden", "RK4")
    
    # Estudio de convergencia
    print("\nEstudio de convergencia:")
    pasos = [0.5, 0.2, 0.1, 0.05, 0.01]
    conv1 = estudio_convergencia(ed_primer_orden, y0_1, t0, tf, 
                                 solucion_analitica_1er_orden, pasos)
    print(conv1.to_string(index=False))
    
    # ========================================================================
    # 2. ED DE SEGUNDO ORDEN
    # ========================================================================
    print("\n\n2. ECUACIÓN DIFERENCIAL DE SEGUNDO ORDEN")
    print("-" * 70)
    
    y0_2 = np.array([1.0, 0.0])  # [y(0), y'(0)]
    
    # Heun
    t_heun2, y_heun2 = metodo_heun(ed_segundo_orden, t0, y0_2, tf, h)
    y_anal2 = solucion_analitica_2do_orden(t_heun2)
    error_heun2 = calcular_error(y_heun2[:, 0], y_anal2)
    print(f"Heun - Error RMSE: {error_heun2['rmse']:.6e}")
    
    # RK4
    t_rk4_2, y_rk4_2 = metodo_rk4(ed_segundo_orden, t0, y0_2, tf, h)
    error_rk4_2 = calcular_error(y_rk4_2[:, 0], y_anal2)
    print(f"RK4  - Error RMSE: {error_rk4_2['rmse']:.6e}")
    
    # Visualización
    graficar_comparacion(t_heun2, y_heun2[:, 0], y_anal2, 
                        "ED_Segundo_Orden", "Heun")
    graficar_comparacion(t_rk4_2, y_rk4_2[:, 0], y_anal2, 
                        "ED_Segundo_Orden", "RK4")
    
    # ========================================================================
    # 3. SISTEMA 2X2 LINEAL
    # ========================================================================
    print("\n\n3. SISTEMA DE ECUACIONES 2X2 LINEAL")
    print("-" * 70)
    
    y0_3 = np.array([1.0, 0.0])
    
    # Heun
    t_heun3, y_heun3 = metodo_heun(sistema_lineal, t0, y0_3, tf, h)
    print(f"Heun - Valores finales: x({tf}) = {y_heun3[-1, 0]:.6f}, y({tf}) = {y_heun3[-1, 1]:.6f}")
    
    # RK4
    t_rk4_3, y_rk4_3 = metodo_rk4(sistema_lineal, t0, y0_3, tf, h)
    print(f"RK4  - Valores finales: x({tf}) = {y_rk4_3[-1, 0]:.6f}, y({tf}) = {y_rk4_3[-1, 1]:.6f}")
    
    # Visualización
    graficar_sistema(t_heun3, y_heun3, "Sistema_Lineal", "Heun")
    graficar_sistema(t_rk4_3, y_rk4_3, "Sistema_Lineal", "RK4")
    
    # ========================================================================
    # 4. SISTEMA 2X2 NO LINEAL
    # ========================================================================
    print("\n\n4. SISTEMA DE ECUACIONES 2X2 NO LINEAL (LOTKA-VOLTERRA)")
    print("-" * 70)
    
    tf_nl = 20  # Tiempo más largo para observar comportamiento
    y0_4 = np.array([2.0, 1.0])
    
    # Heun
    t_heun4, y_heun4 = metodo_heun(sistema_no_lineal, t0, y0_4, tf_nl, h)
    print(f"Heun - Valores finales: y1({tf_nl}) = {y_heun4[-1, 0]:.6f}, y2({tf_nl}) = {y_heun4[-1, 1]:.6f}")
    
    # RK4
    t_rk4_4, y_rk4_4 = metodo_rk4(sistema_no_lineal, t0, y0_4, tf_nl, h)
    print(f"RK4  - Valores finales: y1({tf_nl}) = {y_rk4_4[-1, 0]:.6f}, y2({tf_nl}) = {y_rk4_4[-1, 1]:.6f}")
    
    # Visualización
    graficar_sistema(t_heun4, y_heun4, "Sistema_No_Lineal", "Heun")
    graficar_sistema(t_rk4_4, y_rk4_4, "Sistema_No_Lineal", "RK4")
    
    print("\n" + "="*70)
    print("ANÁLISIS COMPLETADO")
    print("="*70)


if __name__ == "__main__":
    main()