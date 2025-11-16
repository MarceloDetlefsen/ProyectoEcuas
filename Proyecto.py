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

import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, Tuple, List
import pandas as pd


# ============================================================================
# MÉTODOS NUMÉRICOS
# ============================================================================

def metodo_heun(f: Callable, t0: float, y0: np.ndarray, tf: float, h: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Implementa el método de Heun (Euler Mejorado) para resolver EDOs.
    
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