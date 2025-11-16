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

# ============================================================================
# DEFINICIÓN DE ECUACIONES DIFERENCIALES
# ============================================================================

# 1. ED de primer orden: y'(t) + 2y(t) = e^(-t), y(0) = 1
def ed_primer_orden(t: float, y: np.ndarray) -> np.ndarray:
    """y' = -2y + e^(-t)"""
    return np.array([-2*y[0] + np.exp(-t)])

def solucion_analitica_1er_orden(t: np.ndarray) -> np.ndarray:
    """Solución analítica: y(t) = (1 + t)e^(-t)"""
    return (1 + t) * np.exp(-t)


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
    Solución analítica usando exponencial de matrices
    x(t) = (2e^(5t) - e^(2t)) / 1
    y(t) = (2e^(5t) + 2e^(2t)) / 1
    """
    x = (2*np.exp(5*t) - np.exp(2*t)) / 1
    y = (2*np.exp(5*t) + 2*np.exp(2*t)) / 1
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