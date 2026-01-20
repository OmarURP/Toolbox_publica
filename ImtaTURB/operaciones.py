"""
Módulo de operaciones para análisis de turbulencia básicas.

Este módulo contiene funciones para realizar análisis de turbulencia
basadas en el análisis de series temporales de datos de simulaciones
numéricas o experimentales.
"""
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import pandas as pd
import re
from scipy.signal import find_peaks


def suma(a, b):
    """
    Realiza la suma de dos números.

    Args:
        a (int o float): Primer número a sumar.
        b (int o float): Segundo número a sumar.

    Returns:
        int o float: El resultado de a + b.
    """
    return a + b


# Función para calcular las propiedades fluctuantes
def fluctuante(serie, tiempo=None, plot=False, titulo='Gráfico de velocidades'):
    """
    Calcula el valor medio instantáneo y su componente fluctuante-media
    a partir del campo instantáneo. Opcionalmente genera un gráfico de la
    señal instantánea y su parte fluctuante.

    Esta función implementa la descomposición de Reynolds:
        φ(x,t) = φ̄(x) + φ'(x,t)
    donde φ̄ es la media y φ' es la fluctuante.
    
    Parámetros
    ----------
    serie : array_like
        Serie instantánea de datos (por ejemplo, presión, velocidad, etc.).
    tiempo : array_like, optional
        Vector de tiempo asociado a la serie (mismo tamaño que `serie`).
        Solo es necesario si `plot=True`.
    plot : bool, optional
        Si True, genera una figura con:
        - Arriba: serie instantánea y su media.
        - Abajo: serie fluctuante y su media (≈ 0).
        Por defecto False.
    titulo : str, optional
        Título base para la figura (por defecto 'Velocidad en un punto').

    Retorna
    -------
    resultado : dict
        Diccionario con:
        - 'inst_med': Valor medio de la serie instantánea original (φ̄).
        - 'fluc'    : Serie de fluctuaciones (serie - media) (φ').
        - 'fluc_med': Media de las fluctuaciones (teóricamente ≈ 0).
        - 'serie'   : Serie original proporcionada.
    fig : matplotlib.figure.Figure or None
        Si plot=True, devuelve la figura generada.
        Si plot=False, devuelve None.

    Ejemplos
    --------
    >>> u = np.random.randn(1000) + 0.2  # señal con media 0.2
    >>> t = np.linspace(0, 10, 1000)
    >>> u_fluc, fig = fluctuante(u, tiempo=t, plot=True, titulo='u₁ en celda 4')
    >>> print(u_fluc['inst_med'])
    """
    # Convertir a array de numpy
    serie = np.asarray(serie)
    
    # Calcular la media de la serie
    inst_med = np.mean(serie)
    
    # Calcular las fluctuaciones
    fluc = serie - inst_med
    
    # Calcular la media de las fluctuaciones (teóricamente debería ser cercana a 0)
    fluc_med = np.mean(fluc)
    
    resultado = {
        'inst_med': inst_med,
        'fluc': fluc,
        'fluc_med': fluc_med,
        'serie': serie
    }

    fig = None
    if plot:
        if tiempo is None:
            raise ValueError("Si plot=True, debes proporcionar el vector 'tiempo'.")
        tiempo = np.asarray(tiempo)
        if tiempo.shape != serie.shape:
            raise ValueError("El vector 'tiempo' debe tener la misma forma que 'serie'.")

        # Gráfico
        fig, axs = plt.subplots(2, 1, figsize=(10, 5), sharex=True, sharey=False)
        ax1, ax2 = axs

        # Velocidad instantánea
        ax1.plot(tiempo, serie, linewidth=0.5, alpha=0.7)
        ax1.axhline(inst_med,
                    color='darkblue',
                    linestyle=':',
                    label=r'$\overline{U}=$' + f'{inst_med:.3f} m/s')
        ax1.legend(loc='upper right')
        ax1.set_ylabel(r'$u_i$ (m/s)')
        ax1.set_title(f'{titulo} - Velocidad instantánea')

        # Velocidad fluctuante
        ax2.plot(tiempo, fluc, linewidth=0.5, alpha=0.7)
        ax2.axhline(fluc_med,
                    color='k',
                    linestyle=':',
                    label=r'$\overline{u_{i}}^\prime=$' + f'{fluc_med:.3f} m/s')
        ax2.set_ylabel(r'$u_{i}^\prime$ (m/s)')
        ax2.set_xlabel(r'$t$ (s)')
        ax2.legend(loc='upper right')
        ax2.set_title(f'{titulo} - Velocidad fluctuante')

        plt.tight_layout()

        # Cambiar fondo a transparente de la figura y blanco en los ejes
        fig.patch.set_facecolor('none')
        ax1.set_facecolor('white')
        ax2.set_facecolor('white')

    return resultado, fig

# Función para cargar datos de Vectrino Profiler desde archivo .mat
def cargar_vectrino(file):
    """
    Carga datos de velocidad y configuración desde archivo .mat de Vectrino Profiler.
    
    Esta función maneja automáticamente claves con caracteres nulos y espacios
    generados por algunas versiones de MATLAB.
    
    Parámetros
    ----------
    file : str
        Ruta al archivo .mat con datos de Vectrino
    
    Retorna
    -------
    U : dict
        Diccionario con componentes de velocidad (m/s):
        - 'u1': Velocidad en X (Profiles_VelX)
        - 'u2': Velocidad en Y (Profiles_VelY)
        - 'u3': Velocidad en Z1 (Profiles_VelZ1)
        - 'u4': Velocidad en Z2 (Profiles_VelZ2)
        Cada componente tiene shape (n_muestras, n_puntos_espaciales)
    
    tiempo : numpy.ndarray
        Vector de tiempo en segundos (referenciado al instante inicial t=0)
        Shape: (n_muestras,)
    
    parametros : dict
        Parámetros de configuración del equipo:
        - 'frec': Frecuencia de muestreo (Hz)
        - 'vel_sonido': Velocidad del sonido (m/s)
        - 'fecha': Fecha de la medición
        - Otros campos disponibles en Config
    
    Ejemplos
    --------
    >>> # Cargar datos
    >>> U, tiempo, parametros = cargar_vectrino('A_1.mat')
    >>> 
    >>> # Acceder a componentes de velocidad
    >>> u1 = U['u1']  # Velocidad en X
    >>> print(f"Dimensiones: {u1.shape}")
    >>> print(f"Duración: {tiempo[-1]:.2f} segundos")
    >>> print(f"Frecuencia: {parametros['frec']:.1f} Hz")
    >>> 
    >>> # Extraer serie temporal en un punto del perfil
    >>> punto = 5
    >>> u1_serie = U['u1'][:, punto]
    
    Notas
    -----
    - Los datos de Vectrino Profiler tienen múltiples puntos de medición espaciales
      (celdas) a lo largo del volumen de medición
    - El vector tiempo se normaliza restando el instante inicial (t=0)
    - Las velocidades están en m/s
    - La estructura típica es (n_muestras_temporales, n_celdas_espaciales)
    
    Referencias
    -----------
    Nortek AS. Vectrino Profiler User Guide.
    """
    # Cargar archivo .mat
    datos = sio.loadmat(file)
    
    # Encontrar estructura "Data"
    data_key = None
    for key in datos.keys():
        if not key.startswith('__') and 'Data' in key:
            data_key = key
            break
    
    if data_key is None:
        claves_disponibles = [k for k in datos.keys() if not k.startswith('__')]
        raise KeyError(f"No se encontró una clave que contenga 'Data'. "
                      f"Claves disponibles: {claves_disponibles}")
      
    # Encontrar estructura "Config"
    config_key = None
    for key2 in datos.keys():
        if not key2.startswith('__') and 'Config' in key2:
            config_key = key2
            break
    
    if config_key is None:
        claves_disponibles = [k for k in datos.keys() if not k.startswith('__')]
        raise KeyError(f"No se encontró una clave que contenga 'Config'. "
                      f"Claves disponibles: {claves_disponibles}")
    
    # Acceder a las estructuras
    Data = datos[data_key]
    Config = datos[config_key]
    
    # Extraer componentes de velocidad (m/s)
    try:
        u_1 = Data['Profiles_VelX'][0, 0]   # U_X
        u_2 = Data['Profiles_VelY'][0, 0]   # U_Y
        u_3 = Data['Profiles_VelZ1'][0, 0]  # U_Z1
        u_4 = Data['Profiles_VelZ2'][0, 0]  # U_Z2
    except (KeyError, IndexError) as e:
        campos_disponibles = Data.dtype.names if hasattr(Data.dtype, 'names') else []
        raise KeyError(f"Error extrayendo velocidades: {e}. "
                      f"Campos disponibles en Data: {campos_disponibles}")
    
    # Crear diccionario de velocidades
    U = {
        'u1': u_1,
        'u2': u_2,
        'u3': u_3,
        'u4': u_4
    }
    
    # Extraer y procesar vector tiempo
    try:
        timestamp = Data['Profiles_TimeStamp'][0, 0]
        # Restar primer elemento para obtener tiempo relativo (t=0 al inicio)
        tiempo = np.array([timestamp[i, 0] - timestamp[0, 0] 
                          for i in range(timestamp.shape[0])])
    except (KeyError, IndexError) as e:
        raise KeyError(f"Error extrayendo timestamp: {e}")
    
    # Extraer parámetros de configuración
    parametros = {}
    
    # Parámetros principales
    try:
        parametros['frec'] = Config['sampleRate'][0, 0]  # Frecuencia de muestreo (Hz)
        parametros['vel_sonido'] = Config['speedOfSound'][0, 0]  # Velocidad del sonido (m/s)
    except (KeyError, IndexError) as e:
        print(f"Advertencia: Error extrayendo algunos parámetros: {e}")
    
    # Reporte de archivo cargado
    print("Datos cargados")
    print(f"Archivo: {file}")
    print(f"\nComponentes de velocidad [u_i]:")
    for key, val in U.items():
        print(f"  {key}: {val.shape}")
    print(f"\nTiempo= {tiempo.shape}")
    print(f"  Duración = {tiempo[-1]:.2f} s")
    print(f"  dt = {np.mean(np.diff(tiempo)):.4f} s")
    print(f"\nParámetros de configuración de equipo:")
    print(f"  Frecuencia de muestreo: {parametros['frec'][0, 0]:.2f} Hz")
    print(f"  Velocidad del sonido: {parametros['vel_sonido'][0, 0]:.2f} m/s")
    
    return U, tiempo, parametros

# Función para calcular el tensor de autocorrelación normalizado
def autocorrelacion_norm(autoc_ux, autoc_uy, autoc_uz, frecuencia,
                         plot=True, titulo=r'Tensor de Autocorrelación ($R_{ij}$)', max_lag_seg=None):
    """
    Calcula el tensor de autocorrelación normalizado R_ij(τ) para tres componentes
    de velocidad (u_x, u_y, u_z) y devuelve el vector de lags en segundos.

    Definición:
      R_ij(τ) = < u_i(t) * u_j(t + τ) >
    Cálculo en modo 'full' (lags desde -(N-1) hasta +(N-1)).

    Normalización:
      - Diagonales: se normalizan por R_ii(0).
      - Cruzadas: se normalizan por sqrt(R_ii(0) * R_jj(0)).

    Parámetros
    ----------
    autoc_ux, autoc_uy, autoc_uz : array_like, 1D
        Series de fluctuaciones (u' para cada componente). Deben tener la misma longitud.
    frecuencia : float
        Frecuencia de muestreo en Hz (se usa para convertir lags a segundos).
    plot : bool, optional
        Si True, genera la figura 3x3 con las subplots del tensor.
    max_lag_seg : float or None, optional
        Si se proporciona, recorta las curvas a ±max_lag_seg para la gráfica.

    Retorna
    -------
    tensor_r : dict
        Diccionario con las 9 componentes normalizadas:
        'r11','r12','r13','r21','r22','r23','r31','r32','r33'
        Cada entrada es un array de longitud 2*N-1 (modo 'full').
    lags_s : numpy.ndarray
        Vector de lags en segundos (longitud 2*N-1) correspondiente a las entradas.
    fig : matplotlib.figure.Figure or None
        Si plot=True, devuelve la figura; en caso contrario None.
    """
    # Convertir a arrays 1D y validar longitudes
    ux = np.asarray(autoc_ux)
    uy = np.asarray(autoc_uy)
    uz = np.asarray(autoc_uz)

    N = ux.size
    fs = frecuencia

    # Diagonal principal
    r11 = np.correlate(ux, ux, mode='full')
    r22 = np.correlate(uy, uy, mode='full')
    r33 = np.correlate(uz, uz, mode='full')
    # Cruzadas
    r12 = np.correlate(ux, uy, mode='full')
    r13 = np.correlate(ux, uz, mode='full')
    #r21 = np.correlate(uy, ux, mode='full')
    r23 = np.correlate(uy, uz, mode='full')
    #r31 = np.correlate(uz, ux, mode='full')
    #r32 = np.correlate(uz, uy, mode='full')

    # índice central (tau = 0)
    mid = r11.size // 2

    # valores escalares en tau=0 (pueden ser cero)
    r11_norm = r11[mid]
    r22_norm = r22[mid]
    r33_norm = r33[mid]

    # Denominadores para las cruzadas (sin protección)
    denom_12 = np.sqrt(r11_norm * r22_norm)
    denom_13 = np.sqrt(r11_norm * r33_norm)
    denom_23 = np.sqrt(r22_norm * r33_norm)

    tensor_r = {}
    # DIVISIONES DIRECTAS (sin comprobaciones)
    tensor_r['r11'] = r11 / r11_norm
    tensor_r['r22'] = r22 / r22_norm
    tensor_r['r33'] = r33 / r33_norm

    tensor_r['r12'] = r12 / denom_12
    #tensor_r['r21'] = r21 / denom_12

    tensor_r['r13'] = r13 / denom_13
    #tensor_r['r31'] = r31 / denom_13

    tensor_r['r23'] = r23 / denom_23
    #tensor_r['r32'] = r32 / denom_23

    # lags (en muestras) y convertir a segundos
    lags = np.arange(-N + 1, N)
    lags_s = lags / fs

    fig = None
    if plot:
        # Preparar recorte para la gráfica si se pide max_lag_seg
        if max_lag_seg is not None:
            mask = np.abs(lags_s) <= float(max_lag_seg)
            lags_plot = lags_s[mask]
            def _pick(a): return a[mask]
        else:
            lags_plot = lags_s
            def _pick(a): return a

        fig, axs = plt.subplots(3, 3, figsize=(12, 6), sharex=True, sharey=True)
        fig.suptitle(titulo, fontweight='bold')

        # Fila 1 (Correlaciones con ux)
        axs[0, 0].plot(lags_plot, _pick(tensor_r['r11']), 'b-')
        axs[0, 0].set_ylabel('$R_{11}$')

        axs[0, 1].plot(lags_plot, _pick(tensor_r['r12']),
                       linestyle='-', color='b')
        axs[0, 1].set_ylabel('$R_{12}$')

        axs[0, 2].plot(lags_plot, _pick(tensor_r['r13']),
                       linestyle='-', color='b')
        axs[0, 2].set_ylabel('$R_{13}$')

        # Fila 2
        axs[1, 0].axis('off')
        axs[1, 1].plot(lags_plot, _pick(tensor_r['r22']), 'b-')
        axs[1, 1].set_ylabel('$R_{22}$')
        axs[1, 1].tick_params(axis='y', labelleft=True)
        axs[1, 2].plot(lags_plot, _pick(tensor_r['r23']),
                       linestyle='-', color='b')
        axs[1, 2].set_ylabel('$R_{23}$')

        # Fila 3
        axs[2, 0].axis('off')
        axs[2, 1].axis('off')
        axs[2, 2].plot(lags_plot, _pick(tensor_r['r33']), 'b-')
        axs[2, 2].set_ylabel('$R_{33}$')
        axs[2, 2].tick_params(axis='y', labelleft=True)

        # Grid y ajustes para los ejes visibles
        for ax in axs.flat:
            if ax.axison:
                ax.grid(True)

        # Etiquetas en la diagonal (x)
        for i in range(3):
            axs[i, i].set_xlabel('Desfase $\\tau$ (s)')
            axs[i, i].tick_params(axis='x', labelbottom=True)
            if i > 0:
                axs[i, i].tick_params(axis='y', labelleft=True)

        plt.tight_layout()

        # Fondo transparente de la figura y ejes con fondo blanco
        fig.patch.set_facecolor('none')
        for ax in axs.flat:
            ax.set_facecolor('white')

    return tensor_r, lags_s, fig

# Función para graficar velocidades fluctuantes en layout grid
def plot_ui_grid(tiempo, u1_fluc, u2_fluc, u3_fluc,
                 titulo='Velocidades Instantáneas y Fluctuantes',
                 etiquetas=('u_1', 'u_2', 'u_3')):
    """
    Grafica en una sola figura las velocidades instantáneas y fluctuantes
    de tres componentes usando un layout tipo grid (3 filas x 2 columnas).

    Las etiquetas se especifican como expresiones LaTeX SIN los símbolos de dólar.
    Por ejemplo: etiquetas=('u_1', 'u_2', 'u_3').

    Parámetros
    ----------
    tiempo : array_like
        Vector de tiempo (s).
    u1_fluc, u2_fluc, u3_fluc : dict
        Diccionarios devueltos por imta.fluctuante() para cada componente.
        Deben contener las claves: 'serie', 'inst_med', 'fluc', 'fluc_med'.
    titulo : str, opcional
        Título general de la figura.
    etiquetas : tuple of str, opcional
        Etiquetas LaTeX para cada componente (para títulos y leyendas),
        sin los signos de $$. Ejemplo: ('u_1', 'u_2', 'u_3').

    Retorna
    -------
    fig : matplotlib.figure.Figure
        Figura creada.
    axs : ndarray of Axes
        Array 2D (3x2) de ejes.
    """
    tiempo = np.asarray(tiempo)

    comps = [u1_fluc, u2_fluc, u3_fluc]
    colores = ('C0', 'C1', 'C2')  # tres colores distintos

    fig, axs = plt.subplots(3, 2, figsize=(10, 5), sharex=True)
    fig.suptitle(titulo, fontweight='bold')

    for i, (uf, label, color) in enumerate(zip(comps, etiquetas, colores)):
        # label es la expresión LaTeX sin $, por ejemplo 'u_1'
        serie = uf['serie']
        inst_med = uf['inst_med']
        fluc = uf['fluc']
        fluc_med = uf['fluc_med']

        ax_inst = axs[i, 0]
        ax_fluc = axs[i, 1]

        # Instantánea
        ax_inst.plot(tiempo, serie, linewidth=0.5, alpha=0.7, color=color)
        ax_inst.axhline(
            inst_med,
            color=color,
            linestyle=':',
            label=rf'$\overline{{{label}}}={inst_med:.3f}$ m/s'
        )
        ax_inst.set_ylabel(rf'${label}$ (m/s)')
        ax_inst.legend(loc='upper right', fontsize=8)
        if i == 0:
            ax_inst.set_title('Velocidad instantánea')

        # Fluctuante
        ax_fluc.plot(tiempo, fluc, linewidth=0.5, alpha=0.7, color=color)
        ax_fluc.axhline(
            fluc_med,
            color=color,
            linestyle=':',
            label=rf'$\overline{{{label}^\prime}}={fluc_med:.3f}$ m/s'
        )
        ax_fluc.legend(loc='upper right', fontsize=8)
        if i == 0:
            ax_fluc.set_title('Velocidad fluctuante')
        ax_fluc.set_ylabel(rf'${label}^\prime$ (m/s)')

        # Grid
        ax_inst.grid(True, alpha=0.3, linestyle=':')
        ax_fluc.grid(True, alpha=0.3, linestyle=':')

    # Etiquetas de tiempo en la última fila
    axs[-1, 0].set_xlabel(r'$t$ (s)')
    axs[-1, 1].set_xlabel(r'$t$ (s)')

    plt.tight_layout()

    # Fondos
    fig.patch.set_facecolor('none')
    for ax in axs.flat:
        ax.set_facecolor('white')

    return fig, axs

# Función para graficar velocidades fluctuantes en layout 2x1
def plot_ui(tiempo, u1_fluc, u2_fluc, u3_fluc,
          titulo='Velocidades Instantáneas y Fluctuantes',
          etiquetas=('u_1', 'u_2', 'u_3')):
    """
    Grafica en una sola figura las velocidades instantáneas y fluctuantes de
    tres componentes en dos paneles (2 filas x 1 columna).

    Panel superior:  u_i(t) con sus valores medios U_i = \\overline{u_i}
    Panel inferior:  u_i'(t) con la media de las fluctuaciones (≈ 0)

    Parámetros
    ----------
    tiempo : array_like
        Vector de tiempo (s).
    u1_fluc, u2_fluc, u3_fluc : dict
        Diccionarios devueltos por fluctuante() para cada componente.
        Deben contener las claves: 'serie', 'inst_med', 'fluc', 'fluc_med'.
    titulo : str, opcional
        Título general de la figura.
    etiquetas : tuple of str, opcional
        Etiquetas LaTeX sin $ para cada componente, por ejemplo:
        ('u_1', 'u_2', 'u_3').

    Retorna
    -------
    fig : matplotlib.figure.Figure
        Figura creada.
    axs : ndarray of Axes
        Array 1D de ejes (2 elementos: [ax1, ax2]).
    """
    tiempo = np.asarray(tiempo)

    # Extraer datos
    comps = [u1_fluc, u2_fluc, u3_fluc]

    fig, axs = plt.subplots(2, 1, figsize=(10, 5), sharex=True, sharey=False)
    ax1, ax2 = axs
    fig.suptitle(titulo, fontweight='bold')

    # ===== Panel superior: velocidades instantáneas =====
    for uf, label, color in zip(
        comps,
        etiquetas,
        ('C0', 'C1', 'C2')   # colores por defecto de matplotlib
    ):
        serie = uf['serie']
        inst_med = uf['inst_med']

        # curva instantánea
        ax1.plot(tiempo, serie, linewidth=0.5, alpha=0.7)
        # línea de la media
        ax1.axhline(
            inst_med,
            color=color,
            linestyle=':',
            label=rf'$\overline{{{label}}}={inst_med:.3f}$ m/s'
        )

    ax1.set_ylabel(r'$u_i$ (m/s)')
    ax1.set_title('Velocidad instantánea')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3, linestyle=':')

    # ===== Panel inferior: velocidades fluctuantes =====
    for uf, label, color in zip(
        comps,
        etiquetas,
        ('C0', 'C1', 'C2')
    ):
        fluc = uf['fluc']
        fluc_med = uf['fluc_med']

        ax2.plot(tiempo, fluc, linewidth=0.5, alpha=0.7)

    # Una sola línea de referencia para la media de las fluctuaciones
    ax2.axhline(
        u1_fluc['fluc_med'],
        color='k',
        linestyle=':',
        label=rf'$\overline{{u_i^\prime}}={u1_fluc["fluc_med"]:.3f}$ m/s'
    )

    ax2.set_ylabel(r'$u_i^\prime$ (m/s)')
    ax2.set_xlabel(r'$t$ (s)')
    ax2.set_title('Velocidad fluctuante')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3, linestyle=':')

    plt.tight_layout()

    # Fondos
    fig.patch.set_facecolor('none')
    ax1.set_facecolor('white')
    ax2.set_facecolor('white')

    return fig, axs

def espectros_tensor(tensor_r, frecuencia,
                     titulo=r'Espectros de Energía del Tensor de Correlación $S(R_{ij})$',
                     plot=True,
                     dpi=600):
    """
    Calcula y, opcionalmente, grafica los espectros de energía S(R_ij) para
    las componentes del tensor de correlación R_ij(tau).

    Funciona para dos casos de entrada:
      - Tensor simétrico (solo las 6 claves superiores): 'r11','r12','r13','r22','r23','r33'
        -> se dibuja la parte triangular superior en una malla 3x3 (celdas inferiores desactivadas)
        -> se muestran números en X solo en la diagonal principal (r11,r22,r33)
      - Tensor completo (las 9 claves presentes): 'r11',...,'r33'
        -> se dibujan las 9 subplots activas y se muestran números en X e Y en todas ellas

    Parámetros
    ----------
    tensor_r : dict
        Diccionario con las componentes normalizadas del tensor de autocorrelación.
    frecuencia : float
        Frecuencia de muestreo en Hz.
    titulo : str, opcional
        Título global de la figura.
    dpi : int, opcional
        Resolución de la figura.
    plot : bool, opcional
        Si True, genera la figura 3x3 con los espectros. Si False, no grafica.

    Retorna
    -------
    frecs_pos_dict : dict
        Diccionario con las frecuencias positivas para cada componente calculada.
    espectro_dict : dict
        Diccionario con los espectros de energía (|FFT|^2) para cada componente.
    fig : matplotlib.figure.Figure or None
        Figura con los espectros si plot=True; en caso contrario, None.
    """
    fs = float(frecuencia)

    # Detectar si se pasaron las 9 componentes o solo las 6 superiores
    keys = set(tensor_r.keys())
    keys_6 = {'r11','r12','r13','r22','r23','r33'}
    keys_9 = {'r11','r12','r13','r21','r22','r23','r31','r32','r33'}

    if keys_9.issubset(keys):
        full = True
        tensor_ind = [
            ['r11', 'r12', 'r13'],
            ['r21', 'r22', 'r23'],
            ['r31', 'r32', 'r33']
        ]
    elif keys_6.issubset(keys):
        full = False
        tensor_ind = [
            ['r11', 'r12', 'r13'],
            [None,  'r22', 'r23'],
            [None,  None,  'r33']
        ]
    else:
        missing = sorted(list(keys_6 - keys))
        raise ValueError(f"Las claves esperadas no están presentes. Al menos falta(n): {missing}")

    # Colores por fila (mismas que en autocorrelación)
    colores = ['b', 'b', 'b']

    estilos_picos = {
        'P1': {'marker': 'o', 'color': 'darkblue',    'markersize': 6},
        'P2': {'marker': 'o', 'color': 'darkgreen', 'markersize': 5},
        'P3': {'marker': 'o', 'color': 'darkred','markersize': 4},
        'P4': {'marker': 'o', 'color': 'black',   'markersize': 3},
    }
    claves_picos = list(estilos_picos.keys())

    frecs_pos_dict = {}
    espectro_dict = {}

    fig = None
    if plot:
        fig, axs = plt.subplots(3, 3, figsize=(12, 8), sharex=True, sharey=True, dpi=dpi)
        fig.suptitle(titulo, fontweight='bold')

    for i in range(3):
        for j in range(3):
            ind = tensor_ind[i][j]

            # Caso triangular: desactivar subplots que no corresponden
            if ind is None:
                if plot:
                    axs[i, j].axis('off')
                continue

            corr_senal = np.asarray(tensor_r[ind])
            n = len(corr_senal)

            # FFT y frecuencias
            fft_val = np.fft.fft(corr_senal)
            frecs = np.fft.fftfreq(n, d=1.0/fs)

            # Espectro de un solo lado (frecuencias positivas)
            indices_positivos = frecs >= 0
            frecs_pos = frecs[indices_positivos]
            espectro = np.abs(fft_val[indices_positivos])**2

            frecs_pos_dict[ind] = frecs_pos
            espectro_dict[ind] = espectro

            # Encontrar la frecuencia con energía máxima
            idx_max = np.argmax(espectro)
            frec_max = frecs_pos[idx_max]
            energia_max = espectro[idx_max]
            print(f"Componente {ind}: Frecuencia máxima = {frec_max:.3f} Hz, Energía = {energia_max:.3e}")

            if not plot:
                continue

            ax = axs[i, j]

            # Espectro en escala log-log
            ax.loglog(frecs_pos, espectro, color=colores[i], linewidth=0.5)
            ax.set_ylabel(fr'$S(R_{{{ind[1:]}}})(s^2)$')
            ax.grid(True, which="both", ls=":", alpha=0.5)

            # Detección de picos y marcadores
            indices_picos, propiedades = find_peaks(espectro, height=0)
            if len(indices_picos) > 0:
                indices_ordenados = np.argsort(propiedades['peak_heights'])[::-1]
                indices_top = indices_picos[indices_ordenados[:4]]

                for k, pico_idx in enumerate(indices_top):
                    frec_pico = frecs_pos[pico_idx]
                    val_pico = espectro[pico_idx]
                    etiqueta_pico = f'{claves_picos[k]}: {frec_pico:.3f} Hz'

                    ax.plot(
                        frec_pico, val_pico,
                        label=etiqueta_pico,
                        **estilos_picos[claves_picos[k]],
                        zorder=2, linestyle='None'
                    )

                ax.legend(loc='lower center', ncol=2, fontsize=9,
                          facecolor='white', framealpha=0.7)

            # Control de ticks y etiquetas según modo
            if full:
                # Mostrar números en X e Y en TODOS los subplots activos
                ax.tick_params(axis='x', which='both', labelbottom=True)
                ax.tick_params(axis='y', which='both', labelleft=True)
                ax.set_xlabel('Frecuencia (Hz)')
            else:
                # Triangular: solo la diagonal principal muestra números en X
                ax.tick_params(axis='x', which='both', labelbottom=False)
                ax.tick_params(axis='y', which='both', labelleft=True)
                if i == j:
                    ax.set_xlabel('Frecuencia (Hz)')
                    ax.tick_params(axis='x', which='both', labelbottom=True)

    if plot and fig is not None:
        plt.tight_layout()
        # Fondo transparente y ejes blancos
        fig.patch.set_facecolor('none')
        for ax in fig.axes:
            if ax.axison:
                ax.set_facecolor('white')

    return frecs_pos_dict, espectro_dict, fig

# Función para calcular el tensor de correlación cruzada normalizado
def correlacion_cruzada_norm(ux_senal1, ux_senal2,
                             uy_senal1, uy_senal2,
                             uz_senal1, uz_senal2,
                             frecuencia,
                             plot=True,
                             titulo=r'Tensor de Correlación Cruzada ($R_{ij}$)',
                             max_lag_seg=None):
    """
    Calcula el tensor de correlación cruzada normalizado R_ij(τ) entre dos
    conjuntos de series (senal1, senal2) con las tres componentes u_x, u_y, u_z.

    Definición:
      R_ij(τ) = < u_i^(senal1)(t) * u_j^(senal2)(t + τ) >

    Cálculo en modo 'full' (lags desde -(N-1) hasta +(N-1)) y normalización:
      - Se normaliza cada componente por sqrt( autocorr_i_s1(0) * autocorr_j_s2(0) ).

    Parámetros
    ----------
    ux_senal1, ux_senal2, uy_senal1, uy_senal2, uz_senal1, uz_senal2 : array_like, 1D
        Series de fluctuaciones (u' para cada componente) para la señal 1 y señal 2.
        Todas las series deben tener la misma longitud N.
    frecuencia : float
        Frecuencia de muestreo en Hz (se usa para convertir lags a segundos).
    plot : bool, optional
        Si True, genera la figura 3x3 con las 9 subplots (todas activas).
    titulo : str, optional
        Título de la figura.
    max_lag_seg : float or None, optional
        Si se proporciona, recorta las curvas a ±max_lag_seg para la gráfica.

    Retorna
    -------
    tensor_r : dict
        Diccionario con las 9 componentes normalizadas:
        'r11','r12','r13','r21','r22','r23','r31','r32','r33'
        Cada entrada es un array de longitud 2*N-1 (modo 'full').
    lags_s : numpy.ndarray
        Vector de lags en segundos (longitud 2*N-1) correspondiente a las entradas.
    fig : matplotlib.figure.Figure or None
        Si plot=True, devuelve la figura; en caso contrario None.
    """
    # Convertir a arrays 1D y validar longitudes
    ux1 = np.asarray(ux_senal1)
    ux2 = np.asarray(ux_senal2)
    uy1 = np.asarray(uy_senal1)
    uy2 = np.asarray(uy_senal2)
    uz1 = np.asarray(uz_senal1)
    uz2 = np.asarray(uz_senal2)
    # Validar tamaños
    N = ux1.size
    if not (ux2.size == uy1.size == uy2.size == uz1.size == uz2.size == N):
        raise ValueError("Todas las series deben tener la misma longitud")

    fs = float(frecuencia)

    # Autocorrelaciones (modo 'full') para obtener normas en tau=0 de cada componente
    a11 = np.correlate(ux1, ux1, mode='full')
    a22 = np.correlate(uy1, uy1, mode='full')
    a33 = np.correlate(uz1, uz1, mode='full')

    b11 = np.correlate(ux2, ux2, mode='full')
    b22 = np.correlate(uy2, uy2, mode='full')
    b33 = np.correlate(uz2, uz2, mode='full')
    mid = a11.size // 2
    # valores escalares en tau=0 para cada autocorrelación
    a11_0 = a11[mid]
    a22_0 = a22[mid]
    a33_0 = a33[mid]

    b11_0 = b11[mid]
    b22_0 = b22[mid]
    b33_0 = b33[mid]

    # Correlaciones cruzadas (modo 'full')
    r11 = np.correlate(ux1, ux2, mode='full')
    r12 = np.correlate(ux1, uy2, mode='full')
    r13 = np.correlate(ux1, uz2, mode='full')

    r21 = np.correlate(uy1, ux2, mode='full')
    r22 = np.correlate(uy1, uy2, mode='full')
    r23 = np.correlate(uy1, uz2, mode='full')
    r31 = np.correlate(uz1, ux2, mode='full')
    r32 = np.correlate(uz1, uy2, mode='full')
    r33 = np.correlate(uz1, uz2, mode='full')

    # Denominadores usando productos sqrt( autocorr_s1(0) * autocorr_s2(0) )
    denom_11 = np.sqrt(a11_0 * b11_0)
    denom_12 = np.sqrt(a11_0 * b22_0)
    denom_13 = np.sqrt(a11_0 * b33_0)

    denom_21 = np.sqrt(a22_0 * b11_0)
    denom_22 = np.sqrt(a22_0 * b22_0)
    denom_23 = np.sqrt(a22_0 * b33_0)

    denom_31 = np.sqrt(a33_0 * b11_0)
    denom_32 = np.sqrt(a33_0 * b22_0)
    denom_33 = np.sqrt(a33_0 * b33_0)

    # Construir diccionario del tensor (normalizaciones directas)
    tensor_r = {}
    tensor_r['r11'] = r11 / denom_11
    tensor_r['r12'] = r12 / denom_12
    tensor_r['r13'] = r13 / denom_13

    tensor_r['r21'] = r21 / denom_21
    tensor_r['r22'] = r22 / denom_22
    tensor_r['r23'] = r23 / denom_23

    tensor_r['r31'] = r31 / denom_31
    tensor_r['r32'] = r32 / denom_32
    tensor_r['r33'] = r33 / denom_33

    # lags (en muestras) y convertir a segundos
    lags = np.arange(-N + 1, N)
    lags_s = lags / fs

    fig = None
    if plot:
        # Preparar recorte para la gráfica si se pide max_lag_seg
        if max_lag_seg is not None:
            mask = np.abs(lags_s) <= float(max_lag_seg)
            lags_plot = lags_s[mask]
            def _pick(a): return a[mask]
        else:
            lags_plot = lags_s
            def _pick(a): return a

        fig, axs = plt.subplots(3, 3, figsize=(12, 8), sharex=True, sharey=True)
        fig.suptitle(titulo, fontweight='bold')

        colores = ['b', 'b', 'b'] 

        # Recorremos las 9 posiciones (todas activas)
        keys = [['r11', 'r12', 'r13'],
                ['r21', 'r22', 'r23'],
                ['r31', 'r32', 'r33']]

        for i in range(3):
            for j in range(3):
                ind = keys[i][j]
                ax = axs[i, j]

                ax.plot(lags_plot, _pick(tensor_r[ind]), color=colores[i], linewidth=0.5)
                ax.set_ylabel(f'$R_{{{ind[1:]}}}$')
                ax.grid(True)

                # Mostrar números en ambos ejes para TODOS los 9 subplots
                ax.tick_params(axis='x', labelbottom=True)
                ax.tick_params(axis='y', labelleft=True)

                # Mostrar etiqueta del eje x en todas las subplots (solicitado)
                ax.set_xlabel('Desfase $\\tau$ (s)')

        plt.tight_layout()

        # Fondo transparente de la figura y ejes con fondo blanco
        fig.patch.set_facecolor('none')
        for ax in axs.flat:
            ax.set_facecolor('white')

    return tensor_r, lags_s, fig

# Función para cargar datos de U en probes desde un archivo de OpenFOAM
def cargar_U_OpenFOAM(ruta_archivo: str):
    """
    Lee un archivo de postproceso de OpenFOAM (campo U en probes) y regresa:

    - U_probes: dict {probe_id: DataFrame con columnas ['Ux','Uy','Uz'] y eje índice = tiempo}
    - tiempo:   Series con el vector de tiempo (s)
    - coords:   dict {probe_id: (x, y, z)}
    - fs:       Frecuencia de muestreo (Hz)
    """

    # 1) Detectar automáticamente número de probes y línea donde inician los datos
    total_probes = 0
    data_start = None
    with open(ruta_archivo, "r") as f:
        for i, line in enumerate(f):
            if line.startswith("# Probe"):
                total_probes += 1
            elif line.startswith("# Time"):
                data_start = i + 1
                break

    if data_start is None:
        raise ValueError("No se encontró la línea '# Time' en el archivo.")

    # 2) Extraer coordenadas de los probes
    coords = {}  # {índice_probe: (x, y, z)}
    with open(ruta_archivo, "r") as f:
        for line in f:
            if line.startswith("# Time"):
                break
            m = re.match(r"#\s*Probe\s+(\d+)\s*\(([^)]+)\)", line)
            if m:
                idx = int(m.group(1))
                coords[idx] = tuple(map(float, m.group(2).split()))

    # 3) Leer datos numéricos (tiempo y U)
    df = pd.read_csv(
        ruta_archivo,
        skiprows=data_start,
        sep=r"\s+",
        comment="#",
        header=None,
        dtype=str  # Leer como string primero
    )

    # La primera columna es tiempo [s] - convertir directamente
    tiempo = pd.to_numeric(df[0], errors='coerce') # Coerce convierte errores en NaN

    # 4) Limpiar paréntesis y convertir a float todas las demás columnas
    for col in range(1, df.shape[1]):
        # Remover paréntesis usando str.replace
        col_clean = df[col].str.replace('(', '', regex=False).str.replace(')', '', regex=False)
        # Convertir a float
        df[col] = pd.to_numeric(col_clean, errors='coerce').astype('float32') # Omar del futuro (float32-7dígitos) (float64-15dígitos)
                                                                              # Pero float64 incrementa x 40 veces de tiempo, en frecuencias y visual sin diferencias 20/11/25
    # 5) Organizar velocidades por probe (tiempo como índice)
    U_probes = {}
    tiempo_values = tiempo.values  # Convertir una sola vez
    
    for p in range(total_probes):
        c0 = 1 + 3 * p
        c1 = c0 + 1
        c2 = c0 + 2
        U_probes[p] = pd.DataFrame(
            {"Ux": df[c0].values,
             "Uy": df[c1].values,
             "Uz": df[c2].values},
            index=tiempo_values,   # tiempo en el índice
        )

    # 6) Cálculos de duración y frecuencia de muestreo
    t_ini = float(tiempo.iloc[0])
    t_fin = float(tiempo.iloc[-1])
    duracion = t_fin - t_ini           # segundos

    if len(tiempo) > 1:
        dt = float(tiempo.iloc[1] - tiempo.iloc[0])
        fs = 1.0 / dt                  # Hz
    else:
        dt = float("nan")
        fs = float("nan")

    # 7) Reporte
    print(f"Se extrajeron {total_probes} probes.")
    print(f"Tiempo inicial: {t_ini:.2f} s, tiempo final: {t_fin:.2f} s")
    print(f"Duración de la muestra: {duracion:.2f} s")
    print(f"Frecuencia de muestreo: {fs:.2f} Hz (Δt = {dt:.4f} s)")

    return U_probes, tiempo, coords, fs

# Funcion para ventana de tiempo
def recortar_tiempo(U_probes, probe,
                          inicio=0.0,
                          fin=None,
                          columnas=('Ux', 'Uz', 'Uy')):
    """
    Recorta en tiempo los datos de un probe entre los tiempos `inicio` y `fin`.

    Parámetros
    ----------
    U_probes : dict-like
        Contenedor (por ejemplo, dict) con los DataFrames de cada probe.
        Se espera que U_probes[probe] sea un DataFrame con índice = tiempo (s).
    probe : hashable
        Clave del probe dentro de U_probes.
    inicio : float, opcional
        Tiempo inicial (en segundos) a partir del cual se conservan los datos.
        Se filtra con condición t >= inicio.
    fin : float or None, opcional
        Tiempo final (en segundos) hasta el cual se conservan los datos.
        Si es None, se toma todo hasta el final (t <= max).
    columnas : tuple of str, opcional
        Nombres de las columnas de velocidad en el DataFrame del probe.
        Por defecto: ('Ux', 'Uz', 'Uy').

    Retorna
    -------
    u1, u2, u3 : pandas.Series
        Series de velocidades instantáneas recortadas.
    tiempo : pandas.Index
        Índice de tiempo recortado (mismas longitudes que u1, u2, u3).
    dfp_filt : pandas.DataFrame
        DataFrame filtrado completo del probe (por si se quiere reutilizar).
    """
    # DataFrame completo de ese probe
    dfp = U_probes[probe]

    # Asegurar que el índice sea numérico/float (en segundos)
    dfp = dfp.copy()
    dfp.index = pd.to_numeric(dfp.index, errors='coerce')

    # Filtro por inicio y fin
    mask = dfp.index >= float(inicio)
    if fin is not None:
        mask &= dfp.index <= float(fin)

    dfp_filt = dfp[mask]

    # Extraer columnas de velocidad
    col_u1, col_u2, col_u3 = columnas
    u1 = dfp_filt[col_u1]
    u2 = dfp_filt[col_u2]
    u3 = dfp_filt[col_u3]

    # Vector de tiempo filtrado
    tiempo = dfp_filt.index

    return u1, u2, u3, tiempo, dfp_filt

