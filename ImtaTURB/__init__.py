"""
Paquete aritmetica: Librería de operaciones aritméticas.

Este paquete expone las funciones principales del módulo operaciones para
un uso fácil e intuitivo.

"""

from .operaciones import suma, fluctuante, cargar_vectrino, autocorrelacion_norm, plot_ui, plot_ui_grid, espectros_tensor, correlacion_cruzada_norm, cargar_U_OpenFOAM, recortar_tiempo, cargar_p_OpenFOAM, plot_p, cargar_nut_OpenFOAM, plot_nut, cp_promedio_por_theta

__all__ = ["suma", "fluctuante", "cargar_vectrino", "autocorrelacion_norm", "plot_ui", "plot_ui_grid", "espectros_tensor", "correlacion_cruzada_norm", "cargar_U_OpenFOAM", "recortar_tiempo","cargar_p_OpenFOAM", "plot_p", "cargar_nut_OpenFOAM", "plot_nut", "cp_promedio_por_theta"]