# Cómo compartir notebooks en Google Colab con módulos locales

Este README explica cómo hacer que tus notebooks funcionen correctamente en Google Colab cuando dependen del módulo personalizado `ImtaTURB`.

## Problema

Cuando compartes un notebook de Jupyter mediante el botón "Open in Colab", las personas que lo abren reciben el siguiente error:

```
ModuleNotFoundError: No module named 'ImtaTURB'
```

Esto ocurre porque Google Colab no tiene acceso al módulo `ImtaTURB` que existe solo en tu computadora local.

## Solución

Existen dos soluciones principales:

### 1. Instalar el módulo directamente desde GitHub (RECOMENDADO)

Esta es la mejor solución porque permite que cualquier persona ejecute tu notebook sin necesidad de archivos adicionales.

#### Paso 1: Asegúrate de que tu código esté en GitHub

Tu repositorio ya está en GitHub: `https://github.com/OmarURP/Toolbox_publica`

#### Paso 2: Agregar una celda de instalación al inicio del notebook

Agrega una nueva celda **AL PRINCIPIO** de cada notebook (antes de `import ImtaTURB as imta`), con el siguiente código:

```python
# Celda de instalación para Google Colab
try:
    import google.colab
    IN_COLAB = True
except:
    IN_COLAB = False

# Si estamos en Colab, instalar el módulo desde GitHub
if IN_COLAB:
    print("Ejecutando en Google Colab - Instalando ImtaTURB desde GitHub...")
    !pip install git+https://github.com/OmarURP/Toolbox_publica.git
    print("Instalación completada!")
else:
    print("Ejecutando localmente")
```

#### Paso 3: Actualizar los imports en el notebook

No necesitas cambiar nada más. La siguiente celda puede seguir siendo:

```python
import ImtaTURB as imta
```

#### Ventajas de este método:
- ✅ Funciona automáticamente tanto en local como en Colab
- ✅ No requiere archivos adicionales
- ✅ Fácil de mantener
- ✅ Se instala la versión más reciente del código

### 2. Montar Google Drive (alternativa)

Si prefieres subir los archivos a Google Drive:

#### Paso 1: Sube tu carpeta `ImtaTURB` a Google Drive

#### Paso 2: Agrega esta celda al inicio del notebook:

```python
# Montar Google Drive
try:
    import google.colab
    IN_COLAB = True
except:
    IN_COLAB = False

if IN_COLAB:
    from google.colab import drive
    drive.mount('/content/drive')
    
    # Agregar la ruta donde está ImtaTURB al path
    import sys
    sys.path.append('/content/drive/MyDrive/ruta/a/tu/carpeta')
```

Reemplaza `'/content/drive/MyDrive/ruta/a/tu/carpeta'` con la ruta real donde subiste la carpeta que contiene `ImtaTURB`.

## Ejemplo de notebook modificado

Aquí está un ejemplo completo de cómo debería verse tu notebook:

```python
# CELDA 1: Instalación (solo en Colab)
try:
    import google.colab
    IN_COLAB = True
except:
    IN_COLAB = False

if IN_COLAB:
    print("Instalando ImtaTURB...")
    !pip install git+https://github.com/OmarURP/Toolbox_publica.git
    print("¡Listo!")

# CELDA 2: Imports
import ImtaTURB as imta

# CELDA 3: Tu código
file = r'Datos/Vectrino/20_cm/A_1.mat'
U, tiempo, parametros = imta.cargar_vectrino(file)
# ... resto del código
```

## Nota importante sobre datos

Ten en cuenta que los archivos de datos (como `Datos/Vectrino/20_cm/A_1.mat`) tampoco estarán disponibles en Colab. Tienes dos opciones:

1. Subirlos también a GitHub (si no son muy grandes)
2. Proporcionar instrucciones para que los usuarios suban sus propios archivos a Colab

Para subir archivos en Colab, los usuarios pueden usar:

```python
from google.colab import files
uploaded = files.upload()
```

## Dependencias

Asegúrate de que tu archivo `setup.py` liste todas las dependencias necesarias. Revisa y actualiza las dependencias en:

```python
install_requires=[
    "numpy",
    "pandas",
    "matplotlib",
    "scipy",
    # Agrega aquí cualquier otra biblioteca que uses
],
```

## Recursos adicionales

- [Documentación de Google Colab](https://colab.research.google.com/)
- [Instalar paquetes en Colab](https://colab.research.google.com/notebooks/snippets/importing_libraries.ipynb)

## ¿Necesitas ayuda?

Si tienes problemas, revisa:
1. Que el repositorio sea público en GitHub
2. Que el archivo `setup.py` esté en la raíz del repositorio
3. Que todas las dependencias estén listadas en `setup.py`
