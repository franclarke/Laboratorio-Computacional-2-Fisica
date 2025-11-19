# Laboratorio Computacional 2 – Física II

## Descripción

Simulación del **campo magnético** usando la **Ley de Biot–Savart**.  
El proyecto calcula y visualiza el campo generado por:

- Un **alambre recto finito**
- Una **espira circular**
- La **superposición** del campo total (principio de superposición)
- Visualizaciones 2D y 3D del campo magnético

Desarrollado en **Python**, utilizando `numpy` para el cálculo numérico y `matplotlib` para las gráficas.

---

## Autores Principales
- **Antonio Carlos**
- **Tobias Thiessen**
- **Francisco Clarke**
- otros...

---

## Estructura del Proyecto

├── README.md
├── requirements.txt
├── src/
│ ├── main.py # Archivo principal de ejecución
│ ├── alambre.py # Cálculo del campo del alambre
│ ├── espira.py # Cálculo del campo de la espira
│ └── graficos.py # Funciones para graficar en 2D y 3D
├── graphics/ # Carpeta opcional para guardar imágenes
└── .venv/ # Entorno virtual de Python

## Requisitos

- Python 3.8 o superior  
- pip  
- Librerías indicadas en `requirements.txt`

---

## Instalación y Configuración

### 1. Clonar el repositorio

```bash
git clone https://github.com/tu-usuario/Laboratorio-Computacional-2-Fisica.git
cd Laboratorio-Computacional-2-Fisica
```
### 2. Crear y activar el entorno virtual

#### En macOS/Linux:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

#### En Windows:
```bash
python -m venv .venv
.venv\Scripts\activate
```

### 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

## Cómo Ejecutar el Proyecto

### Ejecución desde la raíz del proyecto:

```bash
python src/main.py
```

### Ejecución desde la carpeta src:

```bash
cd src
python main.py
```
