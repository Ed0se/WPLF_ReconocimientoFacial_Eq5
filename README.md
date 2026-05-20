# Reconocimiento Facial por Grafos

Un sistema de reconocimiento facial basado en grafos faciales que utiliza puntos característicos del rostro para identificar y comparar personas.

## 📋 Descripción

Este proyecto implementa un sistema de reconocimiento facial alternativo que no utiliza redes neuronales profundas, sino un enfoque geométrico basado en grafos. El sistema captura puntos faciales clave (30 puntos estratégicos) y construye un grafo que representa la estructura del rostro. Estos grafos se normalizan y se comparan usando métricas de distancia para identificar personas.

**Proyecto final de la materia**: Programación Lógica Funcional (WPLF)
**Equipo**: 5

## ✨ Características Principales

- **Captura de puntos faciales**: Interface interactiva para seleccionar 30 puntos clave en un rostro
- **Registro de personas**: Almacena vectores de características de rostros conocidos
- **Reconocimiento en tiempo real**: Compara rostros capturados contra una base de datos
- **Análisis de similitud**: Proporciona métricas detalladas de similitud (porcentaje y nivel)
- **Visualización de grafos**: Muestra la estructura facial capturada visualmente
- **Interface gráfica**: Aplicación Tkinter amigable y fácil de usar

## 🛠️ Requisitos

- Python 3.7+
- OpenCV (`cv2`)
- NumPy
- Tkinter (incluido con Python)

## 📦 Instalación

### 1. Clonar el repositorio
```bash
git clone https://github.com/Ed0se/WPLF_ReconocimientoFacial_Eq5.git
cd WPLF_ReconocimientoFacial_Eq5
```

### 2. Crear un entorno virtual (recomendado)
```bash
python -m venv .venv
```

### 3. Activar el entorno virtual

**En Windows:**
```bash
.venv\Scripts\activate
```

**En macOS/Linux:**
```bash
source .venv/bin/activate
```

### 4. Instalar dependencias
```bash
pip install opencv-python numpy
```

## 🚀 Uso

### Ejecutar la aplicación
```bash
python reconocimiento.py
```

### Registrar un nuevo rostro

1. Click en el botón "Registrar Rostro"
2. Ingresar el nombre de la persona
3. Seleccionar una imagen que contenga el rostro
4. Hacer click en 30 puntos clave del rostro en orden
   - Los puntos se numerarán automáticamente
   - Presionar ESC para cancelar
5. El sistema guardará el vector de características y mostrará una confirmación

### Comparar/Reconocer un rostro

1. Click en el botón "Comparar Rostro"
2. Seleccionar una imagen con un rostro a identificar
3. Capturar los 30 puntos clave (igual que en el registro)
4. Ver resultados:
   - Indicará si la persona fue reconocida
   - Mostrará el porcentaje de similitud
   - Listará todas las comparaciones contra la base de datos

## 📊 Estructura del Proyecto

```
ReconocimientoFacial/
├── reconocimiento.py       # Aplicación principal
├── base_rostros/           # Directorio con vectores guardados (.npy)
│   ├── Diego.npy
│   ├── Eduardo.npy
│   ├── Hector.npy
│   ├── Kevin.npy
│   └── Maya.npy
├── imagenes/               # Imágenes de prueba
├── wrapper/                # Entorno virtual de Python
└── README.md              # Este archivo
```

## 🧠 Algoritmo de Funcionamiento

### Flujo Principal

1. **Captura de puntos**: Se seleccionan 30 puntos clave en la cara mediante clicks del ratón
2. **Normalización Procrustes**: Los puntos se normalizan usando:
   - Centrado: Trasladar al origen
   - Escalado: Dividir por la norma Euclidiana
3. **Extracción de características**: Se calcula la distancia entre pares conectados de puntos
4. **Comparación**: Se mide la distancia Euclidiana entre vectores de características
5. **Clasificación**: Se determina el nivel de similitud basado en umbrales

### Conexiones del Grafo

El sistema define 30 conexiones que conectan los 30 puntos estratégicamente seleccionados en el rostro, formando un grafo que representa su estructura geométrica.

### Umbral de Reconocimiento

| Nivel | Distancia | Reconocimiento |
|-------|-----------|---|
| Alta similitud | ≤ 0.075 | ✓ Reconocido |
| Media similitud | ≤ 0.15 | ✓ Reconocido |
| Baja similitud | ≤ 0.225 | ✗ No reconocido |
| Muy baja similitud | > 0.225 | ✗ No reconocido |

## ⚙️ Parámetros Configurables

En `reconocimiento.py`:
```python
TOTAL_PUNTOS = 30                    # Número de puntos faciales a capturar
UMBRAL_RECONOCIMIENTO = 0.15         # Umbral de distancia para reconocimiento
BASE_DIR = os.path.dirname(...)      # Directorio base del proyecto
CARPETA_DATOS = "base_rostros"       # Directorio para almacenar características
```

## 📁 Archivos Generados

- **`.npy` en `base_rostros/`**: Vectores de características normalizados (NumPy binary format)
- **`*_grafo.jpg`**: Imágenes con los puntos y conexiones visualizadas

## 🤝 Contribuciones

Las contribuciones son bienvenidas. Por favor:
1. Fork el repositorio
2. Crea una rama para tu feature (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

## 📄 Licencia

Este proyecto está disponible bajo licencia MIT.

## 👥 Autores

- **Equipo 5** - Proyecto WPLF
- Ed0se

## 📞 Soporte

Si encuentras problemas o tienes preguntas, por favor abre un issue en el repositorio.

## 💡 Notas Técnicas

- El sistema utiliza **alineación Procrustes** para normalizar grafos
- Los vectores se almacenan en formato NumPy (.npy) para eficiencia
- La similaridad se calcula usando **distancia Euclidiana**
- La captura manual de puntos permite flexibilidad pero requiere precisión del usuario

---

**Nota**: Este sistema es ideal para prototipos, demostraciones educativas y proyectos de investigación. Para aplicaciones de producción que requieren alta precisión y robustez, se recomienda utilizar métodos basados en deep learning como FaceNet, VGGFace2 o ArcFace.
