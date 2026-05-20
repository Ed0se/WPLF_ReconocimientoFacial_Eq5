import os
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog

import cv2
import numpy as np


TOTAL_PUNTOS = 30
UMBRAL_RECONOCIMIENTO = 0.15
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CARPETA_DATOS = os.path.join(BASE_DIR, "base_rostros")

os.makedirs(CARPETA_DATOS, exist_ok=True)


conexiones = [
    (1, 2), (3, 1), (2, 4), (3, 9), (4, 10),
    (1, 14), (2, 15),
    (1, 20), (2, 22),
    (6, 5),
    (7, 8),
    (9, 19), (10, 23),
    (11, 14), (14, 15),
    (12, 13),
    (16, 17),
    (15, 18),
    (19, 24),
    (23, 28),
    (19, 20),
    (20, 21),
    (21, 22),
    (22, 23),
    (24, 25),
    (25, 27),
    (27, 29),
    (27, 28),
    (29, 30),
    (28, 30),
    (24, 30),
]


def cargar_imagen(ruta):
    img = cv2.imread(ruta)

    if img is None:
        messagebox.showerror("Error", "No se pudo cargar la imagen.")
        return None

    return cv2.resize(img, (500, 500))


def capturar_puntos(img, total_puntos=TOTAL_PUNTOS):
    puntos = []
    img_temp = img.copy()
    ventana = "Dibujar grafo facial"

    def click_event(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(puntos) < total_puntos:
            puntos.append((x, y))

            cv2.circle(img_temp, (x, y), 4, (0, 0, 255), -1)
            cv2.putText(
                img_temp,
                str(len(puntos)),
                (x + 5, y + 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 0, 0),
                1,
            )

            cv2.imshow(ventana, img_temp)

    cv2.imshow(ventana, img_temp)
    cv2.setMouseCallback(ventana, click_event)

    while len(puntos) < total_puntos:
        tecla = cv2.waitKey(1) & 0xFF

        if tecla == 27:
            cv2.destroyAllWindows()
            return None

    cv2.destroyAllWindows()
    return puntos


def normalizar_puntos(puntos):
    P = np.array(puntos, dtype=np.float64)
    centro = np.mean(P, axis=0)
    P -= centro

    norma = np.linalg.norm(P)

    if norma == 0:
        raise ValueError("No se pueden normalizar puntos con norma cero.")

    return P / norma


def alinear_procrustes(A, B):
    U, _, Vt = np.linalg.svd(A.T @ B)
    R = U @ Vt
    return B @ R


def vector_grafo(puntos):
    distancias = []

    for (i, j) in conexiones:
        d = np.linalg.norm(puntos[i - 1] - puntos[j - 1])
        distancias.append(d)

    return np.array(distancias)


def dibujar_grafo(img, puntos):
    img_copy = img.copy()
    puntos = np.array(puntos)

    for i in range(len(puntos)):
        x = int(puntos[i][0])
        y = int(puntos[i][1])
        cv2.circle(img_copy, (x, y), 3, (0, 0, 255), -1)

    for (i, j) in conexiones:
        pt1 = puntos[i - 1]
        pt2 = puntos[j - 1]
        cv2.line(
            img_copy,
            (int(pt1[0]), int(pt1[1])),
            (int(pt2[0]), int(pt2[1])),
            (0, 255, 0),
            1,
        )

    return img_copy


def nombre_archivo_seguro(nombre):
    caracteres_validos = []

    for caracter in nombre.strip():
        if caracter.isalnum() or caracter in (" ", "_", "-"):
            caracteres_validos.append(caracter)

    return "".join(caracteres_validos).strip().replace(" ", "_")


def obtener_vector_desde_imagen(ruta_imagen, ruta_grafo=None):
    img = cargar_imagen(ruta_imagen)

    if img is None:
        return None

    puntos = capturar_puntos(img)

    if puntos is None:
        messagebox.showinfo("Proceso cancelado", "La captura de puntos fue cancelada.")
        return None

    img_grafo = dibujar_grafo(img, puntos)

    if ruta_grafo:
        cv2.imwrite(ruta_grafo, img_grafo)

    P = normalizar_puntos(puntos)
    return vector_grafo(P)


def porcentaje_y_nivel_similitud(distancia):
    similitud = 100 - (distancia / (UMBRAL_RECONOCIMIENTO * 2)) * 100
    similitud = max(0, min(100, similitud))

    if distancia <= UMBRAL_RECONOCIMIENTO * 0.5:
        nivel = "Alta"
    elif distancia <= UMBRAL_RECONOCIMIENTO:
        nivel = "Media"
    elif distancia <= UMBRAL_RECONOCIMIENTO * 1.5:
        nivel = "Baja"
    else:
        nivel = "Muy baja"

    return similitud, nivel


def comparar_vectores(vector_a, vector_b):
    distancia = np.linalg.norm(vector_a - vector_b)
    similitud, nivel = porcentaje_y_nivel_similitud(distancia)

    return {
        "distancia": distancia,
        "similitud": similitud,
        "nivel": nivel,
        "reconocido": distancia <= UMBRAL_RECONOCIMIENTO,
    }


def registrar_persona(nombre, ruta_imagen):
    nombre_seguro = nombre_archivo_seguro(nombre)

    if not nombre_seguro:
        messagebox.showerror("Error", "El nombre no es valido.")
        return False

    ruta_grafo = os.path.join(BASE_DIR, f"{nombre_seguro}_grafo.jpg")
    vector = obtener_vector_desde_imagen(ruta_imagen, ruta_grafo)

    if vector is None:
        return False

    np.save(os.path.join(CARPETA_DATOS, f"{nombre_seguro}.npy"), vector)
    messagebox.showinfo(
        "Registro completo",
        f"{nombre_seguro} registrado correctamente.\nGrafo guardado en: {ruta_grafo}",
    )
    return True


def reconocer_persona(ruta_imagen):
    vector_prueba = obtener_vector_desde_imagen(ruta_imagen)

    if vector_prueba is None:
        return None

    resultados = []

    for archivo in os.listdir(CARPETA_DATOS):
        if not archivo.endswith(".npy"):
            continue

        nombre = archivo.replace(".npy", "")
        vector_guardado = np.load(os.path.join(CARPETA_DATOS, archivo))
        comparacion = comparar_vectores(vector_guardado, vector_prueba)

        resultados.append({
            "nombre": nombre,
            **comparacion,
        })

    resultados.sort(key=lambda item: item["distancia"])
    return resultados


def seleccionar_imagen():
    return filedialog.askopenfilename(
        title="Selecciona una imagen",
        filetypes=[
            ("Imagenes", "*.jpg *.jpeg *.png *.bmp"),
            ("Todos los archivos", "*.*"),
        ],
    )


def accion_registrar():
    nombre = simpledialog.askstring("Registrar rostro", "Nombre de la persona:")

    if not nombre:
        return

    ruta_imagen = seleccionar_imagen()

    if not ruta_imagen:
        return

    registrar_persona(nombre, ruta_imagen)


def construir_texto_resultados(resultados):
    if not resultados:
        return "No hay rostros registrados para comparar."

    mejor = resultados[0]
    lineas = ["Resultado de comparacion", ""]

    if mejor["reconocido"]:
        lineas.append(f"Persona reconocida: {mejor['nombre']}")
    else:
        lineas.append("No se reconoce la persona.")
        lineas.append(f"Rostro mas parecido: {mejor['nombre']}")

    lineas.extend([
        f"Distancia: {mejor['distancia']:.4f}",
        f"Similitud: {mejor['similitud']:.2f}%",
        f"Nivel de similitud: {mejor['nivel']}",
        "",
        "Comparacion contra registros:",
    ])

    for resultado in resultados:
        lineas.append(
            f"- {resultado['nombre']}: distancia {resultado['distancia']:.4f}, "
            f"similitud {resultado['similitud']:.2f}%, nivel {resultado['nivel']}"
        )

    return "\n".join(lineas)


def mostrar_resultados(resultados):
    ventana = tk.Toplevel()
    ventana.title("Resultado de comparacion")
    ventana.geometry("560x360")
    ventana.resizable(False, False)

    texto = tk.Text(ventana, wrap="word", padx=14, pady=14, font=("Segoe UI", 10))
    texto.insert("1.0", construir_texto_resultados(resultados))
    texto.config(state="disabled")
    texto.pack(fill="both", expand=True)

    boton_cerrar = tk.Button(ventana, text="Cerrar", command=ventana.destroy)
    boton_cerrar.pack(pady=10)


def accion_comparar():
    if not any(archivo.endswith(".npy") for archivo in os.listdir(CARPETA_DATOS)):
        messagebox.showwarning(
            "Sin registros",
            "Primero registra al menos un rostro para poder comparar.",
        )
        return

    ruta_imagen = seleccionar_imagen()

    if not ruta_imagen:
        return

    resultados = reconocer_persona(ruta_imagen)

    if resultados is not None:
        mostrar_resultados(resultados)


def iniciar_ventana():
    root = tk.Tk()
    root.title("Reconocimiento facial por grafos")
    root.geometry("420x260")
    root.resizable(False, False)

    contenedor = tk.Frame(root, padx=30, pady=28)
    contenedor.pack(fill="both", expand=True)

    titulo = tk.Label(
        contenedor,
        text="Reconocimiento facial",
        font=("Segoe UI", 18, "bold"),
    )
    titulo.pack(pady=(0, 8))

    subtitulo = tk.Label(
        contenedor,
        text="Selecciona una accion para iniciar.",
        font=("Segoe UI", 10),
    )
    subtitulo.pack(pady=(0, 22))

    boton_registrar = tk.Button(
        contenedor,
        text="Registrar Rostro",
        font=("Segoe UI", 12),
        height=2,
        command=accion_registrar,
    )
    boton_registrar.pack(fill="x", pady=6)

    boton_comparar = tk.Button(
        contenedor,
        text="Comparar Rostro",
        font=("Segoe UI", 12),
        height=2,
        command=accion_comparar,
    )
    boton_comparar.pack(fill="x", pady=6)

    root.mainloop()


if __name__ == "__main__":
    iniciar_ventana()
