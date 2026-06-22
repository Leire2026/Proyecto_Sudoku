import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
from tensorflow.keras.models import load_model


def extraer_celdas(sudoku_crop):
    sudoku_resized = cv2.resize(sudoku_crop, (450, 450))
    cell_size = 50
    margin = 5
    cells = []

    for fila in range(9):
        row = []
        for col in range(9):
            x1 = col * cell_size
            y1 = fila * cell_size
            x2 = x1 + cell_size
            y2 = y1 + cell_size

            cell = sudoku_resized[y1:y2, x1:x2]
            cell = cell[margin:-margin, margin:-margin]
            row.append(cell)

        cells.append(row)

    return cells


def preparar_img_base(cell):
    gray = cv2.cvtColor(cell, cv2.COLOR_RGB2GRAY)

    _, thresh = cv2.threshold(
        gray,
        150,
        255,
        cv2.THRESH_BINARY_INV
    )

    thresh = cv2.resize(thresh, (28, 28))
    thresh = thresh.astype("float32") / 255.0

    return thresh


def predecir_celda(cell, modelo):
    img = preparar_img_base(cell)

    if np.sum(img) < 10:
        return 0

    img = img.reshape(1, 28, 28, 1)

    pred = modelo.predict(img, verbose=0)
    numero = np.argmax(pred)

    return int(numero)


def generar_matriz(cells, modelo_digitos):
    matriz = []

    for fila in range(9):
        row = []

        for col in range(9):
            numero = predecir_celda(
                cells[fila][col],
                modelo_digitos
            )
            row.append(numero)

        matriz.append(row)

    return matriz


def resolver_con_modelo(matriz, modelo_solver):
    sudoku = np.array(matriz).flatten()
    sudoku = sudoku.astype("float32") / 9.0

    pred = modelo_solver.predict(
        sudoku.reshape(1, 81),
        verbose=0
    )

    solucion = np.argmax(pred, axis=-1) + 1
    solucion = solucion.reshape(9, 9)

    return solucion


st.title("🔢 Solver de Sudoku desde imagen")

st.write(
    "SUBE UNA IMAGEN DE UN SUDOKU Y EL SISTEMA LO RESOLVERÁ....O MÁS BIEN NO 😎"
)


@st.cache_resource
def cargar_modelos():
    modelo_yolo = YOLO("models/best.pt")

    modelo_digitos = load_model(
        "models/modelo_digitos_sudoku.keras"
    )

    modelo_solver = load_model(
        "models/modelo_solver_sudoku_v2.keras"
    )

    return modelo_yolo, modelo_digitos, modelo_solver


modelo_yolo, modelo_digitos, modelo_solver = cargar_modelos()

st.success("✅ Modelos cargados correctamente")


uploaded_file = st.file_uploader(
    "Selecciona una imagen",
    type=["jpg", "jpeg", "png"]
)


if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    img = np.array(image)

    st.image(
        img,
        caption="Imagen cargada",
        use_container_width=True
    )

    st.success("✅ Imagen cargada correctamente")

    if st.button("Detectar tablero, leer números y resolver"):
        results = modelo_yolo(img)
        boxes = results[0].boxes

        if len(boxes) == 0:
            st.error("❌ No se ha detectado ningún sudoku.")

        else:
            box = boxes.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = map(int, box)

            sudoku_crop = img[y1:y2, x1:x2]

            st.image(
                sudoku_crop,
                caption="Sudoku recortado",
                use_container_width=True
            )

            st.success(
                "✅ Tablero detectado y recortado correctamente"
            )

            cells = extraer_celdas(sudoku_crop)

            matriz = generar_matriz(
                cells,
                modelo_digitos
            )

            matriz_np = np.array(matriz)

            st.subheader("Matriz detectada")
            st.code(matriz_np, language="python")

            solucion_predicha = resolver_con_modelo(
                matriz,
                modelo_solver
            )

            st.subheader("Solución según el modelo... y ahora es cuando te ríes un rato 🤣")
            st.code(solucion_predicha, language="python")