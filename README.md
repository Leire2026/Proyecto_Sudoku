# Proyecto Sudoku: Resolución Automática de Sudokus a partir de Imágenes

## Objetivo

El objetivo del proyecto ha sido construir un sistema capaz de recibir una imagen de un Sudoku, detectar el tablero, leer los números presentes en las celdas, construir una matriz 9x9 y generar una propuesta de solución.

El proyecto se ha dividido en varias fases independientes para facilitar el desarrollo, la validación y la integración final en una aplicación Streamlit.


## Arquitectura general del sistema

La solución final se ha organizado como un pipeline compuesto por varios módulos:

```text 
Imagen de entrada 
↓
Modelo YOLO para detectar el tablero
↓
Recorte del Sudoku
↓
Segmentación en 81 celdas
↓
Modelo CNN para reconocer números
↓
Matriz 9x9 inicial
↓
Modelo predictivo de resolución
↓
Matriz solución predicha
↓
Aplicación Streamlit
```

Esta división permitió trabajar cada problema por separado: detección de objetos, segmentación de imagen, clasificación de dígitos y resolución del Sudoku.


## Conclusiones

El proyecto permitió construir un flujo completo desde una imagen de Sudoku hasta una solución predicha. Se entrenaron tres tipos de modelos o componentes principales:

1. Un detector YOLO para localizar el tablero.
2. Una CNN para reconocer dígitos en las celdas.
3. Un modelo predictivo para generar una solución a partir de la matriz inicial.

Aunque los resultados de reconocimiento de números y resolución no fueron perfectos, se consiguió implementar un sistema funcional de extremo a extremo y se identificaron claramente las limitaciones técnicas del enfoque.

La principal conclusión es que la detección del tablero funcionó correctamente, mientras que las fases de lectura de dígitos y resolución requieren más datos o un enfoque híbrido para alcanzar resultados robustos.