import pygame
import tensorflow as tf
import numpy as np
import random
import threading
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Reshape, InputLayer
from tensorflow.keras.losses import MeanSquaredError

# Configuración de la ventana en pygame
WIDTH, HEIGHT = 800, 600
BLACK = (0, 0, 0)

# Inicializar pygame
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Copo de Nieve de Koch - Redes Neuronales Colaborativas")
clock = pygame.time.Clock()

# 🔹 **Escalar coordenadas de [-1,1] a [0, WIDTH] y [0, HEIGHT]**
def scale_to_screen(x, y, offset_x=0, offset_y=0, scale=1.0):
    x_scaled = int((x + 1) * (WIDTH / 2) * scale + offset_x)
    y_scaled = int((y + 1) * (HEIGHT / 2) * scale + offset_y)
    return x_scaled, y_scaled

# 🔹 **Función para generar el fractal de Koch**
def koch_curve(p1, p2, depth):
    if depth == 0:
        return [p1, p2]

    p1, p2 = np.array(p1), np.array(p2)
    v = (p2 - p1) / 3
    pA = tuple(p1 + v)
    pB = tuple(p2 - v)

    angle = np.radians(60)
    rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    pC = tuple(pA + np.dot(rotation_matrix, v))

    return (
        koch_curve(p1, pA, depth - 1) +
        koch_curve(pA, pC, depth - 1) +
        koch_curve(pC, pB, depth - 1) +
        koch_curve(pB, p2, depth - 1)
    )

# 🔹 **Generar datos de entrenamiento con transformaciones**
def generate_koch_data(num_samples=1000, depth=3):
    X, Y = [], []
    max_points = 100
    
    for _ in range(num_samples):
        # Desplazamiento y transformación aleatoria
        offset_x = random.uniform(-0.5, 0.5)
        offset_y = random.uniform(-0.5, 0.5)
        scale = random.uniform(0.5, 1.5)

        p1 = (-0.5 + offset_x, -0.5 + offset_y)
        p2 = (0.5 + offset_x, -0.5 + offset_y)
        p3 = (0.0 + offset_x, 0.5 + offset_y)

        # Generar el fractal de Koch
        points = koch_curve(p1, p2, depth) + koch_curve(p2, p3, depth) + koch_curve(p3, p1, depth)
        
        points_array = np.array(points[:max_points])
        if points_array.shape[0] < max_points:
            padding = np.zeros((max_points - points_array.shape[0], 2))
            points_array = np.vstack((points_array, padding))
        
        X.append([p1, p2, p3])  # Entrada: triángulo base con variaciones
        Y.append(points_array)  # Salida: Puntos del fractal transformado

    return np.array(X), np.array(Y)

# 🔹 **Construcción del modelo único compartido**
def build_shared_koch_network():
    model = Sequential([
        InputLayer(input_shape=(3, 2)),  
        Flatten(),
        Dense(256, activation='relu'),
        Dense(512, activation='relu'),
        Dense(1024, activation='relu'),
        Dense(100 * 2, activation='linear'), 
        Reshape((100, 2))  
    ])
    model.compile(optimizer='adam', loss=MeanSquaredError())
    return model

# 🔹 **Usamos un solo modelo compartido**
shared_model = build_shared_koch_network()

# 🔹 **Cargar datos de entrenamiento**
X_train, Y_train = generate_koch_data()

# 🔹 **Entrenar el modelo único**
print(f"Entrenando modelo compartido...")
shared_model.fit(X_train, Y_train, epochs=15, batch_size=32, verbose=1)

# 🔹 **Generar fractales con desplazamiento y escalado**
def generate_koch_with_nn(model, model_id):
    offset_x = random.randint(100, 700)
    offset_y = random.randint(100, 500)
    scale = random.uniform(0.5, 1.5)

    p1 = (-0.5, -0.5)
    p2 = (0.5, -0.5)
    p3 = (0.0, 0.5)
    
    X_input = np.array([[p1, p2, p3]])
    Y_pred = model.predict(X_input)[0]

    # Ordenar por coordenada X para asegurar coherencia en la estructura
    Y_pred = sorted(Y_pred, key=lambda p: p[0])

    # Escalar los puntos generados
    scaled_points = [scale_to_screen(Y_pred[i][0], Y_pred[i][1], offset_x, offset_y, scale) for i in range(len(Y_pred))]

    print(f"[Red {model_id}] Puntos generados escalados: {scaled_points[:5]}")

    return scaled_points

# 🔹 **Loop de pygame con aprendizaje colaborativo**
running = True
while running:
    screen.fill(BLACK)

    for i in range(3):  # Tres fractales generados por el mismo modelo
        color = (random.randint(100, 255), random.randint(100, 255), random.randint(100, 255))
        points = generate_koch_with_nn(shared_model, i)
        
        if len(points) > 1:
            for j in range(len(points) - 1):
                pygame.draw.line(screen, color, points[j], points[j + 1], 1)
        else:
            print(f"❌ [Red {i}] No se generaron suficientes puntos para dibujar")

    pygame.display.flip()  
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    clock.tick(1)

pygame.quit()
