import pygame
import dask
from dask import delayed
import numpy as np
import random

# Configuración de la ventana
WIDTH, HEIGHT = 800, 600
BLACK = (0, 0, 0)

# Inicializar pygame
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Copo de Nieve de Koch - IA Distribuida con Dask")
clock = pygame.time.Clock()

# Función para calcular las líneas del fractal (Koch)
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

# Función para dibujar un copo de nieve
def draw_snowflake(screen, points, color):
    for i in range(len(points) - 1):
        pygame.draw.line(screen, color, points[i], points[i + 1], 1)

# Función para generar copos de nieve aleatorios
def generate_random_snowflake():
    depth = random.randint(2, 3)  # Limitamos profundidad para evitar bloqueos
    size = random.randint(100, 250)  # Tamaño aleatorio
    x_offset = random.randint(50, WIDTH - 50)
    y_offset = random.randint(50, HEIGHT - 50)

    # Color aleatorio
    color = (random.randint(100, 255), random.randint(100, 255), random.randint(100, 255))

    # Puntos del triángulo base
    p1 = (x_offset - size//2, y_offset + size//3)
    p2 = (x_offset + size//2, y_offset + size//3)
    p3 = (x_offset, y_offset - 2 * size//3)

    # Aplicamos Dask a las llamadas principales
    segment_1 = delayed(koch_curve)(p1, p2, depth)
    segment_2 = delayed(koch_curve)(p2, p3, depth)
    segment_3 = delayed(koch_curve)(p3, p1, depth)

    return segment_1, segment_2, segment_3, color

# Número máximo de copos en pantalla
num_snowflakes = 10  # Se irá llenando hasta este número máximo
snowflakes = []

# Loop de pygame
running = True
frame_counter = 0  # Contador para controlar la generación

while running:
    screen.fill(BLACK)

    # Cada 30 frames, añadimos un nuevo copo hasta el límite
    if frame_counter % 30 == 0 and len(snowflakes) < num_snowflakes:
        new_snowflake = generate_random_snowflake()
        snowflakes.append(new_snowflake)

    # Computamos en paralelo solo los copos actuales
    computed_snowflakes = dask.compute(*[s[:3] for s in snowflakes])

    # Dibujar todos los copos generados
    for i, computed_segments in enumerate(computed_snowflakes):
        flat_points = [p for segment in computed_segments for p in segment]
        draw_snowflake(screen, flat_points, snowflakes[i][3])  # El color está en la posición [3]

    pygame.display.flip()
    
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    frame_counter += 1  # Aumentamos el contador de frames
    clock.tick(30)  # Control de FPS

pygame.quit()
