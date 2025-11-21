# Proyecto final [Galaxian] 🚀

**Desarrollado por:** Marco Carbajal (car23025@uvg.edu.gt)

> Universidad del Valle de Guatemala - Aprendizaje por refuerzo (Sección 21)

---

## Estructura del proyecto

```
proyecto_galaxian/
│
├── setup.py                     # Configuración automática del entorno
├── init_env.py                  # Inicialización de entornos ALE
├── test_environment.py          # Prueba rápida del entorno
├── requirements.txt             # Dependencias del proyecto
│
├── train_dqn.py                 # Entrenamiento con Deep Q-Network
├── train_a2c.py                 # Entrenamiento con Advantage Actor-Critic
├── experiment_architectures.py  # Experimentación con variantes de arquitectura
├── analyze.py                   # Análisis de rendimiento y comparación
├── play_final.py                # Evaluación final y generación de videos
│
├── utils.py                     # Utilidades compartidas (preprocesamiento, replay buffer)
├── policy_wrapper.py            # Carga de modelos entrenados
│
├── modelos/                     # Modelos guardados (.pth)
├── graficas/                    # Gráficas de entrenamiento y comparación
├── experimentos/                # Resultados de experimentos con arquitecturas
├── videos_entrenamiento/        # Videos periódicos durante entrenamiento
└── grabaciones/                 # Videos finales para entrega
```

---

## Metodología y proceso de desarrollo

### Fase 1: Configuración del entorno

El proyecto inició con la configuración del entorno de desarrollo utilizando `setup.py`, que automatizó:
- Instalación de dependencias (Gymnasium, ALE-Py, PyTorch, OpenCV)
- Registro de entornos de Arcade Learning Environment [ALE]
- Verificación de que el entorno Galaxian funcionara correctamente

```bash
python setup.py
python test_environment.py  # Verificación exitosa (se genera automáticamente)
```

### Fase 2: Entrenamiento comparativo inicial (DQN vs A2C)

Se implementaron dos de los algoritmos de aprendizaje por refuerzo aprendidos en el curso para comparar su efectividad en Galaxian:

#### 2.1 Deep Q-Network (DQN)
Implementación con las siguientes características:
- **Arquitectura:** 3 capas convolucionales + 2 capas fully connected
- **Replay buffer:** 100,000 experiencias
- **Epsilon-greedy:** Exploración que decae linealmente de 1.0 a 0.01
- **Target network:** Actualizada cada 1,000 pasos

**Hiperparámetros utilizados:**
```python
{
    'n_episodes': 2000,
    'learning_rate': 0.0001,
    'gamma': 0.99,
    'epsilon_decay': 1700,
    'batch_size': 32,
    'target_update_freq': 1000
}
```

#### 2.2 Advantage Actor-Critic (A2C)
Implementación caracterizada por:
- **Arquitectura dual:** Actor y Critic comparten capas convolucionales
- **N-step returns:** Cálculo de retornos con n=5 pasos
- **Entropía:** Coeficiente de 0.01 para fomentar exploración

**Hiperparámetros utilizados:**
```python
{
    'n_episodes': 2000,
    'learning_rate': 0.0005,
    'gamma': 0.99,
    'n_steps': 5,
    'entropy_coef': 0.01,
    'value_coef': 0.5
}
```

#### 2.3 Resultados de la comparación

Después de entrenar ambos métodos por 2,000 episodios, se utilizó `analyze.py` para generar una comparación:

```bash
python analyze.py  # Opción 6: Comparar métodos
```

**Conclusión:** DQN demostró ser superior, con una ventaja consistente en el promedio de recompensas de los últimos 100 episodios. Las gráficas comparativas en `graficas/comparacion_metodos.png` muestran claramente esta diferencia de desempeño.

### Fase 3: Experimentación con arquitecturas

Una vez seleccionado DQN como el método principal, se procedió a explorar variantes de arquitectura neuronal para optimizar el rendimiento.

#### 3.1 Arquitecturas evaluadas

Se implementó `experiment_architectures.py` para probar cuatro variantes:

1. **Original (Baseline):** 3 capas convolucionales + 2 capas fully connected
2. **Deeper:** 4 capas convolucionales + 3 capas connected (más profundidad)
3. **Wider:** 3 capas convolucionales + 2 capas connected con más filtros
4. **Dueling:** Arquitectura Dueling DQN con streams separados para valor y ventaja

Cada variante fue entrenada por 500 episodios para una comparación rápida:

```bash
python experiment_architectures.py
```

#### 3.2 Resultados de la experimentación

El análisis de resultados (disponible en `experimentos/comparacion/architectures_comparison.png`) reveló que:

- **La arquitectura baseline mantuvo el mejor desempeño**
- Las arquitecturas más complejas no proporcionaron mejoras significativas
- El balance entre capacidad y complejidad de la arquitectura original fue óptimo

**Decisión:** Se mantuvo la arquitectura original de 3 capas convolucionales y 2 capas fully connected para el entrenamiento final. Así pues, los 2,000 episodios de entrenamiento previos con DQN pudieron ser aprovechados para continuar entrenando al agente.

### Fase 4: Entrenamiento extensivo

Con la arquitectura y método seleccionados, se procedió al entrenamiento extensivo del agente.

#### 4.1 Configuración del entrenamiento continuo

Se modificó `train_dqn.py` para continuar desde el checkpoint existente:

```python
HIPERPARAMETROS = {
    'resume_training': True,
    'checkpoint_path': 'modelos/dqn_checkpoint.pth',
    'n_episodes': 7212,  # Extensión del entrenamiento
    # ... resto de hiperparámetros sin cambios
}
```

> Realmente el agente no fue entrenado desde el episodio 2,000 hasta el episodio 7,212 de corrido, se fue haciendo por partes. Sin embargo, esta es una simplificación del proceso. 

#### 4.2 Monitoreo del progreso

Durante el entrenamiento, se generaron automáticamente:
- **Checkpoints cada 250 episodios:** Permiten reanudar el entrenamiento
- **Videos de evaluación cada 100 episodios:** Muestran la evolución visual del agente
- **Gráficas de métricas:** Recompensas, loss, epsilon, estadísticas acumuladas

El sistema de checkpoints guardó:
- `dqn_ep250.pth`, `dqn_ep500.pth`, etc. - Checkpoints específicos
- `dqn_checkpoint.pth` - Último estado para continuar
- `dqn_best.pth` - **Mejor episodio del entrenamiento**
- `dqn_final.pth` - Estado final tras 7,212 episodios

#### 4.3 Observaciones del entrenamiento

El análisis de las gráficas de entrenamiento (`graficas/dqn_metrics.png`) reveló:
- **Convergencia gradual** hacia mejores políticas en los primeros 5,000 episodios
- **Máximo rendimiento** alcanzado cerca del episodio 6,500-7,000
- **Degradación en episodios finales:** La media móvil comenzó a descender después del episodio 7,000

Esta degradación sugiere posible *catastrophic forgetting* en las últimas etapas del entrenamiento. 

### Fase 5: Selección del modelo final

Para la evaluación final, se tomó una decisión crítica sobre qué modelo utilizar.

#### 5.1 Análisis de candidatos

Se evaluaron dos opciones principales:
- `dqn_final.pth` - Estado final tras 7,212 episodios
- `dqn_best.pth` - Episodio con mayor puntuación durante el entrenamiento

#### 5.2 Decisión fundamentada

**Modelo seleccionado:** `dqn_best.pth`

**Justificación**:
1. Representa el **mejor desempeño individual** alcanzado durante todo el entrenamiento
2. Corresponde a un episodio cercano al final, pero **antes de la degradación observada**
3. La caída en la media de los últimos episodios sugiere que el modelo final no era óptimo

#### 5.3 Generación de videos finales

Se configuró `play_final.py` con el modelo seleccionado:

```python
CONFIG = {
    'model_path': 'modelos/dqn_best.pth',
    'method': 'dqn',
    'n_episodes': 3,
    'output_dir': 'grabaciones',
}
```

Ejecución:
```bash
python play_final.py
```

Esto generó 3 videos de evaluación en `grabaciones/` con el formato requerido: `car23025_[timestamp]_[score].mp4`. La puntuación con la que competí (la más alta de los 3 videos generados) fue 9230 puntos. 

---

## Arquitectura del modelo final

### Red Neuronal Convolucional (CNN)

```
Entrada: Stack de 4 frames (4 × 84 × 84)
    ↓
Conv2D(4→32, kernel=8, stride=4) + ReLU
    ↓
Conv2D(32→64, kernel=4, stride=2) + ReLU
    ↓
Conv2D(64→64, kernel=3, stride=1) + ReLU
    ↓
Flatten
    ↓
Linear(3136→512) + ReLU
    ↓
Linear(512→6)  [Q-values para 6 acciones]
```

**Total de parámetros:** ~1.6M

### Preprocesamiento de frames

1. **Conversión a escala de grises:** Reducción de dimensionalidad
2. **Redimensionamiento:** 210×160×3 → 84×84×1
3. **Normalización:** Valores entre 0 y 1
4. **Frame stacking:** Stack de los últimos 4 frames para capturar movimiento

### Acciones disponibles en Galaxian

```
0 → NOOP       - Sin acción
1 → FIRE       - Disparar
2 → RIGHT      - Mover derecha
3 → LEFT       - Mover izquierda
4 → RIGHTFIRE  - Mover derecha + disparar
5 → LEFTFIRE   - Mover izquierda + disparar
```