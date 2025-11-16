# Proyecto final [Galaxian] 🚀

### Programador
Marco Carbajal (23025)
car23025@uvg.edu.gt

> Universidad del Valle de Guatemala - Aprendizaje por refuerzo (Sección 21)

## Estructura del proyecto

```
proyecto_galaxian/
│
├── setup.py               # Setup automático (Lo primero que se debe ejecutar)
├── init_env.py            # Inicialización de entornos ALE
├── test_environment.py    # Prueba rápida (generado por `setup.py`)
├── requirements.txt       # Dependencias
│
├── train_dqn.py          # Entrenamiento DQN
├── train_a2c.py          # Entrenamiento A2C
├── analyze.py            # Análisis de rendimiento
├── play_final.py         # Evaluación final
│
├── utils.py              # Utilidades compartidas
├── policy_wrapper.py     # Cargar modelos entrenados
│
├── modelos/              # Modelos guardados
├── graficas/             # Gráficas de entrenamiento
├── videos_entrenamiento/ # Videos periódicos
└── grabaciones/          # Videos finales para entrega
```

## Instalación

**IMPORTANTE: Ejecuta el setup automático primero**

```bash
# Setup automático
python setup.py

# Esto instalará todo en el orden correcto y verificará que funcione
```

El script de setup hará:
1. Crear la estructura de directorios (si todavía no están creados)
2. Instalar dependencias en el orden correcto
3. Registrar entornos de ALE automáticamente
4. Verificar que todo funcione
5. Crear un script de prueba

### Verificación post-instalación

```bash
# Prueba rápida
python test_environment.py

# Si aparece "✓ ¡Todo funciona correctamente!", está listo
```

## Guía de uso

### Fase 1: Entrenamiento inicial (Comparación)

1. **Entrenar DQN** (1000-2000 episodios para comparación):
```bash
python train_dqn.py
```

2. **Entrenar A2C** (1000-2000 episodios para comparación):
```bash
python train_a2c.py
```

3. **Comparar ambos métodos**:
```bash
python compare_methods.py
```

Este script mostrará cuál método funciona mejor y generará:
- `graficas/comparacion_metodos.png` - Comparación visual
- `graficas/comparacion.json` - Estadísticas detalladas

### Fase 2: Entrenamiento final

Una vez elegido el mejor método, continúa con el entrenamiento:

**Si resultó ser DQN:**
```bash
# Editar train_dqn.py:
# - Cambiar 'resume_training': True
# - Aumentar 'n_episodes': 5000 (o más)

python train_dqn.py
```

**Si resultó ser A2C:**
```bash
# Editar train_a2c.py:
# - Cambiar 'resume_training': True
# - Aumentar 'n_episodes': 5000 (o más)

python train_a2c.py
```

### Fase 3: Evaluación final (para la entrega)

```bash
# Editar play_final.py:
# - Configurar 'model_path' con el mejor modelo
# - Configurar 'method' ('dqn' o 'a2c')

python play_final.py
```

Esto generará 3 videos en `grabaciones/` con el formato requerido.

## Configuración de hiperparámetros

### DQN (train_dqn.py)

```python
HIPERPARAMETROS = {
    'n_episodes': 5000,          # Episodios totales
    'learning_rate': 0.00025,    # Tasa de aprendizaje
    'gamma': 0.99,               # Factor de descuento
    'epsilon_start': 1.0,        # Exploración inicial
    'epsilon_end': 0.01,         # Exploración final
    'buffer_size': 100000,       # Tamaño del replay buffer
    'batch_size': 32,            # Tamaño del batch
    'target_update_freq': 1000,  # Actualizar target network
}
```

### A2C (train_a2c.py)

```python
HIPERPARAMETROS = {
    'n_episodes': 5000,          # Episodios totales
    'learning_rate': 0.0007,     # Tasa de aprendizaje
    'gamma': 0.99,               # Factor de descuento
    'n_steps': 5,                # N-step returns
    'entropy_coef': 0.01,        # Coeficiente de entropía
    'value_coef': 0.5,           # Coeficiente del critic
}
```

## Monitoreo del entrenamiento

Durante el entrenamiento, se muestra algo como:
```
Ep  100 | Reward:    340 | Avg100:  285.5 | Eps: 0.900 | Loss: 0.1234
💾 Checkpoint guardado: modelos/dqn_ep100.pth
```

- **Reward**: Puntuación del episodio actual
- **Avg100**: Promedio de últimos 100 episodios
- **Eps**: Epsilon actual (solo DQN)
- **Loss**: Loss promedio del episodio

### Checkpoints automáticos (cada 250 episodios):

- `modelos/[method]_ep250.pth`, `[method]_ep500.pth`, etc. - Checkpoints específicos
- `modelos/[method]_checkpoint.pth` - Último checkpoint (para resumir)
- `modelos/[method]_best.pth` - Mejor modelo hasta ahora
- `modelos/[method]_final.pth` - Modelo final del entrenamiento

### Videos de evaluación (cada 100 episodios):

- `videos_entrenamiento/ep100/` - video de evaluación
- `videos_entrenamiento/ep200/` - video de evaluación
- etc.

Estos videos permiten **ver visualmente cómo mejora el agente** a lo largo del tiempo.

### Gráficas generadas automáticamente:

**Gráfica completa de métricas** (`[method]_metrics.png`):
- Recompensas por episodio con media móvil
- Loss de entrenamiento
- Decay de epsilon (DQN) o pasos por episodio
- Estadísticas acumuladas (media y máximo)

**Estadísticas JSON** (`[method]_stats.json`):
```json
{
  "total_episodes": 1000,
  "best_reward": 1300.0,
  "final_avg_100": 750.0,
  "final_epsilon": 0.01
}
```

## Sistema de Checkpoints

Los checkpoints se guardan automáticamente cada 250 episodios:

**Nomenclatura de archivos**:
- `[method]_ep100.pth` - Checkpoint específico del episodio 100
- `[method]_ep200.pth` - Checkpoint específico del episodio 200
- `[method]_checkpoint.pth` - Último checkpoint (para `resume_training`)
- `[method]_best.pth` - Mejor modelo hasta el momento
- `[method]_final.pth` - Modelo final

**Contenido de cada checkpoint**:
- Estado del modelo (`model_state_dict`)
- Estado del optimizador (`optimizer_state_dict`)
- Número de episodio (`episode`)
- Historial de recompensas (`rewards`)
- Historial de losses (`losses`)
- Historial de epsilons (`epsilons` - solo DQN)
- Mejor recompensa (`best_reward`)

Para continuar un entrenamiento interrumpido:
```python
'resume_training': True  # Cambiar en HIPERPARAMETROS
```

## Sistema de videos

### Videos durante el entrenamiento
Cada 100 episodios, se graban un videos de evaluación:

```
videos_entrenamiento/
  ├── ep100/
  │   ├── dqn_ep100_score510.mp4
  │   └── a2c_ep100_score470.mp4
  ├── ep200/
  ├── ep.../
  └── ep1000/
```

### Analizar progreso
```bash
python analyze.py
```

**Opciones del menú**:
1. **Resumen general** - Estado completo del proyecto
2. **Analizar checkpoints (DQN)** - Tabla de progreso
3. **Analizar checkpoints (A2C)** - Tabla de progreso
4. **Analizar videos (DQN)** - Progreso visual por episodio
5. **Analizar videos (A2C)** - Progreso visual por episodio
6. **Comparar métodos** - Decisión de cuál usar

## Acciones disponibles en galaxian
```
0 -> NOOP       - No hacer nada
1 -> FIRE       - Disparar
2 -> RIGHT      - Mover derecha
3 -> LEFT       - Mover izquierda
4 -> RIGHTFIRE  - Mover derecha + disparar
5 -> LEFTFIRE   - Mover izquierda + disparar
```

## Workflow simplificado

```bash
# 1. SETUP (una vez)
python setup.py
python test_environment.py

# 2. ENTRENAMIENTO INICIAL (Comparación)
python train_dqn.py      # Entrenar ~2000 episodios
python train_a2c.py      # Entrenar ~2000 episodios

# 3. ANÁLISIS Y DECISIÓN
python analyze.py        # Opción 6: Comparar métodos

# 4. ENTRENAMIENTO FINAL
# Editar train_[metodo_elegido].py: resume_training=True, n_episodes=5000 (o más)
python train_[metodo_elegido].py      # (o el método elegido)

# 5. MONITOREO (durante entrenamiento)
python analyze.py        # Ver progreso periódicamente

# 6. EVALUACIÓN FINAL
# Editar play_final.py: model_path='modelos/[metodo_elegido]_best.pth'
python play_final.py     # Generar videos para entrega
```