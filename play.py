'''
Universidad del Valle de Guatemala | Aprendizaje por refuerzo - S21
Marco Carbajal (23025) / car23025@uvg.edu.gt

Evaluación final (Grabar episodios)
'''

# Inicializar entornos de ALE (debe ser lo primero)
import init_env

import ale_py
import os
import math
import random
import gymnasium as gym
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Tuple
import shutil
import warnings
import time

# Importar las políticas entrenadas
from policy_wrapper import load_policy

warnings.filterwarnings('ignore', category=UserWarning, module='gymnasium.wrappers.rendering')

# ==================== CONFIGURACIÓN ====================

CONFIG = {
    # Seleccionar mejor modelo
    'model_path': 'modelos/dqn_best.pth', # Cambiar al mejor modelo
    'method': 'dqn',  # 'dqn' o 'a2c'
    
    # Parámetros de grabación
    'n_episodes': 3,  # Número de episodios a grabar
    'output_dir': 'grabaciones',
}

# ==================== POLÍTICAS BASE ====================

class Policy:
    """Clase base para las políticas."""
    def select_action(self, observacion: np.ndarray) -> int:
        raise NotImplementedError("No se ha implementado select_action")
    
    def reset(self):
        pass

class RandomPolicy(Policy):
    """Política aleatoria."""
    def __init__(self, action_space):
        self.action_space = action_space
        
    def select_action(self, observacion: np.ndarray) -> int:
        return self.action_space.sample()
     
class PolicyFire(Policy):
    """Política que siempre dispara."""
    def __init__(self, action_space):
        self.action_space = action_space
        
    def select_action(self, observacion: np.ndarray) -> int:
        acciones_disparo = [1,4,4,5,5]
        return random.choice(acciones_disparo)

# ==================== FUNCIÓN DE GRABACIÓN ====================

def record_episode(policy: Policy, output_dir: str = "grabaciones", 
                  verbose: bool = True) -> Dict[str, Any]:
    """
    Graba un episodio completo usando la política proporcionada.
    
    Argumento:
        policy: Política a utilizar (debe tener método select_action)
        output_dir: Directorio para guardar videos
        verbose: Mostrar información durante la ejecución
    Retorno:
        Diccionario con información del episodio
    """

    env_name = "ALE/Galaxian-v5"
    render_mode = "rgb_array"
    max_steps = math.inf

    # Crear directorio temporal único para esta grabación
    timestamp_temp = datetime.now().strftime("%Y%m%d%H%M_%f")
    temp_dir = Path(output_dir) / f"temp_{timestamp_temp}"
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    # Crear el entorno con capacidad de renderizado
    env = gym.make(env_name, render_mode=render_mode)
    
    # Wrapper para grabar video
    env = gym.wrappers.RecordVideo(
        env,
        video_folder=str(temp_dir),
        episode_trigger=lambda x: True,
        name_prefix="temp")
    
    # Reiniciar política y entorno
    policy.reset()
    observation, info = env.reset()
    
    # Variables para tracking
    total_reward = 0
    steps = 0
    done = False
    truncated = False
    
    timestamp_start = datetime.now()
    
    if verbose:
        print(f"Iniciando episodio con {policy.__class__.__name__}...")
    
    # Ejecutar episodio
    while not (done or truncated) and steps < max_steps:
        action = policy.select_action(observation)
        observation, reward, done, truncated, info = env.step(action)
        
        total_reward += reward
        steps += 1
        
        if verbose and steps % 100 == 0:
            print(f"  Paso {steps}: Puntuación actual = {total_reward}")
    
    env.close()
    time.sleep(0.5)
    
    timestamp_end = datetime.now()
    
    # Buscar el video generado
    video_files = list(temp_dir.glob("*.mp4"))
    if not video_files:
        raise FileNotFoundError("No se generó el archivo de video")
    
    temp_video = video_files[0]
    
    # Crear nombre final del video
    timestamp_str = timestamp_start.strftime("%Y%m%d%H%M")
    score_str = f"{int(total_reward)}"
    final_filename = f"car23025_{timestamp_str}_{score_str}.mp4"
    final_path = Path(output_dir) / final_filename
    
    final_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(temp_video), str(final_path))
    
    # Limpiar directorio temporal
    try:
        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                shutil.rmtree(temp_dir)
                break
            except (PermissionError, OSError) as e:
                if attempt < max_attempts - 1:
                    time.sleep(0.3)
                else:
                    raise e
    except Exception as e:
        if verbose:
            print(f"   Advertencia: No se pudo eliminar directorio temporal: {e}")
    
    # Información del episodio
    episode_info = {
        "score": total_reward,
        "steps": steps,
        "video_path": str(final_path),
        "timestamp": timestamp_start,
        "execution_time_seconds": (timestamp_end - timestamp_start).total_seconds(),
        "terminated": done,
        "truncated": truncated}
    
    if verbose:
        print(f"\nEpisodio completado")
        print(f"   Puntuación final: {total_reward}")
        print(f"   Pasos totales: {steps}")
        print(f"   Video guardado en: {final_path}")
    
    return episode_info

# ==================== MAIN ====================

def main():
    """Función principal para la entrega."""

    print("=" * 60)
    print("EVALUACIÓN FINAL - Galaxian 🚀")
    print("Marco Carbajal (23025) / car23025@uvg.edu.gt")
    print("=" * 60)
    
    # Cargar la política entrenada
    print("\nCargando modelo entrenado...")
    try:
        policy = load_policy(
            CONFIG['model_path'], 
            method=CONFIG['method']
        )
        print("✓ Modelo cargado exitosamente\n")
    except FileNotFoundError:
        print(f"✗ No se encontró el modelo: {CONFIG['model_path']}")
        print("  Usando política aleatoria como respaldo...\n")
        env = gym.make("ALE/Galaxian-v5")
        policy = RandomPolicy(env.action_space)
        env.close()
    
    # Grabar episodios
    all_scores = []
    
    print("=" * 60)
    print(f"Grabando {CONFIG['n_episodes']} episodios")
    print("=" * 60 + "\n")
    
    for i in range(CONFIG['n_episodes']):
        print(f"\n--- EPISODIO {i+1}/{CONFIG['n_episodes']} ---")
        episode_info = record_episode(policy, CONFIG['output_dir'])
        all_scores.append(episode_info['score'])
    
    # Resumen final
    print("\n" + "=" * 60)
    print("RESUMEN DE EVALUACIÓN")
    print("=" * 60)
    print(f"Episodios grabados: {len(all_scores)}")
    print(f"Puntuaciones: {[int(s) for s in all_scores]}")
    print(f"Mejor puntuación: {max(all_scores):.0f}")
    print(f"Puntuación promedio: {np.mean(all_scores):.2f}")
    print(f"Desviación estándar: {np.std(all_scores):.2f}")
    print("=" * 60)
    
    print(f"\nVideos guardados en: {CONFIG['output_dir']}/")
    print("\n✓ Evaluación completada exitosamente")

if __name__ == "__main__":
    main()