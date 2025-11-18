'''
Universidad del Valle de Guatemala | Aprendizaje por refuerzo - S21
Marco Carbajal (23025) / car23025@uvg.edu.gt

Script de prueba rápida para verificar que todo funciona
'''

import gymnasium as gym
import ale_py

# Registrar entornos de ALE
gym.register_envs(ale_py)

print("\nProbando entorno Galaxian...")

# Crear entorno
env = gym.make("ALE/Galaxian-v5")
obs, info = env.reset()

print(f"✓ Entorno creado exitosamente")
print(f"  Observación shape: {obs.shape}")
print(f"  Acciones disponibles: {env.action_space.n}")

# Probar un paso
action = env.action_space.sample()
obs, reward, done, truncated, info = env.step(action)

print(f"✓ Paso ejecutado exitosamente")
print(f"  Recompensa: {reward}")

env.close()

print("\n✓ Todo funciona correctamente")
