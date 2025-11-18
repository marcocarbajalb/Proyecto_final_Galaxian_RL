"""
Universidad del Valle de Guatemala | Aprendizaje por refuerzo - S21
Marco Carbajal (23025) / car23025@uvg.edu.gt

Inicialización de entornos de ALE.
Este módulo asegura que los entornos de ALE estén registrados correctamente.
"""

import gymnasium as gym
import ale_py

# Registrar entornos de ALE automáticamente al importar este módulo
try:
    gym.register_envs(ale_py)
    _ALE_REGISTERED = True
except Exception as e:
    print(f"⚠ Advertencia: No se pudieron registrar entornos de ALE: {e}")
    print("  Asegúrate de tener ale-py instalado: pip install ale-py==0.8.1")
    _ALE_REGISTERED = False

def is_ale_available():
    """Verifica si los entornos de ALE están disponibles"""
    return _ALE_REGISTERED

def test_environment():
    """Prueba que el entorno Galaxian funcione"""
    if not _ALE_REGISTERED:
        print("✗ Los entornos de ALE no están registrados")
        return False
    
    try:
        env = gym.make("ALE/Galaxian-v5")
        obs, info = env.reset()
        env.close()
        print("✓ Entorno Galaxian funcional")
        return True
    except Exception as e:
        print(f"✗ Error al probar Galaxian: {e}")
        return False

# Verificación automática al importar (opcional, comentar si es molesto)
if __name__ == "__main__":
    print("Probando entornos de ALE...")
    test_environment()