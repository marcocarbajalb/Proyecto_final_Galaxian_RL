"""
Universidad del Valle de Guatemala | Aprendizaje por refuerzo - S21
Marco Carbajal (23025) / car23025@uvg.edu.gt

Script de instalación y configuración del proyecto
"""

import subprocess
import sys
from pathlib import Path

def print_header(text):
    """Imprime un encabezado formateado"""
    print("\n" + "="*60)
    print(text)
    print("="*60 + "\n")

def create_directories():
    """Crea la estructura de directorios"""
    print_header("CREANDO ESTRUCTURA DE DIRECTORIOS")
    
    dirs = ['modelos', 'graficas', 'grabaciones']
    
    for dir_name in dirs:
        Path(dir_name).mkdir(exist_ok=True)
        print(f"✓ Directorio '{dir_name}/' creado")
    
    print("\n✓ Estructura de directorios completa")

def install_dependencies():
    """Instala las dependencias del proyecto en el orden correcto"""
    print_header("INSTALANDO DEPENDENCIAS")
    
    print("Esto puede tomar varios minutos...\n")
    
    # Paso 1: Desinstalar versiones previas que puedan causar conflictos
    print("Paso 1: Limpiando instalaciones previas...")
    packages_to_remove = ['ale-py', 'gymnasium', 'autorom']
    for package in packages_to_remove:
        try:
            subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", package], 
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except:
            pass
    print("✓ Limpieza completada")
    
    # Paso 2: Instalar en orden específico para evitar conflictos
    install_order = [
        ("NumPy", "numpy>=1.24.0"),
        ("Gymnasium", "gymnasium>=0.29.0"),
        ("Gymnasium[other] (MoviePy para videos)", "gymnasium[other]"),  # Para grabación de videos
        ("Shimmy", "shimmy>=2.0.0"),  # Requerido por gymnasium reciente
        ("ALE-Py", "ale-py>=0.10.0"),  # Versión más reciente disponible
        ("AutoROM (con licencia)", "autorom[accept-rom-license]"),
        ("PyTorch", "torch>=2.0.0"),
        ("TorchVision", "torchvision>=0.15.0"),
        ("OpenCV", "opencv-python>=4.8.0"),
        ("Matplotlib", "matplotlib>=3.7.0"),
    ]
    
    for name, package in install_order:
        print(f"\nInstalando {name}...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"✓ {name} instalado")
        except subprocess.CalledProcessError as e:
            print(f"✗ Error instalando {name}: {e}")
            return False
    
    print("\n✓ Todas las dependencias instaladas")
    return True

def register_ale_environments():
    """Registra los entornos de ALE con Gymnasium"""
    print_header("REGISTRANDO ENTORNOS DE ALE")
    
    try:
        import gymnasium as gym
        import ale_py
        
        # Registrar explícitamente los entornos
        gym.register_envs(ale_py)
        
        print("✓ Entornos de ALE registrados correctamente")
        return True
    except Exception as e:
        print(f"✗ Error registrando entornos: {e}")
        return False

def verify_installation():
    """Verifica que todo esté instalado correctamente"""
    print_header("VERIFICANDO INSTALACIÓN")
    
    all_good = True
    
    # Verificar Gymnasium
    try:
        import gymnasium
        print(f"✓ Gymnasium version: {gymnasium.__version__}")
    except ImportError:
        print("✗ Gymnasium no instalado correctamente")
        all_good = False
    
    # Verificar ALE-Py
    try:
        import ale_py
        print(f"✓ ALE-Py instalado")
    except ImportError:
        print("✗ ALE-Py no instalado correctamente")
        all_good = False
    
    # Verificar PyTorch
    try:
        import torch
        print(f"✓ PyTorch version: {torch.__version__}")
        
        if torch.cuda.is_available():
            print(f"✓ CUDA disponible: {torch.cuda.get_device_name(0)}")
        else:
            print("⚠ CUDA no disponible - se usará CPU")
    except ImportError:
        print("✗ PyTorch no instalado correctamente")
        all_good = False
    
    # Verificar OpenCV
    try:
        import cv2
        print(f"✓ OpenCV instalado")
    except ImportError:
        print("✗ OpenCV no instalado correctamente")
        all_good = False
    
    # Verificar Matplotlib
    try:
        import matplotlib
        print(f"✓ Matplotlib version: {matplotlib.__version__}")
    except ImportError:
        print("✗ Matplotlib no instalado correctamente")
        all_good = False
    
    # Verificar que Galaxian esté disponible (CRÍTICO)
    try:
        import gymnasium as gym
        import ale_py
        
        # Registrar entornos
        gym.register_envs(ale_py)
        
        # Intentar crear el entorno
        env = gym.make("ALE/Galaxian-v5")
        env.close()
        print("✓ Entorno Galaxian disponible y funcional")
    except Exception as e:
        print(f"✗ Error al crear entorno Galaxian: {e}")
        print("\n⚠ IMPORTANTE: El entorno Galaxian no funciona correctamente.")
        all_good = False
    
    if all_good:
        print("\n✓ Todas las verificaciones pasaron")
    else:
        print("\n⚠ Algunas verificaciones fallaron. Revisar mensajes de error.")
    
    return all_good

def create_test_script():
    """Crea un script de prueba rápida"""
    print_header("CREANDO SCRIPT DE PRUEBA")
    
    test_script = """'''
Universidad del Valle de Guatemala | Aprendizaje por refuerzo - S21
Marco Carbajal (23025) / car23025@uvg.edu.gt

Script de prueba rápida para verificar que todo funciona
'''

import gymnasium as gym
import ale_py

# Registrar entornos de ALE
gym.register_envs(ale_py)

print("\\nProbando entorno Galaxian...")

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

print("\\n✓ Todo funciona correctamente")
"""
    
    with open('test_environment.py', 'w', encoding='utf-8') as f:
        f.write(test_script)
    
    print("✓ Script de prueba creado: test_environment.py")

def show_next_steps():
    """Muestra los siguientes pasos"""
    print_header("INSTALACIÓN COMPLETA")
    
    print("""
✓ El proyecto está configurado y listo para usar.

PRUEBA RÁPIDA (recomendado):
    python test_environment.py
""")

def main():
    """Función principal"""
    print_header("SETUP DEL PROYECTO - GALAXIAN RL 🚀")
    print("Marco Carbajal (23025) / car23025@uvg.edu.gt")
    
    try:
        # Paso 1: Crear directorios
        create_directories()
        
        # Paso 2: Preguntar si instalar dependencias
        print("\n¿Deseas instalar las dependencias? (S/n): ", end="")
        response = input().strip().lower()
        
        if response == 'n':
            print("\nOmitiendo instalación de dependencias.")
            print("Asegúrate de instalarlas manualmente más tarde.")
        else:
            # Instalar dependencias
            if not install_dependencies():
                print("\n⚠ Hubo errores durante la instalación.")
                print("Intenta ejecutar el setup nuevamente o instala manualmente.")
                return
            
            # Registrar entornos de ALE
            if not register_ale_environments():
                print("\n⚠ No se pudieron registrar los entornos de ALE.")
                print("Intenta ejecutar:")
                print('  pip install "autorom[accept-rom-license]"')
                return
            
            # Verificar instalación
            if not verify_installation():
                print("\n⚠ La verificación encontró problemas.")
                print("\nPuedes intentar:")
                print("1. Ejecutar este script nuevamente: python setup.py")
                print("2. Instalar manualmente las dependencias faltantes")
                return
        
        # Crear script de prueba
        create_test_script()
        
        # Mostrar siguientes pasos
        show_next_steps()
        
    except KeyboardInterrupt:
        print("\n\n⚠ Instalación cancelada por el usuario.")
    except Exception as e:
        print(f"\n✗ Error durante la instalación: {e}")
        print("\nIntenta instalar manualmente las dependencias.")

if __name__ == "__main__":
    main()