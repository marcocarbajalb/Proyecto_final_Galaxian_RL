"""
Universidad del Valle de Guatemala | Aprendizaje por refuerzo - S21
Marco Carbajal (23025) / car23025@uvg.edu.gt

Análisis y gestión unificada del proyecto
"""

import torch
import json
import numpy as np
from pathlib import Path
import re
from utils import TrainingPlotter

# ==================== ANÁLISIS DE CHECKPOINTS ====================

def analyze_checkpoints(method='dqn'):
    """Analiza todos los checkpoints de un método"""
    model_dir = Path('modelos')
    pattern = f'{method}_ep*.pth'
    checkpoints = sorted(model_dir.glob(pattern))
    
    if not checkpoints:
        print(f"✗ No hay checkpoints para {method.upper()}")
        return None
    
    print(f"\n{'='*60}")
    print(f"ANÁLISIS DE CHECKPOINTS - {method.upper()}")
    print(f"{'='*60}\n")
    
    data = []
    for ckpt_path in checkpoints:
        try:
            ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
            episode = ckpt.get('episode', 0)
            rewards = ckpt.get('rewards', [])
            
            if rewards:
                avg_100 = np.mean(rewards[-100:]) if len(rewards) >= 100 else np.mean(rewards)
                max_reward = max(rewards)
                
                data.append({
                    'checkpoint': ckpt_path.name,
                    'episode': episode + 1,
                    'total_episodes': len(rewards),
                    'avg_100': avg_100,
                    'max_reward': max_reward
                })
        except:
            pass
    
    if not data:
        print("✗ No se pudieron cargar datos de los checkpoints")
        return None
    
    # Ordenar por episodio ascendente
    data.sort(key=lambda x: x['episode'])
    
    # Mostrar tabla
    print(f"{'Checkpoint':<20} {'Episodio':>10} {'Avg100':>10} {'Max':>10}")
    print("-" * 60)
    for d in data:
        print(f"{d['checkpoint']:<20} {d['episode']:>10} {d['avg_100']:>10.1f} {d['max_reward']:>10.0f}")
    
    # Mejor checkpoint
    best = max(data, key=lambda x: x['avg_100'])
    print(f"\n🏆 Mejor checkpoint: {best['checkpoint']}")
    print(f"   Episodio: {best['episode']}, Avg100: {best['avg_100']:.1f}\n")
    
    return data

# ==================== ANÁLISIS DE VIDEOS ====================

def analyze_videos(method='dqn'):
    """Analiza videos de entrenamiento"""
    video_dir = Path('videos_entrenamiento')
    
    if not video_dir.exists():
        print(f"✗ No hay videos aún (se generan cada 100 episodios)")
        return None
    
    pattern = f'{method}_ep*_score*.mp4'
    videos = sorted(video_dir.glob(f'**/{pattern}'))
    
    if not videos:
        print(f"✗ No hay videos para {method.upper()}")
        return None
    
    print(f"\n{'='*60}")
    print(f"ANÁLISIS DE VIDEOS - {method.upper()}")
    print(f"{'='*60}\n")
    
    # Extraer información de TODOS los videos primero
    data = []
    for video in videos:
        match = re.search(rf'{method}_ep(\d+)_score(\d+)', video.name)
        if match:
            ep = int(match.group(1))
            score = int(match.group(2))
            data.append({'episode': ep, 'score': score})
    
    # Ordenar por episodio ascendente
    data.sort(key=lambda x: x['episode'])
    
    # Ahora sí imprimir en orden
    print(f"{'Episodio':>10} {'Score':>10}")
    print("-" * 60)
    
    for d in data:
        print(f"{d['episode']:>10} {d['score']:>10.0f}")
    
    if len(data) >= 2:
        improvement = data[-1]['score'] - data[0]['score']
        print(f"\n📈 Mejora: {data[0]['score']:.1f} → {data[-1]['score']:.1f} ({improvement:+.1f})")
    
    print()
    return data

# ==================== COMPARACIÓN DE MÉTODOS ====================

def compare_methods():
    """Compara DQN vs A2C"""
    print(f"\n{'='*60}")
    print("COMPARACIÓN DQN vs A2C")
    print(f"{'='*60}\n")
    
    results = {}
    
    # Analizar ambos métodos
    for method in ['dqn', 'a2c']:
        checkpoint_path = Path('modelos') / f'{method}_checkpoint.pth'
        
        if not checkpoint_path.exists():
            print(f"ERROR: No hay checkpoint para {method.upper()}")
            continue
        
        try:
            ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            rewards = ckpt.get('rewards', [])
            
            if rewards:
                results[method.upper()] = {
                    'episodes': len(rewards),
                    'avg_total': np.mean(rewards),
                    'avg_100': np.mean(rewards[-100:]) if len(rewards) >= 100 else np.mean(rewards),
                    'max': max(rewards),
                    'rewards': rewards
                }
        except:
            print(f"✗ Error cargando {method.upper()}")
    
    if not results:
        print("✗ No hay datos para comparar")
        return
    
    # Mostrar tabla comparativa
    print(f"{'Método':<10} {'Episodios':>10} {'Avg Total':>12} {'Avg100':>12} {'Max':>10}")
    print("-" * 60)
    
    for method, data in results.items():
        print(f"{method:<10} {data['episodes']:>10} {data['avg_total']:>12.1f} "
              f"{data['avg_100']:>12.1f} {data['max']:>10.0f}")
    
    # Determinar ganador
    if len(results) == 2:
        methods = list(results.keys())
        scores = [results[m]['avg_100'] for m in methods]
        
        winner_idx = np.argmax(scores)
        winner = methods[winner_idx]
        difference = scores[winner_idx] - scores[1 - winner_idx]
        
        print(f"\n🏆 MEJOR MÉTODO: {winner}")
        print(f"   Ventaja en Avg100: +{difference:.1f}")
        
        # Generar gráfica comparativa
        plotter = TrainingPlotter('graficas')
        plot_data = {m: results[m]['rewards'] for m in results.keys()}
        plotter.plot_comparison(plot_data)
        
        # Guardar análisis
        output = {
            'winner': winner,
            'difference': float(difference),
            'results': {m: {k: float(v) if isinstance(v, (int, float, np.number)) else v 
                           for k, v in data.items() if k != 'rewards'} 
                       for m, data in results.items()}
        }
        
        output_path = Path('graficas') / 'comparison.json'
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"\n💾 Análisis guardado en: {output_path}")

# ==================== RESUMEN GENERAL ====================

def show_summary():
    """Muestra un resumen general del proyecto"""
    print(f"\n{'='*60}")
    print("RESUMEN DEL PROYECTO")
    print(f"{'='*60}\n")
    
    # Checkpoints
    print("📁 MODELOS:")
    for method in ['dqn', 'a2c']:
        checkpoints = list(Path('modelos').glob(f'{method}_ep*.pth'))
        best = Path('modelos') / f'{method}_best.pth'
        final = Path('modelos') / f'{method}_final.pth'
        
        if checkpoints or best.exists() or final.exists():
            print(f"  {method.upper()}:")
            print(f"    Checkpoints: {len(checkpoints)}")
            print(f"    Mejor modelo: {'✓' if best.exists() else '✗'}")
            print(f"    Modelo final: {'✓' if final.exists() else '✗'}")
    
    # Gráficas
    print("\n📊 GRÁFICAS:")
    graphics = list(Path('graficas').glob('*.png'))
    if graphics:
        for g in graphics:
            print(f"    {g.name}")
    else:
        print("    (No hay gráficas aún)")
    
    # Videos
    print("\n🎥 VIDEOS DE ENTRENAMIENTO:")
    video_dirs = list(Path('videos_entrenamiento').glob('ep*'))
    if video_dirs:
        for vdir in sorted(video_dirs):
            count = len(list(vdir.glob('*.mp4')))
            print(f"    {vdir.name}: {count} videos")
    else:
        print("    (No hay videos aún)")
    
    # Videos finales
    print("\n🎬 VIDEOS FINALES (para entrega):")
    final_videos = list(Path('grabaciones').glob('*.mp4'))
    if final_videos:
        for v in sorted(final_videos)[-5:]:  # Últimos 5
            print(f"    {v.name}")
    else:
        print("    (No hay videos finales aún)")
    
    print(f"\n{'='*60}\n")

# ==================== MENÚ PRINCIPAL ====================

def main():
    """Menú interactivo"""
    while True:
        print(f"\n{'='*60}")
        print("ANÁLISIS Y GESTIÓN DEL PROYECTO - GALAXIAN RL 🚀")
        print(f"{'='*60}")
        print("\n1. Resumen general del proyecto")
        print("2. Analizar checkpoints (DQN)")
        print("3. Analizar checkpoints (A2C)")
        print("4. Analizar videos (DQN)")
        print("5. Analizar videos (A2C)")
        print("6. Comparar métodos")
        print("0. Salir")
        print(f"\n{'='*60}")
        
        choice = input("\nSelecciona una opción: ").strip()
        
        if choice == '1':
            show_summary()
        elif choice == '2':
            analyze_checkpoints('dqn')
        elif choice == '3':
            analyze_checkpoints('a2c')
        elif choice == '4':
            analyze_videos('dqn')
        elif choice == '5':
            analyze_videos('a2c')
        elif choice == '6':
            compare_methods()
        elif choice == '0':
            print("\nHas abandonado el programa exitosamente.\n")
            break
        else:
            print("\n✗ Opción inválida")
        
        input("\nPresiona 'Enter' para continuar...")

if __name__ == "__main__":
    main()