"""
Universidad del Valle de Guatemala | Aprendizaje por refuerzo - S21
Marco Carbajal (23025) / car23025@uvg.edu.gt

Utilidades para el proyecto de Galaxian
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import cv2
import torch
from pathlib import Path

# ==================== PREPROCESAMIENTO ====================

class FramePreprocessor:
    """Preprocesa frames del entorno Galaxian"""
    
    def __init__(self, img_size=(84, 84), grayscale=True):
        self.img_size = img_size
        self.grayscale = grayscale
    
    def process(self, frame):
        """
        Preprocesa un frame: escala de grises, resize y normalización
        
        Args:
            frame: np.array de shape (210, 160, 3)
        Returns:
            frame procesado de shape (84, 84) normalizado entre 0 y 1
        """
        if self.grayscale:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        
        frame = cv2.resize(frame, self.img_size, interpolation=cv2.INTER_AREA)
        frame = frame.astype(np.float32) / 255.0
        
        return frame

class FrameStacker:
    """Apila los últimos N frames para dar contexto temporal"""
    
    def __init__(self, n_frames=4):
        self.n_frames = n_frames
        self.frames = deque(maxlen=n_frames)
    
    def reset(self):
        """Limpia el buffer de frames"""
        self.frames.clear()
    
    def stack(self, frame):
        """
        Agrega un frame y retorna el stack completo
        
        Args:
            frame: frame preprocesado de shape (84, 84)
        Returns:
            stack de shape (4, 84, 84)
        """
        if len(self.frames) == 0:
            # Primera vez, llenar con el mismo frame
            for _ in range(self.n_frames):
                self.frames.append(frame)
        else:
            self.frames.append(frame)
        
        return np.stack(self.frames, axis=0)

# ==================== REPLAY BUFFER (para DQN) ====================

class ReplayBuffer:
    """Buffer de experiencias para DQN"""
    
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        """Agrega una experiencia al buffer"""
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        """Muestrea un batch aleatorio"""
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, next_states, dones = zip(*[self.buffer[i] for i in indices])
        
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(next_states),
            np.array(dones, dtype=np.float32)
        )
    
    def __len__(self):
        return len(self.buffer)

# ==================== VISUALIZACIÓN ====================

class TrainingPlotter:
    """Genera y guarda gráficas de entrenamiento"""
    
    def __init__(self, save_dir="graficas"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
    
    def plot_rewards(self, rewards, method_name, window=100):
        """
        Grafica recompensas por episodio con media móvil
        
        Args:
            rewards: lista de recompensas por episodio
            method_name: nombre del método (DQN, A2C, etc)
            window: tamaño de ventana para media móvil
        """
        plt.figure(figsize=(12, 6))
        
        # Recompensas individuales
        plt.plot(rewards, alpha=0.3, label='Recompensa por episodio')
        
        # Media móvil
        if len(rewards) >= window:
            moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
            plt.plot(range(window-1, len(rewards)), moving_avg, 
                    linewidth=2, label=f'Media móvil ({window} eps)')
        
        plt.xlabel('Episodio')
        plt.ylabel('Recompensa')
        plt.title(f'Entrenamiento - {method_name}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        filename = self.save_dir / f"{method_name.lower()}_rewards.png"
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Gráfica guardada en: {filename}")
    
    def plot_training_metrics(self, rewards, losses, epsilons, steps, method_name, window=100):
        """
        Genera múltiples gráficas de métricas de entrenamiento
        
        Args:
            rewards: lista de recompensas
            losses: lista de losses
            epsilons: lista de epsilons (solo DQN)
            steps: lista de pasos por episodio
            method_name: nombre del método
            window: ventana para media móvil
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Métricas de Entrenamiento - {method_name}', fontsize=16)
        
        # 1. Recompensas
        ax1 = axes[0, 0]
        ax1.plot(rewards, alpha=0.3, color='blue', label='Recompensa')
        if len(rewards) >= window:
            moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
            ax1.plot(range(window-1, len(rewards)), moving_avg, 
                    color='darkblue', linewidth=2, label=f'Media móvil ({window})')
        ax1.set_xlabel('Episodio')
        ax1.set_ylabel('Recompensa')
        ax1.set_title('Recompensas por Episodio')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Loss
        ax2 = axes[0, 1]
        if losses and len(losses) > 0:
            ax2.plot(losses, alpha=0.3, color='red', label='Loss')
            if len(losses) >= window:
                moving_avg_loss = np.convolve(losses, np.ones(window)/window, mode='valid')
                ax2.plot(range(window-1, len(losses)), moving_avg_loss, 
                        color='darkred', linewidth=2, label=f'Media móvil ({window})')
            ax2.set_xlabel('Episodio')
            ax2.set_ylabel('Loss')
            ax2.set_title('Loss de Entrenamiento')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        # 3. Epsilon (solo para DQN) o Steps
        ax3 = axes[1, 0]
        if epsilons and len(epsilons) > 0 and method_name == 'DQN':
            ax3.plot(epsilons, color='green', linewidth=2)
            ax3.set_xlabel('Episodio')
            ax3.set_ylabel('Epsilon')
            ax3.set_title('Decay de Exploración (Epsilon)')
            ax3.grid(True, alpha=0.3)
        elif steps and len(steps) > 0:
            ax3.plot(steps, alpha=0.3, color='purple', label='Steps')
            if len(steps) >= window:
                moving_avg_steps = np.convolve(steps, np.ones(window)/window, mode='valid')
                ax3.plot(range(window-1, len(steps)), moving_avg_steps, 
                        color='indigo', linewidth=2, label=f'Media móvil ({window})')
            ax3.set_xlabel('Episodio')
            ax3.set_ylabel('Pasos')
            ax3.set_title('Pasos por Episodio')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        # 4. Estadísticas acumuladas
        ax4 = axes[1, 1]
        if len(rewards) > 0:
            # Calcular estadísticas acumulativas
            cumulative_mean = [np.mean(rewards[:i+1]) for i in range(len(rewards))]
            cumulative_max = [np.max(rewards[:i+1]) for i in range(len(rewards))]
            
            ax4.plot(cumulative_mean, color='blue', linewidth=2, label='Media acumulada')
            ax4.plot(cumulative_max, color='green', linewidth=2, label='Máximo acumulado')
            ax4.set_xlabel('Episodio')
            ax4.set_ylabel('Recompensa')
            ax4.set_title('Estadísticas Acumuladas')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        filename = self.save_dir / f"{method_name.lower()}_metrics.png"
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Gráfica de métricas guardada en: {filename}")
    
    def plot_comparison(self, results_dict, window=100):
        """
        Compara múltiples métodos en una sola gráfica
        
        Args:
            results_dict: diccionario {nombre_metodo: lista_rewards}
            window: tamaño de ventana para media móvil
        """
        plt.figure(figsize=(12, 6))
        
        for method_name, rewards in results_dict.items():
            if len(rewards) >= window:
                moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
                plt.plot(range(window-1, len(rewards)), moving_avg, 
                        linewidth=2, label=method_name)
        
        plt.xlabel('Episodio')
        plt.ylabel('Recompensa (media móvil)')
        plt.title('Comparación de Métodos')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        filename = self.save_dir / "comparacion_metodos.png"
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Gráfica de comparación guardada en: {filename}")

# ==================== GUARDADO Y CARGA ====================

def save_checkpoint(model, optimizer, episode, rewards, filepath):
    """
    Guarda un checkpoint del entrenamiento
    
    Args:
        model: modelo de red neuronal
        optimizer: optimizador
        episode: número de episodio actual
        rewards: lista de recompensas acumuladas
        filepath: ruta donde guardar
    """
    checkpoint = {
        'episode': episode,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'rewards': rewards
    }
    torch.save(checkpoint, filepath)
    print(f"Checkpoint guardado en: {filepath}")

def load_checkpoint(model, optimizer, filepath):
    """
    Carga un checkpoint del entrenamiento
    
    Args:
        model: modelo de red neuronal
        optimizer: optimizador
        filepath: ruta del checkpoint
    Returns:
        episode, rewards
    """
    checkpoint = torch.load(filepath, weights_only=False)  # weights_only=False para cargar todo el checkpoint
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    episode = checkpoint['episode']
    rewards = checkpoint['rewards']
    
    print(f"Checkpoint cargado desde: {filepath}")
    print(f"Episodio: {episode}, Episodios entrenados: {len(rewards)}")
    
    return episode, rewards