"""
Universidad del Valle de Guatemala | Aprendizaje por refuerzo - S21
Marco Carbajal (23025) / car23025@uvg.edu.gt

Entrenamiento con Deep Q-Network (DQN)
"""

# Inicializar entornos de ALE (debe ser lo primero)
import init_env

import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pathlib import Path
from utils import FramePreprocessor, FrameStacker, ReplayBuffer, TrainingPlotter, save_checkpoint, load_checkpoint

# ==================== HIPERPARÁMETROS ====================

HIPERPARAMETROS = {
    # Entorno
    'env_name': 'ALE/Galaxian-v5',
    
    # Entrenamiento
    'n_episodes': 7010,              # Total
    'max_steps': 10000,              # Pasos máximos por episodio
    'batch_size': 32,                # Tamaño del batch
    'buffer_size': 100000,           # Tamaño del replay buffer
    'learning_rate': 0.0001,         # Learning rate (reducido para estabilidad)
    'gamma': 0.99,                   # Factor de descuento
    
    # Exploración (epsilon-greedy)
    'epsilon_start': 1.0,            # Epsilon inicial (solo para entrenamientos nuevos)
    'epsilon_end': 0.01,             # Epsilon final
    'epsilon_decay': 1700,           # Decay suave: llega a 0.01 en episodio ~1700
    
    # DQN específico
    'target_update_freq': 1000,      # Actualizar target network cada N pasos
    'min_buffer_size': 10000,        # Mínimo de experiencias antes de entrenar
    'train_freq': 4,                 # Entrenar cada N pasos
    
    # Preprocesamiento
    'n_frames': 4,                   # Frames a apilar
    'img_size': (84, 84),            # Tamaño de imagen procesada
    
    # Guardado
    'save_freq': 500,                # Guardar checkpoint cada N episodios
    'video_freq': 250,               # Grabar video cada N episodios
    'model_dir': 'modelos',
    'graphics_dir': 'graficas',
    'video_dir': 'videos_entrenamiento',
    
    # Continuar entrenamiento
    'resume_training': True,         # True para continuar, False para empezar desde 0
    'checkpoint_path': 'modelos/dqn_checkpoint.pth'
}

# ==================== RED NEURONAL ====================

class DQN(nn.Module):
    """Red neuronal convolucional para DQN"""
    
    def __init__(self, n_frames, n_actions):
        super(DQN, self).__init__()
        
        # Capas convolucionales
        self.conv = nn.Sequential(
            nn.Conv2d(n_frames, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        
        # Calcular tamaño después de convoluciones
        conv_out_size = self._get_conv_out((n_frames, 84, 84))
        
        # Capas fully connected
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )
    
    def _get_conv_out(self, shape):
        """Calcula el tamaño de salida de las convoluciones"""
        with torch.no_grad():
            x = torch.zeros(1, *shape)
            x = self.conv(x)
            return int(np.prod(x.size()))
    
    def forward(self, x):
        """Forward pass"""
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# ==================== AGENTE DQN ====================

class DQNAgent:
    """Agente que usa DQN para aprender"""
    
    def __init__(self, n_actions, device, hparams):
        self.n_actions = n_actions
        self.device = device
        self.hparams = hparams
        
        # Redes (policy y target)
        self.policy_net = DQN(hparams['n_frames'], n_actions).to(device)
        self.target_net = DQN(hparams['n_frames'], n_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        # Optimizador
        self.optimizer = optim.Adam(self.policy_net.parameters(), 
                                    lr=hparams['learning_rate'])
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(hparams['buffer_size'])
        
        # Contador de pasos
        self.steps = 0
        
        # Epsilon para exploración
        self.epsilon = hparams['epsilon_start']
    
    def select_action(self, state, training=True):
        """Selecciona acción usando epsilon-greedy"""
        if training and np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_tensor)
            return q_values.argmax().item()
    
    def update_epsilon(self, episode):
        """Actualiza epsilon con decay lineal"""
        self.epsilon = max(
            self.hparams['epsilon_end'],
            self.hparams['epsilon_start'] - 
            (self.hparams['epsilon_start'] - self.hparams['epsilon_end']) * 
            episode / self.hparams['epsilon_decay']
        )
    
    def train_step(self):
        """Realiza un paso de entrenamiento"""
        if len(self.replay_buffer) < self.hparams['min_buffer_size']:
            return None
        
        # Muestrear batch
        states, actions, rewards, next_states, dones = \
            self.replay_buffer.sample(self.hparams['batch_size'])
        
        # Convertir a tensores
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Q-values actuales
        current_q = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze()
        
        # Q-values siguientes (usando target network)
        with torch.no_grad():
            next_q = self.target_net(next_states).max(1)[0]
            target_q = rewards + (1 - dones) * self.hparams['gamma'] * next_q
        
        # Loss y backpropagation
        loss = nn.MSELoss()(current_q, target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)  # Clip más agresivo (1.0 en vez de 10)
        self.optimizer.step()
        
        return loss.item()
    
    def update_target_network(self):
        """Copia pesos de policy network a target network"""
        self.target_net.load_state_dict(self.policy_net.state_dict())

# ==================== FUNCIONES AUXILIARES ====================

def record_evaluation_videos(agent, preprocessor, stacker, hparams, episode_num, n_actions):
    """
    Graba UN video de evaluación durante el entrenamiento
    Usa la misma lógica que play.py original
    """
    from datetime import datetime
    from pathlib import Path
    import shutil
    import time
    from policy_wrapper import create_live_policy
    
    # Crear política compatible con play.py desde el agente actual
    policy = create_live_policy(agent, preprocessor, stacker, method='dqn')
    
    # Directorio para este episodio
    video_dir = Path(hparams['video_dir']) / f'ep{episode_num}'
    video_dir.mkdir(parents=True, exist_ok=True)
    
    # Crear directorio temporal único
    timestamp_temp = datetime.now().strftime("%Y%m%d%H%M_%f")
    temp_dir = video_dir / f'temp_{timestamp_temp}'
    temp_dir.mkdir(exist_ok=True)
    
    # Crear entorno con grabación
    env = gym.make(hparams['env_name'], render_mode='rgb_array')
    env = gym.wrappers.RecordVideo(
        env,
        video_folder=str(temp_dir),
        episode_trigger=lambda x: True,
        name_prefix="eval"
    )
    
    # Reiniciar política y entorno
    policy.reset()
    observation, _ = env.reset()
    
    total_reward = 0
    done = False
    truncated = False
    
    # Ejecutar episodio usando la política
    while not (done or truncated):
        action = policy.select_action(observation)  # Usa el preprocesamiento interno
        observation, reward, done, truncated, _ = env.step(action)
        total_reward += reward
    
    env.close()
    time.sleep(0.5)
    
    # Mover video al directorio final con nombre descriptivo
    video_files = list(temp_dir.glob("*.mp4"))
    if video_files:
        final_name = f'dqn_ep{episode_num}_score{int(total_reward)}.mp4'
        final_path = video_dir / final_name
        shutil.move(str(video_files[0]), str(final_path))
    
    # Limpiar temporal
    try:
        shutil.rmtree(temp_dir)
    except:
        pass
    
    print(f"  📹 Video: Score = {total_reward:.0f}")

# ==================== ENTRENAMIENTO ====================

def train_dqn(hparams):
    """Función principal de entrenamiento"""
    
    # Configuración
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}")
    
    # Crear directorios
    Path(hparams['model_dir']).mkdir(exist_ok=True)
    Path(hparams['graphics_dir']).mkdir(exist_ok=True)
    Path(hparams['video_dir']).mkdir(exist_ok=True)
    
    # Crear entorno
    env = gym.make(hparams['env_name'])
    n_actions = env.action_space.n
    
    # Preprocesamiento
    preprocessor = FramePreprocessor(img_size=hparams['img_size'])
    stacker = FrameStacker(n_frames=hparams['n_frames'])
    
    # Crear agente
    agent = DQNAgent(n_actions, device, hparams)
    
    # Para gráficas y análisis
    plotter = TrainingPlotter(hparams['graphics_dir'])
    
    # Métricas de entrenamiento
    all_rewards = []
    all_losses = []
    all_epsilons = []
    all_steps_per_episode = []
    best_reward = -float('inf')
    
    # Cargar checkpoint si se desea continuar
    start_episode = 0
    
    if hparams['resume_training']:
        try:
            start_episode, all_rewards = load_checkpoint(
                agent.policy_net, 
                agent.optimizer, 
                hparams['checkpoint_path']
            )
            agent.target_net.load_state_dict(agent.policy_net.state_dict())
            
            # Cargar también otras métricas si existen
            checkpoint = torch.load(hparams['checkpoint_path'], weights_only=False)
            if 'losses' in checkpoint:
                all_losses = checkpoint['losses']
            if 'epsilons' in checkpoint:
                all_epsilons = checkpoint['epsilons']
                # IMPORTANTE: Recalcular epsilon basado en el episodio real
                # El epsilon guardado puede estar mal si hubo cambios en epsilon_decay
                agent.update_epsilon(start_episode)
                print(f"Epsilon recalculado para episodio {start_episode}: {agent.epsilon:.4f}")
            if 'steps' in checkpoint:
                all_steps_per_episode = checkpoint['steps']
            if 'best_reward' in checkpoint:
                best_reward = checkpoint['best_reward']
                
        except FileNotFoundError:
            print("No se encontró checkpoint, empezando desde cero")
    
    print("\n" + "="*60)
    print(f"ENTRENAMIENTO DQN - Galaxian")
    print("="*60)
    print(f"Episodios: {start_episode} -> {hparams['n_episodes']}")
    print(f"Learning rate: {hparams['learning_rate']}")
    print(f"Gamma: {hparams['gamma']}")
    print(f"Buffer size: {hparams['buffer_size']}")
    print("="*60 + "\n")
    
    # Loop de entrenamiento
    for episode in range(start_episode, hparams['n_episodes']):
        # Reset
        obs, _ = env.reset()
        obs = preprocessor.process(obs)
        stacker.reset()
        state = stacker.stack(obs)
        
        episode_reward = 0
        episode_loss = []
        
        for step in range(hparams['max_steps']):
            # Seleccionar y ejecutar acción
            action = agent.select_action(state, training=True)
            next_obs, reward, done, truncated, _ = env.step(action)
            
            # Preprocesar
            next_obs = preprocessor.process(next_obs)
            next_state = stacker.stack(next_obs)
            
            # Guardar en buffer
            agent.replay_buffer.push(state, action, reward, next_state, done or truncated)
            
            episode_reward += reward
            state = next_state
            agent.steps += 1
            
            # Entrenar
            if agent.steps % hparams['train_freq'] == 0:
                loss = agent.train_step()
                if loss is not None:
                    episode_loss.append(loss)
            
            # Actualizar target network
            if agent.steps % hparams['target_update_freq'] == 0:
                agent.update_target_network()
            
            if done or truncated:
                break
        
        # Actualizar epsilon después del episodio
        agent.update_epsilon(episode)
        all_rewards.append(episode_reward)
        all_epsilons.append(agent.epsilon)
        all_steps_per_episode.append(step + 1)
        
        # Guardar loss promedio del episodio
        avg_loss = np.mean(episode_loss) if episode_loss else 0
        all_losses.append(avg_loss)
        
        # Actualizar mejor recompensa
        if episode_reward > best_reward:
            best_reward = episode_reward
            # Guardar mejor modelo
            best_model_path = Path(hparams['model_dir']) / 'dqn_best.pth'
            torch.save(agent.policy_net.state_dict(), best_model_path)
            print(f"\n🏆 Nuevo mejor modelo guardado! Recompensa: {best_reward:.0f}")
        
        # Log
        if episode % 10 == 0:
            recent_avg = np.mean(all_rewards[-100:]) if len(all_rewards) >= 100 else np.mean(all_rewards)
            print(f"Ep {episode:4d} | Reward: {episode_reward:6.0f} | "
                  f"Avg100: {recent_avg:6.1f} | Eps: {agent.epsilon:.3f} | "
                  f"Loss: {avg_loss:.4f} | Buffer: {len(agent.replay_buffer)}")
        
        # Guardar checkpoint con nombre descriptivo
        if (episode + 1) % hparams['save_freq'] == 0:
            checkpoint_path = Path(hparams['model_dir']) / f'dqn_ep{episode+1}.pth'
            checkpoint_data = {
                'episode': episode,
                'model_state_dict': agent.policy_net.state_dict(),
                'optimizer_state_dict': agent.optimizer.state_dict(),
                'rewards': all_rewards,
                'losses': all_losses,
                'epsilons': all_epsilons,
                'steps': all_steps_per_episode,
                'best_reward': best_reward
            }
            torch.save(checkpoint_data, checkpoint_path)
            print(f"\n💾 Checkpoint guardado: {checkpoint_path}")
            
            # También guardar como checkpoint general (para resume_training)
            general_checkpoint = Path(hparams['model_dir']) / 'dqn_checkpoint.pth'
            torch.save(checkpoint_data, general_checkpoint)
            
            # Generar gráficas
            plotter.plot_training_metrics(
                all_rewards, all_losses, all_epsilons, 
                all_steps_per_episode, 'DQN'
            )
        
        # Grabar videos periódicamente
        if (episode + 1) % hparams['video_freq'] == 0:
            print(f"\n🎥 Grabando videos de evaluación (episodio {episode+1})...")
            record_evaluation_videos(
                agent, preprocessor, stacker, 
                hparams, episode + 1, n_actions
            )
    
    # Guardar modelo final
    final_path = Path(hparams['model_dir']) / 'dqn_final.pth'
    torch.save(agent.policy_net.state_dict(), final_path)
    print(f"\nModelo final guardado en: {final_path}")
    
    # Gráficas finales
    plotter.plot_training_metrics(
        all_rewards, all_losses, all_epsilons, 
        all_steps_per_episode, 'DQN'
    )
    
    # Guardar estadísticas en JSON
    stats = {
        'total_episodes': len(all_rewards),
        'best_reward': float(best_reward),
        'final_avg_100': float(np.mean(all_rewards[-100:])),
        'final_epsilon': float(agent.epsilon),
        'total_steps': int(agent.steps)
    }
    
    stats_path = Path(hparams['graphics_dir']) / 'dqn_stats.json'
    import json
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"\nEstadísticas guardadas en: {stats_path}")
    
    env.close()
    return all_rewards

# ==================== MAIN ====================

if __name__ == "__main__":
    rewards = train_dqn(HIPERPARAMETROS)
    print("\nENTRENAMIENTO COMPLETADO")
    print(f"Recompensa promedio últimos 100 eps: {np.mean(rewards[-100:]):.2f}")