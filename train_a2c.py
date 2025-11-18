"""
Universidad del Valle de Guatemala | Aprendizaje por refuerzo - S21
Marco Carbajal (23025) / car23025@uvg.edu.gt

Entrenamiento con Advantage Actor-Critic (A2C)
"""

# Inicializar entornos de ALE (debe ser lo primero)
import init_env

import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from utils import FramePreprocessor, FrameStacker, TrainingPlotter, save_checkpoint, load_checkpoint

# ==================== HIPERPARÁMETROS ====================

HIPERPARAMETROS = {
    # Entorno
    'env_name': 'ALE/Galaxian-v5',
    
    # Entrenamiento
    'n_episodes': 2000,              # Total
    'max_steps': 10000,              # Pasos máximos por episodio
    'n_steps': 5,                    # Pasos antes de actualizar (n-step returns)
    'learning_rate': 0.0005,         # Learning rate (ajustado para A2C)
    'gamma': 0.99,                   # Factor de descuento
    
    # A2C específico
    'value_coef': 0.5,               # Coeficiente de loss del critic
    'entropy_coef': 0.01,            # Coeficiente de entropía (exploración)
    'max_grad_norm': 0.5,            # Gradient clipping
    
    # Preprocesamiento
    'n_frames': 4,                   # Frames a apilar
    'img_size': (84, 84),            # Tamaño de imagen procesada
    
    # Guardado
    'save_freq': 250,                # Guardar checkpoint cada N episodios
    'video_freq': 100,               # Grabar video cada N episodios
    'model_dir': 'modelos',
    'graphics_dir': 'graficas',
    'video_dir': 'videos_entrenamiento',
    
    # Continuar entrenamiento
    'resume_training': False,        # True para continuar, False para empezar desde 0
    'checkpoint_path': 'modelos/a2c_checkpoint.pth'
}

# ==================== RED NEURONAL ====================

class ActorCritic(nn.Module):
    """Red neuronal que combina Actor y Critic"""
    
    def __init__(self, n_frames, n_actions):
        super(ActorCritic, self).__init__()
        
        # Capas convolucionales compartidas
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
        
        # Capa compartida
        self.fc_shared = nn.Linear(conv_out_size, 512)
        
        # Actor (política)
        self.actor = nn.Linear(512, n_actions)
        
        # Critic (función de valor)
        self.critic = nn.Linear(512, 1)
    
    def _get_conv_out(self, shape):
        """Calcula el tamaño de salida de las convoluciones"""
        with torch.no_grad():
            x = torch.zeros(1, *shape)
            x = self.conv(x)
            return int(np.prod(x.size()))
    
    def forward(self, x):
        """Forward pass - retorna logits y valor"""
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc_shared(x))
        
        logits = self.actor(x)
        value = self.critic(x)
        
        return logits, value

# ==================== AGENTE A2C ====================

class A2CAgent:
    """Agente que usa Actor-Critic para aprender"""
    
    def __init__(self, n_actions, device, hparams):
        self.n_actions = n_actions
        self.device = device
        self.hparams = hparams
        
        # Red Actor-Critic
        self.model = ActorCritic(hparams['n_frames'], n_actions).to(device)
        
        # Optimizador
        self.optimizer = optim.RMSprop(self.model.parameters(), 
                                       lr=hparams['learning_rate'],
                                       alpha=0.99, eps=1e-5)
    
    def select_action(self, state, training=True):
        """Selecciona acción muestreando de la política"""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            logits, _ = self.model(state_tensor)
            
            if training:
                # Muestrear de la distribución
                probs = F.softmax(logits, dim=-1)
                action = torch.multinomial(probs, 1).item()
            else:
                # Tomar la acción más probable
                action = logits.argmax().item()
            
            return action
    
    def compute_returns(self, rewards, values, dones):
        """Calcula los returns con n-step"""
        returns = []
        R = values[-1] * (1 - dones[-1])  # Bootstrap con último valor
        
        for i in reversed(range(len(rewards))):
            R = rewards[i] + self.hparams['gamma'] * R * (1 - dones[i])
            returns.insert(0, R)
        
        return returns
    
    def train_step(self, states, actions, rewards, values, dones):
        """Realiza un paso de entrenamiento"""
    
        # Protección: nunca entrenar con batch vacío
        if len(states) == 0:
            return 0, 0, 0, 0
        
        # Convertir a tensores
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        
        # Forward pass
        logits, new_values = self.model(states)
        new_values = new_values.squeeze(-1)
        
        # Calcular returns
        returns = self.compute_returns(rewards, values + [new_values[-1].item()], dones)
        returns = torch.FloatTensor(returns).to(self.device)
        
        # Ventajas
        advantages = returns - new_values.detach()
        
        # Actor loss (policy gradient)
        log_probs = F.log_softmax(logits, dim=-1)
        action_log_probs = log_probs.gather(1, actions.unsqueeze(1)).squeeze()
        actor_loss = -(action_log_probs * advantages).mean()
        
        # Critic loss (MSE)
        critic_loss = F.mse_loss(new_values, returns)
        
        # Entropy loss (para exploración)
        probs = F.softmax(logits, dim=-1)
        entropy = -(probs * log_probs).sum(dim=-1).mean()
        
        # Loss total
        total_loss = (actor_loss + 
                     self.hparams['value_coef'] * critic_loss - 
                     self.hparams['entropy_coef'] * entropy)
        
        # Backpropagation
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 
                                       self.hparams['max_grad_norm'])
        self.optimizer.step()
        
        return total_loss.item(), actor_loss.item(), critic_loss.item(), entropy.item()

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
    policy = create_live_policy(agent, preprocessor, stacker, method='a2c')
    
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
        final_name = f'a2c_ep{episode_num}_score{int(total_reward)}.mp4'
        final_path = video_dir / final_name
        shutil.move(str(video_files[0]), str(final_path))
    
    # Limpiar temporal
    try:
        shutil.rmtree(temp_dir)
    except:
        pass
    
    print(f"  📹 Video: Score = {total_reward:.0f}")

# ==================== ENTRENAMIENTO ====================

def train_a2c(hparams):
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
    agent = A2CAgent(n_actions, device, hparams)
    
    # Para gráficas y análisis
    plotter = TrainingPlotter(hparams['graphics_dir'])
    
    # Métricas de entrenamiento
    all_rewards = []
    all_losses = []
    all_steps_per_episode = []
    best_reward = -float('inf')
    
    # Cargar checkpoint si se desea continuar
    start_episode = 0
    
    if hparams['resume_training']:
        try:
            start_episode, all_rewards = load_checkpoint(
                agent.model, 
                agent.optimizer, 
                hparams['checkpoint_path']
            )
            
            # Cargar también otras métricas si existen
            checkpoint = torch.load(hparams['checkpoint_path'], weights_only=False)
            if 'losses' in checkpoint:
                all_losses = checkpoint['losses']
                print(f"Losses restaurados: {len(all_losses)} episodios")
            if 'steps' in checkpoint:
                all_steps_per_episode = checkpoint['steps']
                print(f"Steps restaurados: {len(all_steps_per_episode)} episodios")
            if 'best_reward' in checkpoint:
                best_reward = checkpoint['best_reward']
                print(f"Mejor recompensa restaurada: {best_reward:.0f}")
                
        except FileNotFoundError:
            print("No se encontró checkpoint, empezando desde cero")
    
    print("\n" + "="*60)
    print(f"ENTRENAMIENTO A2C - Galaxian")
    print("="*60)
    print(f"Episodios: {start_episode} -> {hparams['n_episodes']}")
    print(f"Learning rate: {hparams['learning_rate']}")
    print(f"Gamma: {hparams['gamma']}")
    print(f"N-steps: {hparams['n_steps']}")
    print("="*60 + "\n")
    
    # Loop de entrenamiento
    for episode in range(start_episode, hparams['n_episodes']):
        # Reset
        obs, _ = env.reset()
        obs = preprocessor.process(obs)
        stacker.reset()
        state = stacker.stack(obs)
        
        episode_reward = 0
        episode_losses = []
        
        # Buffers para n-step
        states_buffer = []
        actions_buffer = []
        rewards_buffer = []
        values_buffer = []
        dones_buffer = []
        
        for step in range(hparams['max_steps']):
            # Seleccionar acción
            action = agent.select_action(state, training=True)
            
            # Obtener valor del estado actual
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                _, value = agent.model(state_tensor)
                value = value.item()
            
            # Ejecutar acción
            next_obs, reward, done, truncated, _ = env.step(action)
            
            # Preprocesar
            next_obs = preprocessor.process(next_obs)
            next_state = stacker.stack(next_obs)
            
            # Guardar en buffers
            states_buffer.append(state)
            actions_buffer.append(action)
            rewards_buffer.append(reward)
            values_buffer.append(value)
            dones_buffer.append(done or truncated)
            
            episode_reward += reward
            state = next_state
            
            # Entrenar cada n_steps o al terminar episodio
            if len(states_buffer) > 0 and (len(states_buffer) >= hparams['n_steps'] or done or truncated):
                losses = agent.train_step(states_buffer, actions_buffer, 
                                        rewards_buffer, values_buffer, dones_buffer)
                episode_losses.append(losses[0])
                
                # Limpiar buffers
                states_buffer = []
                actions_buffer = []
                rewards_buffer = []
                values_buffer = []
                dones_buffer = []
            
            if done or truncated:
                break
        
        all_rewards.append(episode_reward)
        all_steps_per_episode.append(step + 1)
        
        # Guardar loss promedio del episodio
        avg_loss = np.mean(episode_losses) if episode_losses else 0
        all_losses.append(avg_loss)
        
        # Actualizar mejor recompensa
        if episode_reward > best_reward:
            best_reward = episode_reward
            # Guardar mejor modelo
            best_model_path = Path(hparams['model_dir']) / 'a2c_best.pth'
            torch.save(agent.model.state_dict(), best_model_path)
            print(f"\n🏆 Nuevo mejor modelo guardado! Recompensa: {best_reward:.0f}")
        
        # Log
        if episode % 10 == 0:
            recent_avg = np.mean(all_rewards[-100:]) if len(all_rewards) >= 100 else np.mean(all_rewards)
            print(f"Ep {episode:4d} | Reward: {episode_reward:6.0f} | "
                  f"Avg100: {recent_avg:6.1f} | Loss: {avg_loss:.4f}")
        
        # Guardar checkpoint con nombre descriptivo
        if (episode + 1) % hparams['save_freq'] == 0:
            checkpoint_path = Path(hparams['model_dir']) / f'a2c_ep{episode+1}.pth'
            checkpoint_data = {
                'episode': episode,
                'model_state_dict': agent.model.state_dict(),
                'optimizer_state_dict': agent.optimizer.state_dict(),
                'rewards': all_rewards,
                'losses': all_losses,
                'steps': all_steps_per_episode,
                'best_reward': best_reward
            }
            torch.save(checkpoint_data, checkpoint_path)
            print(f"\n💾 Checkpoint guardado: {checkpoint_path}")
            
            # También guardar como checkpoint general (para resume_training)
            general_checkpoint = Path(hparams['model_dir']) / 'a2c_checkpoint.pth'
            torch.save(checkpoint_data, general_checkpoint)
            
            # Generar gráficas
            plotter.plot_training_metrics(
                all_rewards, all_losses, None,  # A2C no tiene epsilon
                all_steps_per_episode, 'A2C'
            )
        
        # Grabar videos periódicamente
        if (episode + 1) % hparams['video_freq'] == 0:
            print(f"\n🎥 Grabando videos de evaluación (episodio {episode+1})...")
            record_evaluation_videos(
                agent, preprocessor, stacker, 
                hparams, episode + 1, n_actions
            )
    
    # Guardar modelo final
    final_path = Path(hparams['model_dir']) / 'a2c_final.pth'
    torch.save(agent.model.state_dict(), final_path)
    print(f"\nModelo final guardado en: {final_path}")
    
    # Gráficas finales
    plotter.plot_training_metrics(
        all_rewards, all_losses, None,  # A2C no tiene epsilon
        all_steps_per_episode, 'A2C'
    )
    
    # Guardar estadísticas en JSON
    stats = {
        'total_episodes': len(all_rewards),
        'best_reward': float(best_reward),
        'final_avg_100': float(np.mean(all_rewards[-100:])) if len(all_rewards) >= 100 else float(np.mean(all_rewards)),
    }
    
    stats_path = Path(hparams['graphics_dir']) / 'a2c_stats.json'
    import json
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"\nEstadísticas guardadas en: {stats_path}")
    
    env.close()
    return all_rewards

# ==================== MAIN ====================

if __name__ == "__main__":
    rewards = train_a2c(HIPERPARAMETROS)
    print("\nENTRENAMIENTO COMPLETADO")
    if len(rewards) >= 100:
        print(f"Recompensa promedio últimos 100 eps: {np.mean(rewards[-100:]):.2f}")
    else:
        print(f"Recompensa promedio total: {np.mean(rewards):.2f}")