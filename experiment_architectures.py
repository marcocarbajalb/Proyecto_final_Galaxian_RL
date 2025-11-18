"""
Universidad del Valle de Guatemala | Aprendizaje por refuerzo - S21
Marco Carbajal (23025) / car23025@uvg.edu.gt

Experimentación con diferentes arquitecturas DQN
- Cada variante se entrena independientemente y los resultados se comparan al final.
"""

# Inicializar entornos de ALE (debe ser lo primero)
import init_env

import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pathlib import Path
import json
from utils import FramePreprocessor, FrameStacker, ReplayBuffer, TrainingPlotter
import matplotlib.pyplot as plt

# ==================== CONFIGURACIÓN DE EXPERIMENTOS ====================

EXPERIMENT_CONFIG = {
    # Entrenamiento rápido para comparación
    'n_episodes': 500,              # Episodios por variante
    'max_steps': 10000,
    'batch_size': 32,
    'buffer_size': 50000,           # Buffer más pequeño para entrenar más rápido
    'learning_rate': 0.0001,
    'gamma': 0.99,
    
    # Exploración
    'epsilon_start': 1.0,
    'epsilon_end': 0.01,
    'epsilon_decay': 400,           # Decay ajustado a 500 episodios
    
    # DQN específico
    'target_update_freq': 1000,
    'min_buffer_size': 5000,        # Menos para empezar a entrenar antes
    'train_freq': 4,
    
    # Preprocesamiento
    'n_frames': 4,
    'img_size': (84, 84),
    
    # Directorios separados por experimento
    'base_dir': 'experimentos',
    'save_freq': 100,               # Guardar cada 100 eps
}

# ==================== ARQUITECTURAS A PROBAR ====================

class DQN_Original(nn.Module):
    """Arquitectura original (baseline)"""
    
    def __init__(self, n_frames, n_actions):
        super(DQN_Original, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(n_frames, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        
        conv_out_size = self._get_conv_out((n_frames, 84, 84))
        
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )
    
    def _get_conv_out(self, shape):
        with torch.no_grad():
            x = torch.zeros(1, *shape)
            x = self.conv(x)
            return int(np.prod(x.size()))
    
    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class DQN_Deeper(nn.Module):
    """Arquitectura más profunda - más capas convolucionales"""
    
    def __init__(self, n_frames, n_actions):
        super(DQN_Deeper, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(n_frames, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1),  # Capa extra
            nn.ReLU()
        )
        
        conv_out_size = self._get_conv_out((n_frames, 84, 84))
        
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, 256),  # Capa extra
            nn.ReLU(),
            nn.Linear(256, n_actions)
        )
    
    def _get_conv_out(self, shape):
        with torch.no_grad():
            x = torch.zeros(1, *shape)
            x = self.conv(x)
            return int(np.prod(x.size()))
    
    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class DQN_Wider(nn.Module):
    """Arquitectura más ancha - más filtros por capa"""
    
    def __init__(self, n_frames, n_actions):
        super(DQN_Wider, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(n_frames, 64, kernel_size=8, stride=4),   # 32 -> 64
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2),        # 64 -> 128
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1),       # 64 -> 128
            nn.ReLU()
        )
        
        conv_out_size = self._get_conv_out((n_frames, 84, 84))
        
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 1024),  # 512 -> 1024
            nn.ReLU(),
            nn.Linear(1024, n_actions)
        )
    
    def _get_conv_out(self, shape):
        with torch.no_grad():
            x = torch.zeros(1, *shape)
            x = self.conv(x)
            return int(np.prod(x.size()))
    
    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class DQN_Dueling(nn.Module):
    """Dueling DQN - separa valor y ventaja"""
    
    def __init__(self, n_frames, n_actions):
        super(DQN_Dueling, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(n_frames, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        
        conv_out_size = self._get_conv_out((n_frames, 84, 84))
        
        # Stream de valor (V)
        self.value_stream = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )
        
        # Stream de ventaja (A)
        self.advantage_stream = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )
    
    def _get_conv_out(self, shape):
        with torch.no_grad():
            x = torch.zeros(1, *shape)
            x = self.conv(x)
            return int(np.prod(x.size()))
    
    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        
        value = self.value_stream(x)
        advantage = self.advantage_stream(x)
        
        # Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        
        return q_values

# ==================== CATÁLOGO DE ARQUITECTURAS ====================

ARCHITECTURES = {
    'original': {
        'class': DQN_Original,
        'description': 'Arquitectura baseline (3 capas convolucionales + 2 fully connected)',
    },
    'deeper': {
        'class': DQN_Deeper,
        'description': 'Mas profunda (4 capas convolucionales + 3 fully connected)',
    },
    'wider': {
        'class': DQN_Wider,
        'description': 'Mas ancha (mas filtros por capa)',
    },
    'dueling': {
        'class': DQN_Dueling,
        'description': 'Dueling DQN (separa valor y ventaja)',
    },
}

# ==================== AGENTE GENÉRICO ====================

class DQNAgent:
    """Agente DQN genérico que funciona con cualquier arquitectura"""
    
    def __init__(self, architecture_class, n_actions, device, hparams):
        self.n_actions = n_actions
        self.device = device
        self.hparams = hparams
        
        # Redes (policy y target)
        self.policy_net = architecture_class(hparams['n_frames'], n_actions).to(device)
        self.target_net = architecture_class(hparams['n_frames'], n_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        # Optimizador
        self.optimizer = optim.Adam(self.policy_net.parameters(), 
                                    lr=hparams['learning_rate'])
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(hparams['buffer_size'])
        
        # Contadores
        self.steps = 0
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
        
        # Q-values siguientes
        with torch.no_grad():
            next_q = self.target_net(next_states).max(1)[0]
            target_q = rewards + (1 - dones) * self.hparams['gamma'] * next_q
        
        # Loss y backpropagation
        loss = nn.MSELoss()(current_q, target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        
        return loss.item()
    
    def update_target_network(self):
        """Copia pesos de policy network a target network"""
        self.target_net.load_state_dict(self.policy_net.state_dict())

# ==================== ENTRENAMIENTO DE UNA VARIANTE ====================

def train_variant(variant_name, architecture_class, config):
    """Entrena una variante de arquitectura"""
    
    print(f"\n{'='*60}")
    print(f"Entrenando variante: {variant_name.upper()}")
    print(f"Descripción: {ARCHITECTURES[variant_name]['description']}")
    print(f"{'='*60}\n")
    
    # Configuración
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Crear directorios para esta variante
    variant_dir = Path(config['base_dir']) / variant_name
    model_dir = variant_dir / 'modelos'
    graphics_dir = variant_dir / 'graficas'
    
    model_dir.mkdir(parents=True, exist_ok=True)
    graphics_dir.mkdir(parents=True, exist_ok=True)
    
    # Crear entorno
    env = gym.make('ALE/Galaxian-v5')
    n_actions = env.action_space.n
    
    # Preprocesamiento
    preprocessor = FramePreprocessor(img_size=config['img_size'])
    stacker = FrameStacker(n_frames=config['n_frames'])
    
    # Crear agente
    agent = DQNAgent(architecture_class, n_actions, device, config)
    
    # Métricas
    all_rewards = []
    all_losses = []
    best_reward = -float('inf')
    
    # Entrenamiento
    for episode in range(config['n_episodes']):
        # Reset
        obs, _ = env.reset()
        obs = preprocessor.process(obs)
        stacker.reset()
        state = stacker.stack(obs)
        
        episode_reward = 0
        episode_loss = []
        
        for step in range(config['max_steps']):
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
            if agent.steps % config['train_freq'] == 0:
                loss = agent.train_step()
                if loss is not None:
                    episode_loss.append(loss)
            
            # Actualizar target network
            if agent.steps % config['target_update_freq'] == 0:
                agent.update_target_network()
            
            if done or truncated:
                break
        
        # Actualizar epsilon
        agent.update_epsilon(episode)
        all_rewards.append(episode_reward)
        
        avg_loss = np.mean(episode_loss) if episode_loss else 0
        all_losses.append(avg_loss)
        
        # Actualizar mejor recompensa
        if episode_reward > best_reward:
            best_reward = episode_reward
        
        # Log
        if episode % 10 == 0:
            recent_avg = np.mean(all_rewards[-100:]) if len(all_rewards) >= 100 else np.mean(all_rewards)
            print(f"Ep {episode:3d} | Reward: {episode_reward:6.0f} | "
                  f"Avg100: {recent_avg:6.1f} | Eps: {agent.epsilon:.3f}")
        
        # Guardar checkpoint
        if (episode + 1) % config['save_freq'] == 0:
            checkpoint_path = model_dir / f'checkpoint_ep{episode+1}.pth'
            torch.save({
                'episode': episode,
                'model_state_dict': agent.policy_net.state_dict(),
                'rewards': all_rewards,
                'best_reward': best_reward
            }, checkpoint_path)
    
    # Guardar modelo final
    final_path = model_dir / 'final.pth'
    torch.save(agent.policy_net.state_dict(), final_path)
    
    # Generar gráficas individuales
    plotter = TrainingPlotter(str(graphics_dir))
    plotter.plot_rewards(all_rewards, variant_name.upper())
    
    # Guardar resultados
    results = {
        'variant': variant_name,
        'description': ARCHITECTURES[variant_name]['description'],
        'total_episodes': len(all_rewards),
        'best_reward': float(best_reward),
        'final_avg_100': float(np.mean(all_rewards[-100:])) if len(all_rewards) >= 100 else float(np.mean(all_rewards)),
        'final_avg_50': float(np.mean(all_rewards[-50:])),
        'rewards': [float(r) for r in all_rewards],
    }
    
    results_path = variant_dir / 'results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    env.close()
    
    print(f"\n✓ Variante '{variant_name}' completada")
    print(f"  Mejor recompensa: {best_reward:.0f}")
    print(f"  Promedio últimos 100: {results['final_avg_100']:.2f}")
    
    return results

# ==================== COMPARACIÓN DE RESULTADOS ====================

def compare_variants(all_results):
    """Compara los resultados de todas las variantes"""
    
    print(f"\n{'='*60}")
    print("COMPARACIÓN DE ARQUITECTURAS")
    print(f"{'='*60}\n")
    
    # Tabla comparativa
    print(f"{'Variante':<15} {'Descripción':<35} {'Best':>8} {'Avg100':>8}")
    print("-" * 70)
    
    for result in all_results:
        print(f"{result['variant']:<15} {result['description']:<35} "
              f"{result['best_reward']:>8.0f} {result['final_avg_100']:>8.1f}")
    
    # Mejor variante
    best = max(all_results, key=lambda x: x['final_avg_100'])
    print(f"\n🏆 MEJOR ARQUITECTURA: {best['variant'].upper()}")
    print(f"   {best['description']}")
    print(f"   Avg100: {best['final_avg_100']:.1f}")
    
    # Gráfica comparativa
    plt.figure(figsize=(12, 6))
    
    window = 50  # Media móvil de 50 episodios
    
    for result in all_results:
        rewards = result['rewards']
        if len(rewards) >= window:
            moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
            plt.plot(range(window-1, len(rewards)), moving_avg, 
                    linewidth=2, label=f"{result['variant']}")
    
    plt.xlabel('Episodio')
    plt.ylabel(f'Recompensa (media móvil {window} eps)')
    plt.title('Comparación de Arquitecturas DQN')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    comparison_dir = Path(EXPERIMENT_CONFIG['base_dir']) / 'comparacion'
    comparison_dir.mkdir(parents=True, exist_ok=True)
    
    plt.savefig(comparison_dir / 'architectures_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Guardar resumen
    summary = {
        'best_architecture': best['variant'],
        'best_avg_100': best['final_avg_100'],
        'all_results': all_results
    }
    
    with open(comparison_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n✓ Gráfica de comparación guardada en: {comparison_dir / 'architectures_comparison.png'}")
    print(f"✓ Resumen guardado en: {comparison_dir / 'summary.json'}")

# ==================== MAIN ====================

def main():
    """Función principal"""
    
    print("="*60)
    print("EXPERIMENTACIÓN CON ARQUITECTURAS DQN")
    print("Marco Carbajal (23025) / car23025@uvg.edu.gt")
    print("="*60)
    print(f"\nEntrenamiento rápido: {EXPERIMENT_CONFIG['n_episodes']} episodios por variante")
    print("Esto permite comparar rápidamente todas las arquitecturas.\n")
    
    # Listar arquitecturas a entrenar
    print("Arquitecturas disponibles:")
    for i, (name, info) in enumerate(ARCHITECTURES.items(), 1):
        print(f"  {i}. {name}: {info['description']}")
    
    variants_to_train = list(ARCHITECTURES.keys())
    
    print(f"\n✓ Se entrenarán {len(variants_to_train)} variantes: {', '.join(variants_to_train)}")
    input("\nPresiona 'Enter' para comenzar...")
    
    # Entrenar cada variante
    all_results = []
    
    for variant_name in variants_to_train:
        results = train_variant(
            variant_name, 
            ARCHITECTURES[variant_name]['class'],
            EXPERIMENT_CONFIG
        )
        all_results.append(results)
    
    # Comparar resultados
    if len(all_results) > 1:
        compare_variants(all_results)
    
    print("\n" + "="*60)
    print("✓ EXPERIMENTACIÓN COMPLETADA")
    print("="*60)
    print(f"\nResultados guardados en: {EXPERIMENT_CONFIG['base_dir']}/")

if __name__ == "__main__":
    main()