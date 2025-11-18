"""
Universidad del Valle de Guatemala | Aprendizaje por refuerzo - S21
Marco Carbajal (23025) / car23025@uvg.edu.gt

Wrapper para cargar y usar políticas entrenadas
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils import FramePreprocessor, FrameStacker

# ==================== REDES NEURONALES ====================

class DQN(nn.Module):
    """Red neuronal convolucional para DQN"""
    
    def __init__(self, n_frames, n_actions):
        super(DQN, self).__init__()
        
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

class ActorCritic(nn.Module):
    """Red neuronal que combina Actor y Critic"""
    
    def __init__(self, n_frames, n_actions):
        super(ActorCritic, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(n_frames, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        
        conv_out_size = self._get_conv_out((n_frames, 84, 84))
        
        self.fc_shared = nn.Linear(conv_out_size, 512)
        self.actor = nn.Linear(512, n_actions)
        self.critic = nn.Linear(512, 1)
    
    def _get_conv_out(self, shape):
        with torch.no_grad():
            x = torch.zeros(1, *shape)
            x = self.conv(x)
            return int(np.prod(x.size()))
    
    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc_shared(x))
        
        logits = self.actor(x)
        value = self.critic(x)
        
        return logits, value

# ==================== POLÍTICAS ENTRENADAS (Compatible con play.py) ====================

class TrainedDQNPolicy:
    """
    Política entrenada con DQN - Compatible con play.py
    Esta clase implementa la interfaz que espera play.py
    """
    
    def __init__(self, model_path, n_actions=6, n_frames=4, img_size=(84, 84)):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Cargar modelo
        self.model = DQN(n_frames, n_actions).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
        self.model.eval()
        
        # Preprocesamiento
        self.preprocessor = FramePreprocessor(img_size=img_size)
        self.stacker = FrameStacker(n_frames=n_frames)
        
        print(f"DQN policy cargada desde: {model_path}")
    
    def select_action(self, observacion: np.ndarray) -> int:
        """
        Selecciona la mejor acción según el modelo
        
        Args:
            observacion: Frame RGB del entorno (210, 160, 3)
        Returns:
            Acción a ejecutar (int)
        """
        # Preprocesar
        frame = self.preprocessor.process(observacion)
        state = self.stacker.stack(frame)
        
        # Predecir
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.model(state_tensor)
            action = q_values.argmax().item()
        
        return action
    
    def reset(self):
        """Reinicia el stacker de frames"""
        self.stacker.reset()

class TrainedA2CPolicy:
    """
    Política entrenada con A2C - Compatible con play.py
    Esta clase implementa la interfaz que espera play.py
    """
    
    def __init__(self, model_path, n_actions=6, n_frames=4, img_size=(84, 84)):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Cargar modelo
        self.model = ActorCritic(n_frames, n_actions).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
        self.model.eval()
        
        # Preprocesamiento
        self.preprocessor = FramePreprocessor(img_size=img_size)
        self.stacker = FrameStacker(n_frames=n_frames)
        
        print(f"A2C policy cargada desde: {model_path}")
    
    def select_action(self, observacion: np.ndarray) -> int:
        """
        Selecciona la mejor acción según el modelo
        
        Args:
            observacion: Frame RGB del entorno (210, 160, 3)
        Returns:
            Acción a ejecutar (int)
        """
        # Preprocesar
        frame = self.preprocessor.process(observacion)
        state = self.stacker.stack(frame)
        
        # Predecir
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            logits, _ = self.model(state_tensor)
            action = logits.argmax().item()  # Tomar la acción más probable
        
        return action
    
    def reset(self):
        """Reinicia el stacker de frames"""
        self.stacker.reset()

# ==================== POLÍTICAS EN VIVO (Durante entrenamiento) ====================

class LiveDQNPolicy:
    """
    Política DQN en vivo durante el entrenamiento
    Wrapper que convierte el agente de entrenamiento en una política compatible con play.py
    """
    
    def __init__(self, agent, preprocessor, stacker):
        self.agent = agent
        self.preprocessor = preprocessor
        self.stacker = stacker
    
    def select_action(self, observacion: np.ndarray) -> int:
        """Selecciona acción usando el agente actual"""
        frame = self.preprocessor.process(observacion)
        state = self.stacker.stack(frame)
        # Sin exploración (epsilon=0) para evaluación
        return self.agent.select_action(state, training=False)
    
    def reset(self):
        """Reinicia el stacker"""
        self.stacker.reset()

class LiveA2CPolicy:
    """
    Política A2C en vivo durante el entrenamiento
    Wrapper que convierte el agente de entrenamiento en una política compatible con play.py
    """
    
    def __init__(self, agent, preprocessor, stacker):
        self.agent = agent
        self.preprocessor = preprocessor
        self.stacker = stacker
    
    def select_action(self, observacion: np.ndarray) -> int:
        """Selecciona acción usando el agente actual"""
        frame = self.preprocessor.process(observacion)
        state = self.stacker.stack(frame)
        # Sin exploración para evaluación
        return self.agent.select_action(state, training=False)
    
    def reset(self):
        """Reinicia el stacker"""
        self.stacker.reset()

# ==================== FUNCIONES AUXILIARES ====================

def load_policy(model_path, method='dqn', n_actions=6, n_frames=4, img_size=(84, 84)):
    """
    Carga una política entrenada
    
    Args:
        model_path: ruta al modelo guardado (.pth)
        method: 'dqn' o 'a2c'
        n_actions: número de acciones (default: 6)
        n_frames: frames apilados (default: 4)
        img_size: tamaño de imagen procesada (default: (84, 84))
    
    Returns:
        Política lista para usar con play.py
    """
    if method.lower() == 'dqn':
        return TrainedDQNPolicy(model_path, n_actions, n_frames, img_size)
    elif method.lower() == 'a2c':
        return TrainedA2CPolicy(model_path, n_actions, n_frames, img_size)
    else:
        raise ValueError(f"Método no reconocido: {method}. Use 'dqn' o 'a2c'")

def create_live_policy(agent, preprocessor, stacker, method='dqn'):
    """
    Crea una política en vivo desde un agente de entrenamiento
    
    Args:
        agent: DQNAgent o A2CAgent
        preprocessor: FramePreprocessor
        stacker: FrameStacker
        method: 'dqn' o 'a2c'
    
    Returns:
        Política compatible con play.py
    """
    if method.lower() == 'dqn':
        return LiveDQNPolicy(agent, preprocessor, stacker)
    elif method.lower() == 'a2c':
        return LiveA2CPolicy(agent, preprocessor, stacker)
    else:
        raise ValueError(f"Método no reconocido: {method}")