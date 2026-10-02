import torch
import torch.nn as nn
import torch.nn.functional as F

class ACNet(nn.Module):
    def __init__(self, state_size, action_size):
        super(ACNet, self).__init__()
        
        # Consistent with NeurIPS 2020 architecture
        self.fc1 = nn.Linear(state_size, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256, 256)
        self.fc5 = nn.Linear(256, 128)
        self.fc6 = nn.Linear(128, 64)
        self.fc7 = nn.Linear(64, 1024)
        
        # Shared heads
        self.policy_head = nn.Linear(1024, action_size)
        self.value_head = nn.Linear(1024, 1)
        
        # Initialize weights (normalized column style)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=nn.init.calculate_gain('relu'))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
        # Specific gains for output layers
        nn.init.orthogonal_(self.policy_head.weight, gain=0.01)
        nn.init.orthogonal_(self.value_head.weight, gain=1.0)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))
        
        policy = F.softmax(self.policy_head(x), dim=-1)
        value = self.value_head(x)
        
        return policy, value
