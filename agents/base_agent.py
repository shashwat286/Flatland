from abc import ABC, abstractmethod

class BaseMARLAgent(ABC):
    @abstractmethod
    def act(self, state, epsilon=0.1):
        """Select action based on current observation."""
        pass

    @abstractmethod
    def step(self, state, action, reward, next_state, done):
        """Record transition and perform internal updates."""
        pass

    @abstractmethod
    def save(self, path):
        """Persist model weights."""
        pass

    @abstractmethod
    def load(self, path):
        """Load model weights."""
        pass
