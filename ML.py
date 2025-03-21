import pygame
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import matplotlib.pyplot as plt

# Set device for PyTorch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Initialize pygame
pygame.init()
width, height = 700, 800
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("Sky Runner")
clock = pygame.time.Clock()

background_image = pygame.image.load("images/01.jpg")
background_image = pygame.transform.scale(background_image, (width, height))

# Neural Network for Agent
class AgentBrain(nn.Module):
    def __init__(self):
        super(AgentBrain, self).__init__()
        self.fc1 = nn.Linear(8, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256, 3)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x

# Obstacle class
class Obstacle:
    def __init__(self):
        self.images = [
            pygame.image.load("images/00-removebg-preview.png"),
            pygame.image.load("images/02-removebg-preview.png"),
        ]
        self.image = random.choice(self.images)
        self.width = 70
        self.height = 15
        self.x = random.randint(0, width - self.width)
        self.y = -self.height
        self.speed = 6
        self.image = pygame.transform.scale(self.image, (self.width, self.height))

    def move(self):
        self.y += self.speed

    def show(self):
        screen.blit(self.image, (self.x, self.y))

    def check_collision(self, plane):
        return (
            plane.x < self.x + self.width and
            plane.x + plane.width > self.x and
            plane.y < self.y + self.height and
            plane.y + plane.height > self.y
        )

# Plane class
class Plane:
    def __init__(self):
        self.width = 50
        self.height = 70
        self.image = pygame.image.load("images/04-removebg-preview.png")
        self.image = pygame.transform.scale(self.image, (self.width, self.height))
        self.x = width // 2 - self.width // 2
        self.y = height - self.height - 20
        self.speed = 20

    def move(self, action):
        if action == 0:  # Move left
            self.x -= self.speed
        elif action == 2:  # Move right
            self.x += self.speed
        self.x = max(0, min(width - self.width, self.x))

    def show(self):
        screen.blit(self.image, (self.x, self.y))

    def distance_to_gap(self, obstacle):
        if obstacle:
            return abs((self.x + self.width / 2) - (obstacle.x + obstacle.width / 2))
        return float('inf')

# Environment class
class Environment:
    def __init__(self):
        self.plane = Plane()
        self.obstacles = [Obstacle()]
        self.max_obstacles = 5

    def reset(self):
        self.plane = Plane()
        self.obstacles = [Obstacle()]
        return self.get_state()

    def get_state(self):
        state = [
            self.plane.x / width,
            (self.plane.x + self.plane.width / 2) / width,
        ]
        for obstacle in self.obstacles[:2]:
            state.extend([
                obstacle.x / width,
                obstacle.y / height,
                obstacle.speed / 10
            ])
        while len(state) < 8:
            state.extend([0, 0, 0])
        return state

    def step(self, action):
        previous_distance_to_gap = self.plane.distance_to_gap(self.obstacles[0] if self.obstacles else None)
        self.plane.move(action)

        reward = 0.1  # For being alive
        if action == 1:
            reward -= 0.1

        # elif action == 1:
        #     reward += 0.3


        # Move obstacles1
        for obstacle in self.obstacles:
            obstacle.move()

        # Check collision
        if any(obstacle.check_collision(self.plane) for obstacle in self.obstacles):
            reward -= 10
            return reward, self.get_state(), True

        # Remove passed obstacles
        if self.obstacles[0].y > height:
            reward += 50
            self.obstacles.pop(0)
        else:
            current_distance_to_gap = self.plane.distance_to_gap(self.obstacles[0])
            if current_distance_to_gap < previous_distance_to_gap:
                reward += 1

        # Add new obstacles
        while len(self.obstacles) < self.max_obstacles:
            self.obstacles.append(Obstacle())

        return reward, self.get_state(), False

    def render(self):
        screen.blit(background_image, (0, 0))
        self.plane.show()
        for obstacle in self.obstacles:
            obstacle.show()
        pygame.display.flip()

# DQN setup
agent = AgentBrain().to(device)
target_agent = AgentBrain().to(device)
target_agent.load_state_dict(agent.state_dict())
optimizer = optim.Adam(agent.parameters(), lr=0.001)
loss_fn = nn.MSELoss()

gamma = 0.99
epsilon = 1.0
epsilon_min = 0.05
epsilon_decay = 0.995
replay_buffer = deque(maxlen=10000)
batch_size = 64
epochs = 400
reward_history = []

# Action selection
def select_action(state, epsilon=0):
    if random.random() < epsilon:
        return random.randint(0, 2)
    else:
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        agent.eval()
        with torch.no_grad():
            q_values = agent(state_tensor)
        return q_values.argmax().item()

# Training function
def train():
    if len(replay_buffer) < batch_size:
        return
    minibatch = random.sample(replay_buffer, batch_size)
    states, actions, rewards, next_states, dones = zip(*minibatch)
    states = torch.FloatTensor(states).to(device)
    actions = torch.LongTensor(actions).to(device)
    rewards = torch.FloatTensor(rewards).to(device)
    next_states = torch.FloatTensor(next_states).to(device)
    dones = torch.FloatTensor(dones).to(device)

    q_values = agent(states).gather(1, actions.unsqueeze(1)).squeeze()
    with torch.no_grad():
        target_q_values = target_agent(next_states).max(1)[0]
    targets = rewards + gamma * target_q_values * (1 - dones)

    loss = loss_fn(q_values, targets)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Training loop
env = Environment()
print("Starting training...")
for epoch in range(epochs):
    state = env.reset()
    total_reward = 0
    done = False

    while not done:
        action = select_action(state, epsilon)
        reward, next_state, done = env.step(action)
        replay_buffer.append((state, action, reward, next_state, done))
        train()
        state = next_state
        total_reward += reward

    if epsilon > epsilon_min:
        epsilon *= epsilon_decay

    reward_history.append(total_reward)
    if (epoch + 1) % 100 == 0:
        print(f"Epoch {epoch + 1}, Reward: {total_reward}")

    if epoch % 10 == 0:
        target_agent.load_state_dict(agent.state_dict())

print("Training completed.")

plt.plot(reward_history)
plt.xlabel("Epoch")
plt.ylabel("Total Reward")
plt.title("Training Reward History")
plt.show()

# Run the game
env = Environment()
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    _, _, done = env.step(select_action(env.get_state(), epsilon=0))
    env.render()
    if done:
        env.reset()
    clock.tick(60)

pygame.quit()
