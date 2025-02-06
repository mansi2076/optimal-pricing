import numpy as np
import pandas as pd
import gym
from gym import spaces

data = pd.read_csv('soapnutshistory_cleaned.csv')
data["Report Date"] = pd.to_datetime(data["Report Date"])
data = data.dropna(subset=["Product Price", "Total Sales"])

class PricingEnv(gym.Env):
    def __init__(self, data):
        super(PricingEnv, self).__init__()
        self.data = data
        self.current_index = 0
        
        self.observation_space = spaces.Box(
            low=np.array([data['Product Price'].min(), 0, 0, 0]),
            high=np.array([data['Product Price'].max(), 100, 100, data['Total Sales'].max()]),
            dtype=np.float32
        )
        
        self.action_space = spaces.Discrete(3)
        
    def reset(self):
        self.current_index = 0
        row = self.data.iloc[self.current_index]
        return np.array([row['Product Price'], row['Organic Conversion Percentage'], row['Ad Conversion Percentage'], row['Total Sales']], dtype=np.float32)
    
    def step(self, action):
        row = self.data.iloc[self.current_index]
        
        if action == 0:
            new_price = row['Product Price'] - 0.5
        elif action == 1:
            new_price = row['Product Price']
        else:
            new_price = row['Product Price'] + 0.5
        
        closest_row = self.data.iloc[(self.data['Product Price'] - new_price).abs().argsort()[:1]]
        new_sales = closest_row['Total Sales'].values[0]
        
        reward = new_sales + closest_row['Organic Conversion Percentage'].values[0] + closest_row['Ad Conversion Percentage'].values[0]
        
        done = self.current_index >= len(self.data) - 1
        self.current_index += 1
        
        return np.array([new_price, closest_row['Organic Conversion Percentage'].values[0], closest_row['Ad Conversion Percentage'].values[0], new_sales], dtype=np.float32), reward, done, {}
    
env = PricingEnv(data)
state = env.reset()

for _ in range(len(data)):
    action = env.action_space.sample()
    next_state, reward, done, _ = env.step(action)
    print(f"Action: {action}, Next State: {next_state}, Reward: {reward}")