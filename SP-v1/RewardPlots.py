import re
import matplotlib.pyplot as plt

file_path_0 = "models/SP_A2C_r2_v3_Results0.txt"
#file_path_1 = "C:/Users/z101841/Work/Machine Learning/Projects/RL application in Pendulum control/Continuous Control/v1/models/SinglePendulum_SAC_r2_Results1.txt"
#file_path_2 = "C:/Users/z101841/Work/Machine Learning/Projects/RL application in Pendulum control/Continuous Control/v1/models/SinglePendulum_SAC_r2_Results2.txt"

total_steps_0 = []
reward_per_episode_0 = []
total_steps_1 = []
reward_per_episode_1 = []
total_steps_2 = []
reward_per_episode_2 = []

with open(file_path_0, 'r') as file:
    lines = file.readlines()
    for line in lines:
        total_steps_match = re.search(r"'totalSteps': (\d+),", line)
        reward_match = re.search(r"'rewardPerEpisode': ([\d.]+)", line)

        if total_steps_match:
            total_steps_0.append(int(total_steps_match.group(1)))
        
        if reward_match:
            reward_per_episode_0.append(float(reward_match.group(1)))
            
# with open(file_path_1, 'r') as file:
#     lines = file.readlines()[6:-1]
#     for line in lines:
#         total_steps_match = re.search(r"'totalSteps': (\d+),", line)
#         reward_match = re.search(r"'rewardPerEpisode': ([\d.]+)", line)

#         if total_steps_match:
#             total_steps_1.append(int(total_steps_match.group(1)))
        
#         if reward_match:
#             reward_per_episode_1.append(float(reward_match.group(1)))
            
# with open(file_path_2, 'r') as file:
#     lines = file.readlines()[6:-1]
#     for line in lines:
#         total_steps_match = re.search(r"'totalSteps': (\d+),", line)
#         reward_match = re.search(r"'rewardPerEpisode': ([\d.]+)", line)

#         if total_steps_match:
#             total_steps_2.append(int(total_steps_match.group(1)))
        
#         if reward_match:
#             reward_per_episode_2.append(float(reward_match.group(1)))    
   
plt.figure(figsize=(10,6))
ax = plt.subplot(1, 1, 1)

# Plot the data
plt.plot(total_steps_0, reward_per_episode_0, label='case 1')
# plt.plot(total_steps_1, reward_per_episode_1, label='case 2')
# plt.plot(total_steps_2, reward_per_episode_2, label='case 3')

# Add labels and title
plt.xlabel('Total Training Steps')
plt.ylabel('Reward per Episode')
plt.legend()
plt.show()