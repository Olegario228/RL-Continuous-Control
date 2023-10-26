import os
#import moviepy.video.io.ImageSequenceClip
import matplotlib.pyplot as plt

import exudyn as exu
from exudyn.utilities import *
from exudynNLinkLibV2continuous import InvertedNPendulumEnv
from stable_baselines3 import A2C, PPO, SAC, TD3, DDPG
from exudyn.interactive import SolutionViewer

# env initialization
localEnv = InvertedNPendulumEnv(nArms = 3,
                                massArmFact = 1, massCartFact = 1, lengthFact = 1,
                                relativeFriction = 0,
                                stepUpdateTime = 0.02,
                                cartForce = 0,
                                actionThreshold = 1,
                                forceFactor = 1,
                                curicculumLearning = {'decayType': 'quad', # lin, quad, x^5, exp, or discrete 
                                                    'decaySteps': [0, 30000, 60000, 150000], # learning steps at which to change to the next controlValues
                                                    'controlValues': [[10,10,10,10],  # in decayStep i the i-th row of controlValues is written to the 
                                                                      [0,2,1,0.5], 
                                                                      [0,0,1,0.5], 
                                                                      [0,0,0,0]], 
                                                    'dFactor': 0.05},
                                rewardMode = 2)

localEnv.randomInitializationValue = 0.1

# visualize the untrained multibody model
solFileName = 'solution.txt'

localEnv.TestModel(numberOfSteps = int(10/0.02), timesteps = 0.02, model = None,
                  useRenderer = True, sleepTime = 0, showTimeSpent = False,
                  saveAnimation = False, solutionFileName = solFileName)

sol = LoadSolutionFile('solution.txt')
SolutionViewer(localEnv.mbs, sol)

# load the trained model
model = PPO.load('models\TP_PPO_r2_n64_v2_Model0_PPO_186368.zip')

# test the trained model and visualize it
localEnv.TestModel(numberOfSteps = int(10/0.02), timesteps = 0.02, model = model,
                  useRenderer = True, sleepTime = 0, showTimeSpent = False,
                  saveAnimation = False, solutionFileName = solFileName)

sol = LoadSolutionFile('solution.txt')
SolutionViewer(localEnv.mbs, sol)

image_folder='images'
fps=30

image_files = [os.path.join(image_folder, img)
               for img in os.listdir(image_folder)
               if img.endswith(".png")]
clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(image_files, fps=fps)
clip.write_videofile('TP_PPO_186k_79tests_r2.mp4')