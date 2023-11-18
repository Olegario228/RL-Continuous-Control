import os
#import moviepy.video.io.ImageSequenceClip
import matplotlib.pyplot as plt

import exudyn as exu
from exudyn.utilities import *
from exudynNLinkLibV2continuous import InvertedNPendulumEnv
from stable_baselines3 import A2C, PPO, SAC, TD3, DDPG
from exudyn.interactive import SolutionViewer
import moviepy
import moviepy.video.io.ImageSequenceClip

# env initialization
localEnv = InvertedNPendulumEnv(nArms = 3,
                                massArmFact = 1, massCartFact = 1, lengthFact = 1,
                                relativeFriction = 0,
                                stepUpdateTime = 0.02,
                                cartForce = 60,
                                actionThreshold = 1,
                                forceFactor = 1,
                                rewardMode = 2)

localEnv.randomInitializationValue = 0.1

# visualize the untrained multibody model
solFileName = 'solution.txt'

localEnv.TestModel(numberOfSteps = int(10/0.02), timesteps = 0.02, model = None,
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
clip.write_videofile('TP_untrained.mp4')

# load the trained model
model = PPO.load('models\Curriculum-Learning\TP\PPO_exp_v0_rpf09_Model1_PPO_253952_91_tests.zip')

# test the trained model and visualize it
localEnv.TestModel(numberOfSteps = int(10/0.02), timesteps = 0.02, model = model,
                  useRenderer = True, sleepTime = 0, showTimeSpent = False,
                  saveAnimation = False, solutionFileName = solFileName)

sol = LoadSolutionFile('solution.txt')
SolutionViewer(localEnv.mbs, sol)

# image_folder='images'
# fps=30

# image_files = [os.path.join(image_folder, img)
#                for img in os.listdir(image_folder)
#                if img.endswith(".png")]
# clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(image_files, fps=fps)
# clip.write_videofile('TP_trained_PPO_253952_91_tests.mp4')