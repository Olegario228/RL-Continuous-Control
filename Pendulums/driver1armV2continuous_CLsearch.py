#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# This is an EXUDYN example
#
# Details:  This file does a parameter variation for Reinforcement-Learning of an 
#           inverted n-Link Pndulum on a cart. The standard-parameters are specified
#           in the parameterFunctionData. nCase can be used for statistical statements.
#           In the parameterfunction a multibody model of the inverten n-Link Pendulum is created using 
#           Exudyn in an OpenAIGymInterfaceEnv and learning the stabilization using stable-baselines 3 (version 1.5) 
#           The value computed in every parameter variation is the number of hightest number of successful tests reached.  
#
# Author:   Johannes Gerstmayr
# Date:     2023-01-23
#
# Copyright:This file is part of Exudyn. Exudyn is free software. You can redistribute it and/or modify it under the terms of the Exudyn license. See 'LICENSE.txt' for more details.
#
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# import sys
# sys.exudynFast = True #this variable is used to signal to load the fast exudyn module

import exudyn as exu
from exudyn.utilities import *
from exudyn.processing import ParameterVariation
from exudyn.plot import PlotSensorDefaults, PlotSensor
from exudyn.artificialIntelligence import *

import stable_baselines3.common.logger as sblog
sblog.configure('.', ['stdout']) #avoid logging too much

import re
import itertools
from decimal import Decimal
import sys
import numpy as np #for postprocessing
from math import pi
import scipy
# import time

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

from stable_baselines3 import A2C, PPO, DQN, SAC, TD3, DDPG

from exudynNLinkLibV2continuous import InvertedNPendulumEnv, EvaluateEnv, ParameterFunction, LoadAndVisualizeData, AppendVersionToResultFile


# --------- #

def parse_decay_steps(input_string):
    # Разделяем строку по запятым
    parts = input_string.split(',')

    # Результат будет списком списков
    result = []

    for part in parts:
        if '-' in part:
            # Для диапазона: находим числа и шаг
            numbers = list(map(int, re.findall(r'\d+', part)))
            if len(numbers) == 2:
                # Если шаг не указан, используем шаг 1
                range_values = list(range(numbers[0], numbers[1] + 1, 1))
            elif len(numbers) == 3:
                # Если шаг указан
                range_values = list(range(numbers[0], numbers[1] + 1, numbers[2]))
            result.append(range_values)
        else:
            # Для одиночного числа: преобразуем в число
            number = int(part.strip())
            result.append([number])

    return result

# ------------- #

def parse_ranges(input_string):
    # Разбиваем входную строку на отдельные диапазоны
    groups = input_string.split(', ')
    result = []

    for group in groups:
        subgroups = group.split(' ')
        subresult = []
        for subgroup in subgroups:
            numbers = list(map(int, re.findall(r'\d+', subgroup)))
            if '-' in subgroup:  # Если это диапазон
                range_values = list(range(numbers[0], numbers[1] + 1, numbers[2] if len(numbers) > 2 else 1))
                subresult.append(range_values)
            else:  # Если это одно число
                subresult.append(numbers)
        result.append(subresult)
    return result


def generate_collection_combinations(parsed_ranges):
    for product in itertools.product(*[itertools.product(*subgroup) for subgroup in parsed_ranges]):
        yield list(map(list, product))

def generate_decay_combinations(steps):
    # Предполагаем, что первый список содержит только одно значение
    first_value = steps[0][0]

    # Второй и третий списки могут иметь разную длину, поэтому мы берем максимальную длину из них
    max_length = max(len(steps[1]), len(steps[2]))

    # Генерируем комбинации
    for i in range(max_length):
        # Получаем значения из второго и третьего списков, учитывая их длину
        second_value = steps[1][min(i, len(steps[1]) - 1)]
        third_value = steps[2][min(i, len(steps[2]) - 1)]

        # Выводим комбинацию
        yield [first_value, second_value, third_value]


def generate_sequential_combinations(parsed_ranges):
    max_steps = min(len(r) for group in parsed_ranges for r in group)
    
    for step in range(max_steps):
        current_combination = []
        for group in parsed_ranges:
            current_group = []
            for range_values in group:
                # Добавляем в группу минимальное значение из range_values, увеличенное на step
                # Но не превышающее максимальное значение в range_values
                current_value = min(range_values[0] + step, range_values[-1])
                current_group.append(current_value)
            current_combination.append(current_group)
        yield current_combination

def parse_d_factor_range(range_string):
    numbers = [Decimal(n) for n in re.findall(r"\d+\.\d+", range_string)]
    
    if len(numbers) == 3:
        start, end, step = numbers
    elif len(numbers) == 2:
        start, end = numbers
        step = Decimal('0.0001')  # Значение шага по умолчанию

    return start, end, step

def generate_d_factor_values(start, end, step):
    current = start
    while current <= end:
        yield current
        current += step
#-------------- #

#the following totally overrides specific settings for single, parallelized runs (use with care!):
#these variables must be treated differently in __init__ of InvertedNPendulumEnv
nArms = 1 #number of inverted links ...
#set the following line for vectorized environments:
#InvertedNPendulumEnv.nArms = nArms #nArms cannot be passed to environment for multithreaded learning (substeps=-1), must be overwritten here!


#%%++++++++++++++++++++++++++++++++++++++++++++++++++
#now perform parameter variation
if __name__ == '__main__': #include this to enable parallel processing
    import time

    evaluationSteps = 1000
    episodeSteps = 1024    # SP - 1024; DP - 1229; TP = 1639
    outputName = 'models\Curriculum-Learning\SP\PPO_finalTest_v1' + '_'
    dirName = os.path.dirname(os.path.abspath(outputName))
    
    if not os.path.exists(dirName):
        print('WARNING: directory does not exist: "' + dirName + '", will try to create it')
        os.makedirs(dirName)
    
    parameterFunctionData = {'nArms': nArms,
                             'evaluationSteps': evaluationSteps,
                             'episodeSteps': episodeSteps, 
                             'episodeStepsMax': int(episodeSteps*1.25), #if episodes do not finish
                             'totalLearningSteps': int(0.035e6),  #max number of steps for total training
                             'logLevel': 3,  # 0=per step, 1=per rollout, 2=per episode, 3=per learn (higher is less logs!)
                             'lossThreshold': 1e-2,      # above that, no evaluation is performed
                             'rewardThreshold': 0.93,   # 0.95,    # above  that, no evaluation is performed (currently reward, not mean reward)
                             'meanSampleSize': 10,		#for computation of mean reward
                             'RLalgorithm': 'PPO',		#learning algorithm
                             'rewardMode': 2,			 #1=sum angles + position cart, 2=sumAngles + position tip, 3=last link angle + position tip
                             'rewardPositionFactor': 0.5, # % take a look on that (from 0.1)
                             'stepUpdateTime': 0.02,     #step size for single step
                             'thresholdFactor': 0.5,     # SP - 0.5, DP - 1.5; TP - 2.25
                             'cartForce': 12,			# SP - 12; DP - 40; TP - 60 # vertical Force acting on the cart for the control; needs to be increased for bigger model # 12 for single
                             'forceFactor': 1,
                              # 'randomInitializationValue': 0.15,          # 
                              # 'randomInitializationFactorTest': 2/3,       # a factor to scale the random initialization for testing in relation to the training 
                             'numberOfTests': 50,                # SP/DP - 50; TP - 100        # number of test evaluations; this number may be high to get high confidence, but may lead to larger evaluation time
                             'relativeFriction': 0*0.02, 
                             'storeBestModel': outputName + 'Model', 	  # add name here to store best model (model with the hightest number of successful tests)
                             'netarchLayerSize': 64,                      # default: 64, should be probably exp of 2; larger=slower
                             'netarchNumberOfLayers': 2,                   # default: 2, number of hidden layers, larger=slower, but may be more performant
                             'stopWhenAllTestsSuccess': True,             # default: True (V2) stop training when all Tests are were successful
                             'breakTestsWhenFailed': False,                # default: True (V2) when set true evaluation of tests is skipped when the first fails; can save computation time at costs of statistics
                             'nThreadsTraining': 1,                       # 1 for single run; >1: vectorized envs, will also change behavior significantly
                             'resultsFile': outputName + 'Results',       # local results file
                             'verbose': True,
                             'curicculumLearning': {'decayType': "exp", # exp | lin, quad, x^5, exp, discrete, sqrt 
                                                    'decaySteps': [0, 6000, 12000], # [0, 5000-10000, 8000 - 16000] ; learning steps at which to change to the next controlValues
                                                    'controlValues': [
                                                                      [2,8],  # in decayStep i the i-th row of controlValues is written to the [0-4; 4-10],                                                                           
                                                                      [0,4],
                                                                      [0,0]
                                                                      ], 
                                                    'dFactor': 0.0005 }, # 0.0005 | in Segment i: dControl[i] = controlValues[i] * dFactor
                             }

    if False: #just evaluate and test one full learningSteps period (with graphics)
        refVal = ParameterFunction({'functionData':parameterFunctionData}) # compute reference solution
    else:
        print('start variation:')
        start_time = time.time()
        
        nCases = 2 # repeat for statistics, parameters are unchanged

        bestNSteps = sys.maxsize
        bestValues = {}

        # params = {
        #     "decaySteps": {
        #         "variants": "0, 6000-10000(500), 10000-16000(1000)",
        #         "brut": True,  # Если True, то проходит все комбинации. Если False, то только пары комбинаций
        #     },
        #     "controlValues": {
        #         "variants": "2-4(1) 4-10(1), 0-0 1-9(1), 0-0 0-0",
        #         "brut": True,
        #     },
        #     "dFactor": {
        #         "variants": "0.0005-0.0010(0.0001)",
        #     },
        #     "decayType": {
        #         "variants": "exp, lin, quad, x^5, discrete, sqrt",
        #     },
        # }
        
        params = {
            "decaySteps": {
                "variants": "0, 6000-6500(500), 10000-11000(1000)",
                "brut": False,  # Если True, то проходит все комбинации. Если False, то только пары комбинаций
            },
            "controlValues": {
                "variants": "2-2(1) 4-4(1), 0-0 1-1(1), 0-0 0-0",
                "brut": False,
            },
            "dFactor": {
                "variants": "0.0005-0.0005(0.0001)",
            },
            "decayType": {
                "variants": "exp",
            },
        }

        parsed_decay_values = parse_decay_steps(params["decaySteps"]["variants"])
        parsed_control_values = parse_ranges(params["controlValues"]["variants"])
        parsed_range = parse_d_factor_range(params["dFactor"]["variants"])
        decay_types = params["decayType"]["variants"].split(", ")

        # Генерация комбинаций для decaySteps
        if params["decaySteps"]["brut"]:
            decay_combinations = itertools.product(*parsed_decay_values)
        else:
            decay_combinations = generate_decay_combinations(parsed_decay_values)


        if params["controlValues"]["brut"]:
            control_combinations = generate_collection_combinations(parsed_control_values)
        else:
            control_combinations = generate_sequential_combinations(parsed_control_values)

        control_combinations = list(control_combinations)
        decay_combinations = list(decay_combinations)

        # Перебор всех комбинаций
        for control_combination in control_combinations:
            for decay_combination in decay_combinations:
                for d_factor in generate_d_factor_values(parsed_range[0], parsed_range[1], parsed_range[2]):
                    for decay_type in decay_types:
                        parameterFunctionData["curicculumLearning"]["decaySteps"] = decay_combination
                        parameterFunctionData["curicculumLearning"]["controlValues"] = control_combination
                        parameterFunctionData["curicculumLearning"]["dFactor"] = float(d_factor)
                        parameterFunctionData["curicculumLearning"]["decayType"] = decay_type

                        [pDict, values] = ParameterVariation(parameterFunction=ParameterFunction,
                                                        parameters = { 
                                                                    #'rewardPositionFactor': [0.9, 0.8, 0.7],
                                                                    #'thresholdFactor':(0.5,1.,3), #standard is 0.75
                                                                    #'RLalgorithm':(0,2,3), #[0,1,2] == [A2C, PPO, DQN]
                                                                    #'learningRateFactor':(0.5,2,5), #factor on original learning rate
                                                                    #'testMassArmFact':(0.8,1.2,3),
                                                                    #'testMassCartFact':(0.8,1.2,3),
                                                                    #'lengthFact':(0.5,2,7),
                                                                    #'relativeFriction':(0.,0.01,2),
                                                                    #'cartForce': [50, 55, 60],
                                                                    # 'numActions': (2,3,2), 
                                                                    # 'randomInitializationFactorTest': (0.5, 1, 3), 
                                                                    # 'randomInitializationValue': (0.08, 0.1, 2), 
                                                                    # 'rewardMode': (0,3,4), # better give the values as a tuple, not a list; otherwise the postprocessing makes problems
                                                                    'case': (1, nCases, nCases), #check statistics, not varying parameters
                                                                    },
                                                        parameterFunctionData = parameterFunctionData,
                                                        debugMode=False,         #more output
                                                        #useLogSpace=True,       #geometric (log) interpolation of parameter values: True: 1-10-100, False: 1-50.5-100
                                                        resultsFile=outputName+'ResultsVar.txt',
                                                        numberOfThreads=20,       #this is the number of parallel threads that are used to perform computations; usually max. 2 x number of cores
                                                        addComputationIndex=True, #False writes out messages in serial mode
                                                        useMultiProcessing=True, #turn this on to use numberOfThreads threads
                                                        showProgress=True,
                                                        )

                        min_tuple = min(values, key=lambda x: x[1])
                        min_value = min_tuple[1]

                        if min_value < bestNSteps: 
                            bestNSteps = min_value
                            bestValues =  {  
                                            'decaySteps': decay_combination, 
                                            'controlValues': control_combination, 
                                            'dFactor': d_factor, 
                                            'decayType': decay_type
                                         }
                            AppendVersionToResultFile(outputName+'ResultsVar.txt')
                            print("--- %s seconds ---" % (time.time() - start_time))

    print('bestNSteps = ', bestNSteps, " with values: ", bestValues)
    
    #%%+++++++++++++++++++++++
    #load values from file
    if False:
        PlotSensorDefaults().fontSize = 10 #for changes to become active, restart iPython!
        PlotSensorDefaults().sizeInches=[6.4,4.8] #larger size gives higer resolution, but smaller font
        #for pdfs:
        # PlotSensorDefaults().fontSize = 10 #for changes to become active, restart iPython!
        # PlotSensorDefaults().sizeInches=[6.4,4.8] #larger size gives higer resolution, but smaller font
        #PlotSensorDefaults().markerSizes = 8 #for pdfs
        LoadAndVisualizeData(outputName+'Results', showFigure=True, exportFigure=True, showRewardPerEpisode=True,
                             showStepsPerEpisode=False, showLoss=True, outputName=outputName+'Results', caseInFileName=True,
                             caseInLegend=True, exportFileType='png')

    #%%++++++++++++++++++++++++++++++++++
    if False:
        #each run will be done with different random initialization
        try:
            model = A2C.load(outputName+'Model'+'0_'+'A2C') #0 for computation index, depends on number of variations
        except:
            print('no model available!')
            sys.exit()
            
        #test environment and learning environment need to be decoupled! (gives different results!)
        testEnv = InvertedNPendulumEnv(nArms=nArms, 
                                       #massArmFact=1, massCartFact=1, lengthFact=1, #these factors only for testing!
                                       relativeFriction = parameterFunctionData['relativeFriction'],
                                       stepUpdateTime = parameterFunctionData['stepUpdateTime'],
                                       cartForce = parameterFunctionData['cartForce'],
                                       ) 
        testEnv.randomInitializationValue = 0.1
        model.doLogging = False
        
        solutionFile='solution/learningCoordinates.txt'
        testEnv.simulationSettings.solutionSettings.sensorsWritePeriod = testEnv.stepUpdateTime
        testEnv.TestModel(numberOfSteps=500*2, model=model, solutionFileName=solutionFile, 
                      stopIfDone=False, useRenderer=False, sleepTime=0., showTimeSpent=False) 
        if True: #show model in solution viewer
            #visualize (and make animations) in exudyn:
            from exudyn.interactive import SolutionViewer
            testEnv.SC.visualizationSettings.general.autoFitScene = False
            solution = LoadSolutionFile(solutionFile)
            SolutionViewer(testEnv.mbs, solution, timeout=0.01) #loads solution file via name stored in mbs


