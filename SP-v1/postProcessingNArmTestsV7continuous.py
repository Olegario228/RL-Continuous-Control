#!/usr/bin/env python3 
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 12:48:21 2023

@author: Peter Manzl, UIBK
# description: variation of the number of tests 
                create fractal plot ...
"""
import numpy as np

from stable_baselines3 import A2C, PPO, DQN, SAC, TD3, DDPG
from exudynNLinkLibV2continuous import LoadAndVisualizeData, InvertedNPendulumEnv, PostProcessingTests, plotMyStates2D, setFigSize, SetMatplotlibSettings
from exudyn.processing import ParameterVariation, ProcessParameterList
import os
import sys



#%% 


factorCtlRate = 1
# Method = 'PPO' # 'PPO' # 'DQN' # loading DQN  model seems to not work properly
flagRand = True # randomize nTestRand times in each test +/- dx/2
flagDebug = True
flagFinished = False # will be set to true after processing parameters
strDescription = ''

listArguments = ['nArms', 'iVar', 'nTests1', 'nTests2', 'xState', 'yState', 'inputPath', 'nTestsRand']
listArgDesc = ['number of links from 1 (single pendulum) to 3 (triple pendulum)', 
        'the number of states wich are changed; e.g. for the single pendulum (nArms=1) iVar=[1,3] varies the link angle and link angular velocity', 
        'number of tests done in first dimension', 
        'number of tests done in second dimension', 
        'minimum and maximum value in which the grid cells are discretized in the first dimension, e.g. [-1, 1]', 'minimum and maximum value in which the grid cells are discretized in the second dimension, e.g. [-1,1]', 
        'the relative path to the saved agent the (without the *.zip postfix)',
        'number of tests done in each grid cell']
typeList = [int, list, int, int, np.array, np.array, str, int]

# listArguments = [1, [1, 3], 10, 10, np.array([-1, 1]), np.array([-1, 1]), "plots/PendulumA2C_r2_Model0_A2C", 10]
# python postProcessingNArmTestsV7continuous.py nArms=1 iVar=[1,3] nTests1=100 nTests2=100 xState=[-1.75,1.75] yState=[-6,6] inputPath=models/SinglePendulum_PPO_r2_Model0_PPO nTestsRand=20

conv = lambda dtype, inp : dtype(inp)

if '-h' in sys.argv: 
    print('\n' + '-'*20)
    print('This python script is used to evaluate the reliability (successes) of a previously trained RL agent in the inverted pendulum environment.') 
    print('For the calculation the following arguments are needed:')
    for arg, desc in zip(listArguments, listArgDesc): 
        print(arg, ': \t', desc)
    
    print('\n' + '-'*20 + '\n')
    sys.exit()
elif len(sys.argv) < len(listArguments) + 1: 
    print('Error: insufficient number of arguments provided. With the argument "-h" all needed arguments are listed using the help.')
    sys.exit()
else: 
    class param: pass
    myArgs = {}
    for arg in listArguments: 
        for i, myArg in enumerate(sys.argv): 
            if arg in myArg: 
                myArgs[arg] = myArg.split('=')[-1]
                continue
            
    for key, value  in myArgs.items(): 
        i = listArguments.index(key)
        myVal = conv(typeList[i], value)
        if typeList[i] == str: 
            myVal = str(value)
        elif typeList[i]: 
            myVal = eval(value)
           #  setattr(param,key,eval(value))
        
        setattr(param,key,myVal)
        if 0: print('i=', i , key, value, 'type: ', type(myVal), 'typeList[i] =' , typeList[i]) # only for debugging
        if key == 'inputPath': 
            if 'A2C' in value: Method='A2C'
            elif 'PPO' in value: Method='PPO'
            elif 'DQN' in value: Method='DQN'
            elif 'SAC' in value: Method='SAC'
            elif 'TD3' in value: Method='TD3'
            elif 'DDPG' in value: Method='DDPG'
            # print('inputPath = ', value)

# dirName = '../LaTex/modelFigures/'
param.xState += [(float(param.xState[1]) - float(param.xState[0])) / param.nTests1 ]
param.yState += [(float(param.yState[1]) - float(param.yState[0])) / param.nTests2 ]
param.xState = np.array(param.xState)
param.yState = np.array(param.yState)

strModelName = param.inputPath 
# strModelName = "C:/Users/z101841/Work/Machine Learning/Projects/RL application in Pendulum control/Continuous Control/v1/plots/SinglePendulum_A2C_r2_v1Model0_A2C"

if param.nArms == 1: cartForce = 12
elif param.nArms == 2: cartForce = 40
elif param.nArms == 3: cartForce = 60


#%% 
# inherit model for custom testModel function
class testModel(InvertedNPendulumEnv): 
    def TestModelState(self, numberOfSteps=500, model = None, useRenderer=True, sleepTime=0.01, startState = None, **kwargs): 
        if useRenderer and sleepTime != 0: 
            import time
        self.simulationSettings.solutionSettings.writeSolutionToFile = False

        self.useRenderer = useRenderer #set this true to show visualization
        self.flagNan = False
        
        # reset state to given startState: 
        self.state = startState
        observation = startState
        self.steps_beyond_done = None
        self.State2InitialValues()
        self.simulationSettings.timeIntegration.endTime = 0
        self.dynamicSolver.InitializeSolver(self.mbs, self.simulationSettings) #needed to update initial conditions
        if model is None: 
            raise ValueError('provide a RL agent model for the test')
        
        for _ in range(numberOfSteps):
            
            if np.isnan(observation).any(): 
                self.flagNan = True
                break # 
            try: 
                action, _state = model.predict(observation, deterministic=True)
            except ValueError: 
                print('valueError for startStep{}, observation={}, state={}'.format(startState, observation, self.state))
                self.flagNan = True
                # debug? 
            # self.state[0] = np.nan
            observation, reward, done, info = self.step(action)
            if done: 
                # print('done at step {} with state {}'.format(_ ,observation))
                break
            self.render()
            if self.mbs.GetRenderEngineStopFlag(): #user presses quit
                break
            if useRenderer and sleepTime!=0:
                time.sleep(sleepTime)        
        self.close()
        return
        
        
# load model and env globally to prevent multible loading for multiprocessing
# set environment variable; could also be done in the shell by export CUDA_VISIBLE_DEVICES=""
os.environ['CUDA_VISIBLE_DEVICES'] = ""  # prevent of trying to use GPU; the GPU does not work in the variations
if Method == 'A2C': 
    model = A2C.load(strModelName, device='cpu')
elif Method == 'PPO': 
    model = PPO.load(strModelName, device='cpu')
elif Method == 'DQN': 
    model = DQN.load(strModelName, device='cpu')
elif Method == 'SAC': 
    model = SAC.load(strModelName, device='cpu')
elif Method == 'TD3': 
    model = TD3.load(strModelName, device='cpu')
elif Method == 'DDPG': 
    model = DDPG.load(strModelName, device='cpu')

    
# print('create local env and load model, type={}'.format(model.device.type))
localEnv = testModel(nArms = param.nArms, 
                               # change it if you want to change the evaluated environment compared to the 
                               massArmFact = 1, massCartFact = 1, lengthFact = 1, #these factors only for testing!; 
                               relativeFriction = 0,
                               stepUpdateTime = 0.02 / factorCtlRate,  #  parameterFunctionData['stepUpdateTime'],
                               cartForce = cartForce, 
                               numActions = 2, 
                               thresholdFactor = 10 # bigger threshold Factor increses the range of the states in which they can be 
                               ) 

if param.nArms == 1: 
    testErrorMax = 0.2
    nStepsTest = 800
elif param.nArms == 2: 
    testErrorMax = 0.5
    nStepsTest = 800
elif param.nArms ==3: 
    testErrorMax = 0.75
    nStepsTest = 1200

paramSetGlobal = {'strModelName': strModelName, 'nSteps': nStepsTest*factorCtlRate, 'testErrorMax': testErrorMax, 
                  'flagVerbose': False, 'cartForce': cartForce, 'nArms': param.nArms, 'flagRand': flagRand, 
                  'state{}dx'.format(param.iVar[0]): param.xState[2],  'state{}dx'.format(param.iVar[1]): param.yState[2], 
                  'description': strDescription, 'nTestsRand': param.nTestsRand}

if flagDebug: 
    import psutil
    
def testFunction(parameterSet):
    parameterSet['functionData'] = paramSetGlobal
    success = 1
    error =  None
    # time.time()
  
    np.random.seed(parameterSet['computationIndex']) # todo
    iTest = 0
    nSuccess = 0
    maxError_ = 0
    errorCoordinates_ = np.array([0]*(param.nArms+1))
    while iTest < parameterSet['functionData']['nTestsRand']: 
        # print(iTest)
        state0 = []
        for i in range(2*param.nArms+2):  # put together state from dict
            strParam = 'state{}'.format(i)
            if strParam in parameterSet.keys(): 
                state0 += [parameterSet[strParam]]
                if parameterSet['functionData']['flagRand']:  # 
                    state0[-1] += parameterSet['functionData']['state{}dx'.format(i)] * (np.random.random()-0.5)
            else: 
                state0 += [0]
        try: 
            localEnv.TestModelState(numberOfSteps=parameterSet['functionData']['nSteps'], model=model, 
                          stopIfDone=False, useRenderer=False, sleepTime=0., showTimeSpent=False, startState=state0) 
            done = localEnv.Output2StateAndDone() 
            
            results = localEnv.mbs.GetSensorStoredData(localEnv.sCoordinates)
            iTest += 1
        except: 
            print('WARNING: an error occured in cIndex={}, step {}'.format(parameterSet['computationIndex'], iTest))
            continue
        if done: 
            if 0: print('failed with done=', done)    
            success = 0
            # errorCoordinates = [2*np.pi]*(nArms+1)
            # error = 2*np.pi
            # return success, error, list(errorCoordinates)
        if localEnv.flagNan: 
            print('nan occured for state0 {}, results={}'.format(state0, results))
        nStepsFinal = int(0.75*len(results))
        
        
        resultsCut = results[nStepsFinal:,:] * np.array([0] + [1]*(results.shape[1]-1))#  * [0 + 1]  +[0.4,2,2,2] # wheighing for test results
        iMax = np.argmax(np.abs(resultsCut[:,1:param.nArms+2]), axis=0)
        errorCoordinates = []
        for j, i in enumerate(iMax): 
            errorCoordinates += [resultsCut[:,1:param.nArms+2][i,j]]
        # np.unravel_index(np.argmax(resultsCut[:,1:nArms+2]), resultsCut[:,1:nArms+2].shape)
        maxError = np.max(errorCoordinates)
        if parameterSet['functionData']['flagVerbose']: 
            maxErrorPos = max(abs((results[nStepsFinal:,1])))
            iMaxErrPos = list(np.abs(results[:,1])).index(maxErrorPos)
            maxErrorAngle = np.max(abs((results[nStepsFinal:,2:param.nArms+2]))) #last arm angle; this should work even for nArms>
            imaxErrorAngle = np.argmax(np.abs(results[nStepsFinal:,2:param.nArms+2])) + nStepsFinal # list(np.abs(results[:,1:nArms+1])).index(maxErrorAngle)
            error = max(maxErrorPos, maxErrorAngle)
            resultsCut = results[nStepsFinal:,:] * np.array([0] + [1]*(results.shape[1]-1))#  * [0 + 1]  +[0.4,2,2,2] # wheighing for test results
            imaxError = np.unravel_index(np.argmax(resultsCut[:,1:param.nArms+2]), resultsCut[:,1:param.nArms+2].shape)
            imaxErrorCoordinate = imaxError[1] # coordinate in which max error occures
            imaxErrorIndex = imaxError[0] + nStepsFinal
            
        
        # posError
        if maxError > parameterSet['functionData']['testErrorMax'] or done:
            success = 0
        #     if parameterSet['functionData']['flagVerbose']: 
        #         print('failed: maxError={}, max error Index = {}, coordinate {}'.format(error, imaxErrorIndex, imaxErrorCoordinate))
        # else:
        #     if parameterSet['functionData']['flagVerbose']: 
        #         print('success: maxError={}, max error Index = {}, coordinate {}'.format(error, imaxErrorIndex, imaxErrorCoordinate))
        # if done and 0: print(maxError)
        if success: 
            nSuccess +=1
        if maxError_ < maxError: 
            maxError_ = maxError
        for i in range(len(errorCoordinates)): 
            if errorCoordinates_[i] < errorCoordinates[i]: 
                errorCoordinates_[i] = errorCoordinates[i]
    if flagDebug and not(parameterSet['computationIndex'] % 200):
        print('iComp={} \t - current (resident set size) memory is {}MB, vss {}MB'.format(parameterSet['computationIndex'], 
                round(psutil.Process().memory_info().rss / 1024**2, 2), round(psutil.Process().memory_info().vms / 1024**2, 2)), flush=True)
    # print('iComp={}, iTest={}, nSuccess={}'.format(parameterSet['computationIndex'], iTest, nSuccess))
    return nSuccess, maxError, errorCoordinates # , state0 # , errorsMaxCoordinates




#%%  ++++++++++++++++++++++++++++++++++
if __name__ == '__main__': 
    import datetime
    import time
    flagPlotError = True 
    fileName = param.inputPath.split('/')[-1] + '{}tests'.format(param.nTests1*param.nTests2*param.nTestsRand) + datetime.datetime.now().strftime("%Y_%m_%d_%Hh%Mm%Ss")
    # fileName = 'plots/SinglePendulum_A2C_r2_v1Results0'
    print('start Testing of RL-Agent, write into \n{}'.format(fileName))
    import os
    nTasks = os.cpu_count()
    try:
        nTasks = os.environ ['SLURM_NTASKS'] # from env variable $SLURM_NTASKS
        flagMPI=True
        nTasks = int(nTasks) - 1
        print('slurm tasks: {}'.format(nTasks)) # flagMPI = {}'.format(nTasks)# , flagMPI))
    except:
        flagMPI=True
        pass   
    #each run will be done with different random initialization
    
    start_time = time.time()
    
 
    paramList = []
    # paramSetGlobal = {'functionData': paramSetGlobal}
    l = 0
    for i in range(param.nTests1): 
        x = param.xState[0] + (param.xState[1]-param.xState[0])/(param.nTests1-1) * i
        xDict = {'state' + str(param.iVar[0]): x}
        for j in range(param.nTests2): 
            y = param.yState[0] + (param.yState[1]-param.yState[0])/(param.nTests2-1) * j  
            yDict = {'state' + str(param.iVar[1]): y}
            paramList += [{ **xDict, **yDict, **{'computationIndex': l}}]
            l+=1 # add computation index, currently not supported from processparameterList function (exudyn 1.6.31.dev1)
    # import sys
    # sys.exit()
    # testFunctionLambda = lambda parameterSet: testFunction(parameterSet, paramSetGlobal)
    print('start calculate {} tests, using {} tasks '.format(len(paramList), nTasks) +  'with MPI'*flagMPI + 'without MPI'*bool(not(flagMPI)))
    data = ProcessParameterList(parameterFunction=testFunction, 
                                            parameterList=paramList,
                                            addComputationIndex=True, 
                                            useMultiProcessing=True,
                                            numberOfThreads=int(nTasks), # *2 because of hyperthreading
                                            showProgress=False, 
                                            resultsFile= fileName + '.txt', 
                                            useMPI=flagMPI, 
                                            )
    
    val = []
    for i, tmp in enumerate(data): 
        val += [[]]
        for i in range(len(tmp)): 
            if type(tmp[i]) == list: 
                val[-1] += tmp[i]
            else: 
                val[-1] += [tmp[i]]

    val = np.array(val)
    dt = time.time() - start_time
    flagFinished = True
    print("--- %s seconds ---" % (dt))
    print("--- {} s/it ---".format(dt/len(paramList)))

#%% 
    SetMatplotlibSettings()
    try: 
        print(param)
    except: # param was not loaded
        class param: pass
        param.iVar = [1,3]
        flagMPI = False
    if not(flagMPI) or 1: # read from file
        
        import matplotlib.pyplot as plt

        # strLoad = 'NArmTest/DoubleA_400000tests2023_03_31_22h16m28s.txt'
        # strLoad = 'NArmTest/DoubleA_800000tests2023_04_04_15h52m38s.txt'
        # strLoad = 'NArmTest/SinglePendulumF_800000tests2023_04_07_14h11m16s.txt'
        # strLoad = 'NArmTest/SinglePendulumF_1600000tests2023_04_07_14h50m17s.txt'
        # strLoad = 'NArmTest/SinglePendulumF_800000testsPPO2023_04_07_15h35m31s.txt'
        # strLoad = 'NArmTest/DoubleB_3200000tests2023_04_07_16h50m13s.txt'
        # strLoad = 'NArmTest/DoubleB_3200000tests2023_04_11_10h59m00s.txt'
        # strLoad = 'NArmTest/DoubleB_3200000tests2023_04_11_17h53m58s.txt'
        # strLoad = 'NArmTest/DoubleB_1800000tests2023_04_12_08h59m00s.txt'
        # strLoad = 'NArmTest/DoubleB_1800000tests2023_04_12_12h41m41s.txt'
        # strLoad = 'NArmTest/DoubleB_10000000tests2023_04_12_17h12m53s.txt'
        # strLoad = 'NArmTest/SinglePendulumF_Model0_A2C25000tests2023_05_22_16h09m40s.txt'
        # strLoad = 'SinglePendulumF_Model4_A2C25000tests2023_05_22_16h14m10s.txt'
        
        # strLoad =  'NArmTest/TripleC_320000tests2023_04_11_13h58m48s.txt' # phi1, phi2
        # strLoad =  'NArmTest/TripleC_320000tests2023_04_11_14h06m37s.txt' # showing phi2phi3
        # strLoad =  'NArmTest/TripleC_800000tests2023_04_11_14h11m48s.txt'
        
        # strLoad =  'SinglePendulumF_Model4_A2C100000000tests2023_05_22_16h49m13s.txt'
        # strLoad =  'DoubleB_Model0_A2C100000000tests2023_05_22_17h37m58s.txt'
        # strLoad =  'DoubleB_Model1_A2C100000000tests2023_05_22_18h58m13s.txt'
        # strLoad =  'SinglePendulumF_Model0_A2C9000000tests2023_05_22_19h42m28s.txt'
        strLoad = 'SinglePendulum_A2C_r2_Model0_A2C200000tests2023_08_16_15h30m35s.txt'
        
        if flagMPI: strLoad = fileName + '.txt'
        # if not('NArmTest') in strLoad: strLoad = 'NArmTest/' + strLoad
        if 'Single' in strLoad: nArms = 1
        elif 'Double' in strLoad: nArms = 2
        elif 'Triple' in strLoad: nArms = 3 
        
        if nArms == 1: 
            stateStr = ['x', r'$\varphi_1$', r'$\dot{x}$', r'$\dot{\varphi}_1$']
        elif nArms == 2: 
            stateStr = ['x', r'$\varphi_1$', r'$\varphi_2$', r'$\dot{x}$', r'$\dot{\varphi}_1$', r'$\dot{\varphi}_2$']
        elif nArms == 3: 
            stateStr = ['x', r'$\varphi_1$', r'$\varphi_2$',r'$\varphi_3$', r'$\dot{x}$', r'$\dot{\varphi}_1$', r'$\dot{\varphi}_2$', r'$\dot{\varphi}_3$']
    
        if flagFinished: strLoad = fileName + '.txt'
        iTest_, iPlot_, successes_, valueList_, state0_ , nTestsRand_ = PostProcessingTests(strLoad, flagLegacyV5=False)
        state1 = set(np.round(state0_[:,0], 14))
        state2 = set(np.round(state0_[:,1], 14))
        
        states = [list(state1), list(state2)]
        for i in range(2): 
            states[i].sort()
        if len(iPlot_) == 0: 
            iPlot_ = param.iVar
        ls1, ls2 = len(state1), len(state2)
        plotMyStates2D(states, successes_,  iPlot_, nTestsRand_, stateStr=stateStr, flagRand=flagRand, flagDebug=False) # , pDict)
        plt.tight_layout()
        plt.subplots_adjust(top=0.95,bottom=0.2,left=0.14,right=0.97,hspace=0.2,wspace=0.2)
        plt.gca().invert_yaxis()
        plt.show()
        plt.savefig(strLoad[:-4]+ '_V3.png', dpi=1000)
        
        if 1:
            
            from matplotlib import patches 
            xSum = states[0][-1] - states[0][0]
            ySum = states[1][-1] - states[1][0]
            xMean = ls1 / 2
            yMean = ls2 / 2
            
            
            
            xDiffTrain = (0.1/ xSum) * ls1
            yDiffTrain =  (0.1/ ySum) * ls2

            plt.gca().add_patch(patches.Rectangle((xMean-xDiffTrain, yMean-yDiffTrain), 2*xDiffTrain, 2*yDiffTrain, alpha=1, linewidth = 1, 
                                                  edgecolor='red', fill=False, label='Train'))


            if nArms == 3: 
                xDiffTest = 0.06*ls1
                yDiffTest = 0.06*ls2
                plt.gca().add_patch(patches.Rectangle((xMean-xDiffTest, yMean-yDiffTest), 2*xDiffTest, 2*yDiffTest, alpha=1, 
                                                      ec='green', fill=False, label='Test'))
                
            if nArms == 2 and strLoad == 'NArmTest/DoubleB_100000000tests2023_04_17_11h08m32s.txt': 
                xMean2 = int((0.0345 + xSum/2)/(xSum) *ls1)
                yMean2 = int((0.0915 + xSum/2)/(ySum) *ls2)
                plt.gca().add_patch(patches.Rectangle((xMean2-5, yMean2-5), 2*5, 2*5, alpha=1,linewidth = 1, 
                                                      edgecolor='c', fill=False, label='Train'))
            plt.show()
            plt.savefig(strLoad[:-4] + '_Annotated_V3.png', dpi=1000)
    else: 
        print('Finished job at HPC')
        
    


# %%
