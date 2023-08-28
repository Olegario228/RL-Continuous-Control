from exudynNLinkLibV2continuous import LoadAndVisualizeData, InvertedNPendulumEnv, PostProcessingTests, plotMyStates2D, setFigSize, SetMatplotlibSettings
import matplotlib.pyplot as plt
from matplotlib import patches
import numpy as np

class param: pass
param.iVar = [1,3]
# param.nArms = 1
# param.nTests1 = 100
# param.nTests2 = 100
# param.xState = [-1.75, 1,75]
# param.yState = [-6, 6]
#param.inputPath = 


flagMPI = False

# SetMatplotlibSettings()

strLoad = 'stability_zones/SinglePendulum_A2C_r2_v1Model1_A2C200000tests2023_08_16_14h50m08s.txt'

flagRand = True        

if 'Single' in strLoad: nArms = 1
elif 'Double' in strLoad: nArms = 2
elif 'Triple' in strLoad: nArms = 3 
        
if nArms == 1: 
        stateStr = ['x', r'$\varphi_1$', r'$\dot{x}$', r'$\dot{\varphi}_1$']
elif nArms == 2: 
        stateStr = ['x', r'$\varphi_1$', r'$\varphi_2$', r'$\dot{x}$', r'$\dot{\varphi}_1$', r'$\dot{\varphi}_2$']
        param.iVar = [1,2]
elif nArms == 3: 
        stateStr = ['x', r'$\varphi_1$', r'$\varphi_2$',r'$\varphi_3$', r'$\dot{x}$', r'$\dot{\varphi}_1$', r'$\dot{\varphi}_2$', r'$\dot{\varphi}_3$']
        param.iVar = [1,2]

iTest_, iPlot_, successes_, valueList_, state0_ , nTestsRand_ = PostProcessingTests(strLoad, flagLegacyV5=False)
state1 = set(np.round(state0_[:,0], 14))
state2 = set(np.round(state0_[:,1], 14))
        
states = [list(state1), list(state2)]
for i in range(2): 
    states[i].sort()
if len(iPlot_) == 0: 
    iPlot_ = param.iVar
ls1, ls2 = len(state1), len(state2)
data =  plotMyStates2D(states, successes_,  iPlot_, nTestsRand_, stateStr=stateStr, flagRand=flagRand, flagDebug=False) # , pDict)
        
stabilityFig = data[-1]
stabilityFig.tight_layout()
stabilityFig.subplots_adjust(top=0.95,bottom=0.2,left=0.14,right=0.97,hspace=0.2,wspace=0.2)
stabilityFig.savefig(strLoad[:-4]+ '_V3.png', dpi=1000)
        
        
if 1:  
    # xSum, Mean etc. are in pixels; thereby
    xSum = states[0][-1] - states[0][0]
    ySum = states[1][-1] - states[1][0]
    xMean = ls1 / 2
    yMean = ls2 / 2
    

    
    xDiffTrain = (0.1 / xSum) * ls1
    yDiffTrain =  (0.1 / ySum) * ls2
    stabilityFig.gca().add_patch(patches.Rectangle((xMean-xDiffTrain, yMean-yDiffTrain), 2*xDiffTrain, 2*yDiffTrain, alpha=1,linewidth = 1, 
                                        edgecolor='red', fill=False, label='Train'))

    # for nArms = 3 we tested with different intialization (0.06) 
    if nArms == 3: 
        xDiffTest = 0.06*ls1
        yDiffTest = 0.06*ls2
        stabilityFig.gca().add_patch(patches.Rectangle((xMean-xDiffTest, yMean-yDiffTest), 2*xDiffTest, 2*yDiffTest, alpha=1, 
                                            ec='green', fill=False, label='Test'))
    # in this specific case we want to highlight a specific region  
    if nArms == 2 and strLoad == 'NArmTest/DoubleB_100000000tests2023_04_17_11h08m32s.txt': 
        xMean2 = int((0.0345 + xSum/2)/(xSum) *ls1)
        yMean2 = int((0.0915 + xSum/2)/(ySum) *ls2)
        stabilityFig.gca().add_patch(patches.Rectangle((xMean2-5, yMean2-5), 2*5, 2*5, alpha=1,linewidth = 1, 
                                            edgecolor='c', fill=False, label='Train'))
    stabilityFig.savefig(strLoad[:-4] + '_Annotated_V3.png', dpi=1000)
else: 
    print('Finished job at HPC')
