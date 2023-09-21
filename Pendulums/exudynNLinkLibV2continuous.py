#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# This is an EXUDYN example
#
# Details:  This contains the n-link inverted pendulum on a cart for 
#           Reinforcement Learning of the stabilization. 
#           It uses an Exudyn multibody model inside an OpenAIGymInterfaceEnv. 
#           The stable-baselines 3 (version 1.5) learning algorithm is used. 
#           This module is a library and should be called from the driver file. 
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
from exudyn.robotics import *
from exudyn.artificialIntelligence import *
# import math

from stable_baselines3 import A2C, PPO, DQN, SAC, TD3, DDPG
# import stable_baselines3.common.logger as sblog
from stable_baselines3.common.callbacks import BaseCallback
#from stable_baselines3.common.callbacks import StopTrainingOnMaxEpisodes

# sblog.configure('.', ['stdout']) #avoid logging too much

import sys
import numpy as np #for postprocessing
from math import pi, sin, cos
import time
import json

import torch 
if torch.get_num_threads() > 1: 
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1) # 

import numpy as np
import random
# import os 
# os.environ['PYTHONASHSEED'] = '0' 

# import os
# os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

#%%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#use openAI interface class and overload
class InvertedNPendulumEnv(OpenAIGymInterfaceEnv):

    #initialize parameters and store in class instance    
    def __init__(self, nArms = 1, massarm = 0.1, length = 1., 
                 thresholdFactor = 0.75, cartForce = 10., stepUpdateTime = 0.02,
                 rewardPositionFactor = 0.5,
                 massArmFact = 1, massCartFact = 1, lengthFact = 1, #these factors are applied in evaluation
                 relativeFriction = 0., #link friction will be multiplied with g*m*0.5*length, such that 1 would always stick 
                 rewardMode = 1, #1=sum angles + position cart, 2=sumAngles + position tip, 3=last link angle + position tip
                 numActions = 2,   #if set to 3 the "zero action" is included
                 forceFactor = 1,
                 actionThreshold = 1, 
                 **kwargs):

        #+++++++++++++++++++++++++++++++++++++++++++++
        #these args may already exist as static variables; then we do not override:
        if 'nArms' not in self.__dir__(): self.nArms = nArms
        if 'thresholdFactor' not in self.__dir__(): self.thresholdFactor = thresholdFactor
        if 'cartForce' not in self.__dir__(): self.cartForce = cartForce
        if 'stepUpdateTime' not in self.__dir__(): self.stepUpdateTime = stepUpdateTime

        #+++++++++++++++++++++++++++++++++++++++++++++
        self.masscart = 1.*massCartFact
        self.massarm = massarm*massArmFact
        self.length = length*lengthFact
        self.relativeFriction = relativeFriction #friction at link joints (not at cart)
        self.rewardMode = rewardMode
        self.numActions = numActions
        self.forceFactor = forceFactor
        self.actionThreshold = actionThreshold
        
        self.rewardStep = 0 #make it accessible from outside
        self.rewardPositionFactor = rewardPositionFactor
        super(InvertedNPendulumEnv, self).__init__(**kwargs)
        
    
    #**classFunction: OVERRIDE this function to create multibody system mbs and setup simulationSettings; call Assemble() at the end!
    #                 you may also change SC.visualizationSettings() individually; kwargs may be used for special setup
    def CreateMBS(self, SC, mbs, simulationSettings, **kwargs):

        #%%++++++++++++++++++++++++++++++++++++++++++++++
        #this model uses kwargs: thresholdFactor
        
        # global nArms
        #self.nArms = nArms
        self.nTotalLinks = self.nArms+1 #nArms=number of arms, nArms=bodies (cart+pendulums)
        
        gravity = 9.81                  #gravity on all bodies
        #self.length = 1.
        width = 0.1*self.length         #for drawing
        massarm = self.massarm          
        # initially length**2*0.5*massArm; should probably be either (length/2)**2*massArm (pointmass) or length**2*massArm/3 (thin rod)
        armInertia = (self.length)**2*massarm/3

        L = self.length
        w = width
        gravity3D = [0.,-gravity,0]

        #added friction in arm joints:
        self.dynamicFrictionTorque = self.relativeFriction * self.length*0.5 * self.massarm * gravity
        # print('friction torque=', self.dynamicFrictionTorque)

        background = GraphicsDataCheckerBoard(point= [0,0.5*self.length,-0.5*width], 
                                              normal= [0,0,1], size=6, size2=3, nTiles=10, nTiles2=4)
        
        oGround=self.mbs.AddObject(ObjectGround(referencePosition= [0,0,0],  #x-pos,y-pos,angle
                                           visualization=VObjectGround(graphicsData= [background])))


        graphicsBaseList = [GraphicsDataOrthoCubePoint(size=[L*4, 0.8*w, 0.8*w], color=color4grey)] #rail
        
        newRobot = Robot(gravity=gravity3D,
                      base = RobotBase(visualization=VRobotBase(graphicsData=graphicsBaseList)),
                      tool = RobotTool(HT=HTtranslate([0,0.5*L,0]), visualization=VRobotTool(graphicsData=[
                          GraphicsDataOrthoCubePoint(size=[w, L, w], color=color4orange)])),
                      referenceConfiguration = []) #referenceConfiguration created with 0s automatically
        
        #cart:
        inertiaLink = RigidBodyInertia(self.masscart, np.diag([0.1*self.masscart,0.1*self.masscart,0.1*self.masscart]), [0,0,0])
        link = RobotLink(inertiaLink.Mass(), inertiaLink.COM(), inertiaLink.InertiaCOM(), 
                         jointType='Px', preHT=HT0(), 
                         # PDcontrol=(pControl, dControl),
                         visualization=VRobotLink(linkColor=color4lawngreen))
        newRobot.AddLink(link)
    
        
        #JG: check creation of inertia parameters
        for i in range(self.nArms):
            inertiaLink = RigidBodyInertia(massarm, np.diag([armInertia,0.1*armInertia,armInertia]), [0,0.5*L,0]) #only inertia_ZZ is important
            #inertiaLink = inertiaLink.Translated([0,0.5*L,0])
            preHT = HT0()
            if i > 0:
                preHT = HTtranslateY(L)
    
            link = RobotLink(inertiaLink.Mass(), inertiaLink.COM(), inertiaLink.InertiaCOM(), 
                             jointType='Rz', preHT=preHT, 
                             #PDcontrol=(pControl, dControl),
                             visualization=VRobotLink(linkColor=color4blue))
            newRobot.AddLink(link)
        
        self.inertiaLink = inertiaLink
        
        
        dKT = newRobot.CreateKinematicTree(mbs)
        self.oKT = dKT['objectKinematicTree']
        self.nKT = dKT['nodeGeneric']

        self.sCoordinates = mbs.AddSensor(SensorNode(nodeNumber=self.nKT, storeInternal=True,
                                          outputVariableType=exu.OutputVariableType.Coordinates))  
        
        #control force
        mCartCoordX = self.mbs.AddMarker(MarkerNodeCoordinate(nodeNumber=self.nKT, coordinate=0))
        self.lControl = self.mbs.AddLoad(LoadCoordinate(markerNumber=mCartCoordX, load=0.))
        
        # trying to extract action from the sensor     
        self.sAction = mbs.AddSensor(SensorLoad(loadNumber=self.lControl, writeToFile=True, storeInternal=True))
        
        if self.dynamicFrictionTorque != 0:
            nGround = mbs.AddNode(NodePointGround()) #for ground coordinate marker
            mGround = mbs.AddMarker(MarkerNodeCoordinate(nodeNumber=nGround, coordinate=0))
            mCoordsList = [mGround] #first link friction acts against ground
            
            for i in range(self.nArms):
                mLink = self.mbs.AddMarker(MarkerNodeCoordinate(nodeNumber=self.nKT, coordinate=1+i)) #0=cart
                mCoordsList += [mLink]
                
                #with data node, initialization would be more involved!
                #nGeneric = mbs.AddNode(NodeGenericData(initialCoordinates=[1,0,0],numberOfDataCoordinates=3))
                mbs.AddObject(CoordinateSpringDamperExt(markerNumbers=[mGround,mCoordsList[i+1]],
                                                        fDynamicFriction=self.dynamicFrictionTorque,
                                                        frictionProportionalZone=5e-3, #rad/s
                                                        # nodeNumber=nGeneric,
                                                        # stickingStiffness = 1e5, stickingDamping=5e2, #bristle stiffness and damping
                                                        #fStaticFrictionOffset = 0.25*self.dynamicFrictionTorque,
                                                        ))
        
        
        #%%++++++++++++++++++++++++
        self.mbs.Assemble() #computes initial vector
        
        self.simulationSettings.timeIntegration.numberOfSteps = 1
        self.simulationSettings.timeIntegration.endTime = 0 #will be overwritten in step
        self.simulationSettings.timeIntegration.verboseMode = 0
        self.simulationSettings.solutionSettings.writeSolutionToFile = False
        self.simulationSettings.solutionSettings.sensorsWritePeriod = 1000*self.stepUpdateTime #essentially do not write
        
        self.simulationSettings.timeIntegration.newton.useModifiedNewton = True
        self.simulationSettings.timeIntegration.generalizedAlpha.spectralRadius = 1 #no numerical damping
        
        self.SC.visualizationSettings.general.drawWorldBasis=True
        self.SC.visualizationSettings.general.graphicsUpdateInterval = 0.01
        self.SC.visualizationSettings.openGL.multiSampling=4
        
        #self.simulationSettings.solutionSettings.solutionInformation = "Open AI gym"
        
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++
        # Angle at which to fail the episode
        # these parameters are used in subfunctions
        self.theta_threshold_radians = self.thresholdFactor* 12 * (2 * pi / 360)
        self.x_threshold = self.thresholdFactor*2.4
        
        #must return state size
        stateSize = (self.nTotalLinks)*2 #the number of states (position/velocity that are used by learning algorithm)
        return stateSize

    #**classFunction: OVERRIDE this function to set up self.action_space and self.observation_space
    def SetupSpaces(self):

        high = np.array(
            [
                self.x_threshold * 2, #TODO: check meaning of 2: should we increase in case of TriplePendulum?
            ] +
            [
                self.theta_threshold_radians * 2,
            ] * self.nArms +
            [                
                # % for the velocities the observation space is  not restricted
                np.finfo(np.float32).max, 
            ] * self.nTotalLinks
            ,
            dtype=np.float32,
        )
        
        #print('high=',high)

        #+++++++++++++++++++++++++++++++++++++++++++++++++++++
        #see https://github.com/openai/gym/blob/64b4b31d8245f6972b3d37270faf69b74908a67d/gym/core.py#L16
        #for Env:
        self.action_space = spaces.Box(-self.actionThreshold, self.actionThreshold, (1,), dtype=np.float32) # +/- force
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)  
        # self.actionHist = []
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++


    #**classFunction: OVERRIDE this function to map the action given by learning algorithm to the multibody system, e.g. as a load parameter
    def MapAction2MBS(self, action):
        force = action * self.cartForce * self.forceFactor
        self.mbs.SetLoadParameter(self.lControl, 'load', force)

    #**classFunction: OVERRIDE this function to collect output of simulation and map to self.state tuple
    #**output: return bool done which contains information if system state is outside valid range
    def Output2StateAndDone(self):
        
        #+++++++++++++++++++++++++
        statesVector =  self.mbs.GetNodeOutput(self.nKT, variableType=exu.OutputVariableType.Coordinates)
        statesVector_t =  self.mbs.GetNodeOutput(self.nKT, variableType=exu.OutputVariableType.Coordinates_t)
        self.state = tuple(list(statesVector) + list(statesVector_t)) #sorting different from previous implementation
        cartPosX = statesVector[0]

        done = bool(
            cartPosX < -self.x_threshold
            or cartPosX > self.x_threshold
            or max(statesVector[1:self.nTotalLinks]) > self.theta_threshold_radians 
            or min(statesVector[1:self.nTotalLinks]) < -self.theta_threshold_radians 
            )
        
        return done

    
    #**classFunction: OVERRIDE this function to maps the current state to mbs initial values
    #**output: return [initialValues, initialValues\_t] where initialValues[\_t] are ODE2 vectors of coordinates[\_t] for the mbs
    def State2InitialValues(self):
        #+++++++++++++++++++++++++++++++++++++++++++++
        initialValues = self.state[0:self.nTotalLinks]
        initialValues_t = self.state[self.nTotalLinks:]
        # print('initial values=',initialValues)
        return [initialValues,initialValues_t]
        


    #**classFunction: openAI gym interface function which is called to compute one step
    def step(self, action):
        err_msg = f"{action!r} ({type(action)}) invalid"
        assert self.action_space.contains(action), err_msg
        assert self.state is not None, "Call reset before using step method."
        
        #++++++++++++++++++++++++++++++++++++++++++++++++++
        #main steps:
        [initialValues,initialValues_t] = self.State2InitialValues()
        self.mbs.systemData.SetODE2Coordinates(initialValues, exu.ConfigurationType.Initial)
        self.mbs.systemData.SetODE2Coordinates_t(initialValues_t, exu.ConfigurationType.Initial)

        self.MapAction2MBS(action)
        
        #this may be time consuming for larger models!
        self.IntegrateStep()
        
        done = self.Output2StateAndDone()

        #++++++++++++++++++++++++++++++++++++++++++++++++++
        #compute rewardStep and done
        cartPosX  = self.state[0]
        #angle  = self.state[1]
        
        #original, all angles equal:
        # angle  = abs(np.array(self.state[1:self.nTotalLinks])).sum() #.max() or .sum(); good question, if sum would be better as it reflects better the dependency on parameters
        # rewardCart = 1 - self.rewardPositionFactor*abs(cartPosX)/self.x_threshold - (1-self.rewardPositionFactor)*abs(angle)/self.theta_threshold_radians
        #new: top angle at higher priority
        sumAngles  = abs(np.array(self.state[1:self.nTotalLinks])).sum() #.max() or .sum(); good question, if sum would be better as it reflects better the dependency on parameters
        factAngle = self.nArms # we have to divide the sum of angles angle by the number of arms

        sumAngles = sumAngles/factAngle
        if self.rewardMode == 0: # rewardmode from openAIGym; reward is given until the environment leaves the allowed observation space
            rewardCart = 1 
            
        elif self.rewardMode == -1: # this reward is from the OpanAIGym inverted pendulum for the swingup; does not really work 
            # torque = self.mbs.GetLoadParameter(self.lControl, 'load') # read from MBS
            vCart = self.state[self.nTotalLinks]
            theta =  self.state[1:self.nTotalLinks]
            theta_t = self.state[1+self.nTotalLinks:]
            rewardCart = - (vCart**2*0.1) # - 0.01*torque**2 * 0
            for i in range(len(theta)): 
                rewardCart -= theta[i]**2 + theta_t[i]**2
            
            # rewardCart = -(theta**2 + 0.1 * theta_dt**2 + 0.001 * torque**2)
        elif self.rewardMode == 1: # rewardmode 
            rewardCart = 1 - self.rewardPositionFactor*abs(cartPosX)/self.x_threshold - (1-self.rewardPositionFactor)*sumAngles/self.theta_threshold_radians

        elif self.rewardMode == 2 or self.rewardMode == 3:
            angles = self.state[1:self.nTotalLinks]
            L = self.length
            px = cartPosX
            py = 0
            angleLast = 0
            #compute last arm tip position (px,py)
            for i in range(self.nArms):
                px -= L*sin(angles[i] + angleLast) #check sign / positive rotation of angle
                py += L*cos(angles[i] + angleLast)
                angleLast += angles[i]

            if self.rewardMode == 2:
                if 0: # depreciated, bad performance 
                    rewardCart = 1. - (1.-py/(L*self.nArms) )/self.theta_threshold_radians - abs(px/self.x_threshold) 
                else: 
                    # Peter: I was testing different weighting for the errors produced by the angles; it is not working better (yet)
                    # now it is the syame as rewardMode1 but with px instead of cartPosX
                    rewardCart = 1 - (1-self.rewardPositionFactor)*sumAngles/self.theta_threshold_radians \
                                      - self.rewardPositionFactor*abs(px)/self.x_threshold
                    # rewardCart = 1. - (1.-self.rewardPositionFactor) * (1-py/(L*self.nArms))self.theta_threshold_radians \
                    #                  -     self.rewardPositionFactor *  abs(px/self.x_threshold)
                    # rewardCart = 1 - 5*np.sqrt((py-L*self.nArms)**2 + px**2)/self.x_threshold # deviation of the tip 
            
            
            elif self.rewardMode == 3: 
                rewardCart = (1. - (1-self.rewardPositionFactor)*abs(angles[-1])/self.theta_threshold_radians 
                              - self.rewardPositionFactor*abs(px/self.x_threshold) )
                
        elif self.rewardMode == 4:
            cart_pos = cartPosX
            angle_1 = self.state[1]
            angle_2 = self.state[2]
            distance = abs(cart_pos) + abs(angle_1) + abs(angle_2)
            rewardCart = max(0.0, 1.0 - distance)
        
        elif self.rewardMode == 5:
            v_angle1 = self.state[4]
            v_angle2 = self.state[5]
            v_cart = self.state[3]
            total_energy = (v_angle1 ** 2) + (v_angle2 ** 2) + (v_cart ** 2)
            rewardCart = 1.0 / (1.0 + total_energy)
        
        elif self.rewardMode == 6:
            angle_1 = self.state[1]
            angle_2 = self.state[2]
            angle_penalty = -0.1 * (abs(angle_1) + abs(angle_2))
            rewardCart = 1.0 + angle_penalty
        
        elif self.rewardMode == 7:
            control_penalty = -0.001 * action ** 2
            rewardCart = 1.0 + control_penalty
        
        elif self.rewardMode == 8:
            w1 = 0.5
            w2 = 0.3
            w3 = 0.2
            cart_pos = cartPosX
            angle_1 = self.state[1]
            angle_2 = self.state[2]
            v_angle1 = self.state[4]
            v_angle2 = self.state[5]
            v_cart = self.state[3]
            distance = abs(cart_pos) + abs(angle_1) + abs(angle_2)
            rewardCart_1 = max(0.0, 1.0 - distance)
            total_energy = (v_angle1 ** 2) + (v_angle2 ** 2) + (v_cart ** 2)
            rewardCart_2 = 1.0 / (1.0 + total_energy)
            angle_penalty = -0.1 * (abs(angle_1) + abs(angle_2))
            rewardCart_3 = 1.0 + angle_penalty
            rewardCart = w1 * rewardCart_1 + w2 * rewardCart_2 + w3 * rewardCart_3
        
        elif self.rewardMode == 9:
            theta_1 = self.state[1]
            theta_2 = self.state[2]
            theta_3 = self.state[3]
            dtheta_1 = self.state[5]
            dtheta_2 = self.state[6]
            dtheta_3 = self.state[7]
            
            cart_position = cartPosX
            cart_velocity = self.state[4]
            
            target_position = 0.0
            target_angle = [0.0, 0.0, 0.0]  
            target_velocity = 0.0  
            target_angular_velocity = [0.0, 0.0, 0.0]
            
            position_weight = 0.3
            angle_weight = [0.2, 0.2, 0.2]
            velocity_weight = 0.3
            angular_velocity_weight = [0.2, 0.2, 0.2]
            
            # Calculate the distance from the cart to the target position
            position_error = abs(cart_position - target_position)

            # Calculate the deviation of each pendulum link from its target angle
            angle_error = [np.abs(theta_1 - target_angle[0]), np.abs(theta_2 - target_angle[1]), np.abs(theta_3 - target_angle[2])]

            # Calculate the deviation of cart velocity from the target velocity
            velocity_error = abs(cart_velocity - target_velocity)
        
            angular_velocity_error = [np.abs(dtheta_1 - target_angular_velocity[0]), np.abs(dtheta_2 - target_angular_velocity[1]), np.abs(dtheta_3 - target_angular_velocity[2])]
        
            reward = 1.0 - (
                    position_weight * position_error
                    + np.dot(angle_weight, angle_error)
                    + velocity_weight * velocity_error
                    + np.dot(angular_velocity_weight, angular_velocity_error)
                    )
            
            rewardCart = np.clip(reward, 0, 1)
        
        
        else: 
            raise ValueError('rewardMode unknown! Choose from mode 0 to 3.')

        if not done:
            self.rewardStep = rewardCart#1.0
        elif self.steps_beyond_done is None:
            # Arm1 just fell, episode ends in next step!
            self.steps_beyond_done = 0
            self.rewardStep = rewardCart#1.0
        else: 
            #may only happen in testing; should not be called in training 
            self.steps_beyond_done += 1
            self.rewardStep = 0.0

        # print('reward step=',self.rewardStep)
        return np.array(self.state, dtype=np.float32), self.rewardStep, done, {}



#function to evaluate model after some training    
def EvaluateEnv(model, testEnv, solutionFile, P):
    testEnv.simulationSettings.solutionSettings.sensorsWritePeriod = testEnv.stepUpdateTime

    resultsList = []
    maxErrorPos = 0
    maxErrorAngle = 0
    nSteps = int(P.testingTime/P.stepUpdateTime) 

    nStepsFinal = int(0.75*nSteps) #number of steps after which we do evaluation; before that, control will try to stabilize the inverted n-pendulum
    errorList = []
    model.doLogging = False
    
    #perform several tests
    for i in range(P.numberOfTests):
        testEnv.TestModel(numberOfSteps=nSteps, model=model, solutionFileName=solutionFile, 
                      stopIfDone=False, useRenderer=False, sleepTime=0, showTimeSpent=False) #just compute solution file
        #testEnv.simulationSettings.solutionSettings.solutionWritePeriod = 0.05

        results = testEnv.mbs.GetSensorStoredData(testEnv.sCoordinates)
        # force = testEnv.mbs.GetSensorStoredData(testEnv.sAction) # how do we extract the force? how do we plot it?
        # force_to_list = force.tolist()
        # with open('action_force.json', 'w') as file:
        #     json.dump(force_to_list, file)
        
        if np.isnan(results).any() or testEnv.flagNan: 
            errorList += [float('nan')] # nan leads to a failed test 
        else: 
            maxErrorPos = max(abs((results[nStepsFinal:,1])))
            maxErrorAngle = max(abs((results[nStepsFinal:,P.nArms+1]))) #last arm angle; this should work even for nArms>1
            errorList += [max(maxErrorPos, maxErrorAngle)]
        
        if P.breakTestsWhenFailed and errorList[-1] > P.testErrorMax:
            model.doLogging = True
            return errorList

    model.doLogging = True
    #print('reward log=',model.logger.name_to_value['train/reward'])            
    return errorList



#++++++++++++++++++++++++++++++++++++++++++++++
#parameter function which is run multiple times
#this is the function which is repeatedly called from ParameterVariation
#parameterSet contains dictinary with varied parameters
# parameterSet = {}
# if __name__ == '__main__': #include this to enable parallel processing
# if True:
def ParameterFunction(parameterSet):
    #DELETE, not needed:
    # global mbs
    # mbs.Reset()
    
    #++++++++++++++++++++++++++++++++++++++++++++++
    #++++++++++++++++++++++++++++++++++++++++++++++
    #store default parameters in structure (all these parameters can be varied!)
    class P: pass #create emtpy structure for parameters; simplifies way to update parameters

    #default values
    P.computationIndex = 'Ref'

    P.learningRateFactor = 1    #factor upon original learning rate of method used (e.g., 0.0007 in A2C)
    P.totalLearningSteps = 5000      #total steps, lasts over several episodes
    P.episodeSteps = 200       #these are the number of steps after which we do evaluation
    P.episodeStepsMax = 250    #these are the max number of steps after which we do evaluation
    P.evaluationSteps = 5000    #after this amount of steps, we do evaluation and compute errors (costly)
    P.logLevel = 2
    P.lossThreshold = 1e-2      #loss needs to be below this value to evaluate model
    P.rewardThreshold = 0.95      #mean reward needs to above this value to evaluate model; good: 0.95-0.98
    P.meanSampleSize = 50
    P.stepUpdateTime = 0.02     #this is the step size used by stable baselines
    P.nThreadsTraining = 1      #only possible if just single training is running
    
    # for SAC, TD3 and DDPG
    P.learningStarts = 5000
    P.bufferSize = 1000000
    P.trainFrequency = 1
    P.gradientSteps = 2
    P.tau = 0.005 # from 0.005 to 0.01 usually
    P.targetPolicyNoise = 0.2
    P.targetNoiseClip = 0.1
    P.targetTau = 0.005
    
    P.RLalgorithm = 'A2C'       #way to change training method
    P.rewardMode = 1            #1=sum angles + position cart, 2=sumAngles + position tip, 3=last link angle + position tip
    P.netarchLayerSize = None   #default:64, should be probably exp of 2; larger=slower
    P.netarchNumberOfLayers = None #default:2, larger=slower, but may be more performant
    P.numActions = 2            # default: 2, number of actions; +/- Force; if == 3: zero action
    
    P.relativeFriction = 0.      #friction parameter, 1=full sticking in horizontal position
    P.testMassArmFact = 1       #change mass of arm during testing
    P.testMassCartFact = 1      #change mass of car during testing
    P.lengthFact = 1            #change arm length during testing

    P.numberOfTests = 8                 #number of tests to evaluate performance
    P.testingTime = 10                  #time in seconds for testing (evaluating) model
    P.rewardPositionFactor = 0.5        #0.9-0.95 is less performant for smaller number of learning steps
    P.randomInitializationValue = 0.1   # 
    P.randomInitializationFactorTest= 1 # randomInitializationValue for tests is randomInitializationValue multiplied with this factor
    P.thresholdFactor = 0.75
    P.cartForce = 10            #nominal force applied; needs to be larger for more pendulums in order to be more reactive

    P.storeBestModel = ''       #if name is provided, best model is stored
    P.resultsFile = ''   #this is the file to store parameters for this case
    P.testErrorMax = 0.2        #max. error for test; if this level is achieved, model is saved
    P.stopWhenAllTestsSuccess = True  #will stop before reaching totalLearningSteps
    P.breakTestsWhenFailed = True #there will be many tests, but stopped if one test fails; return value proportional to number of successful tests
    P.verbose = None            #write continuous information; not suitable for multi-threaded parameter variation
    P.bestTestError = np.inf 	# the value of the best stest yet; infinite before first test was done
    
    P.nArms = 1                 #single pendulum nArms=1, double pendulum nArms=2, etc.
    P.case = 0
    # #now update parameters with parameterSet (will work with any parameters in structure P)
    for key,value in parameterSet.items():
        setattr(P,key,value)
        
    #functionData are some values that are not parameter-varied but may be changed for different runs!
    #these values OVERRIDE parameter variation values!!!
    if 'functionData' in parameterSet:
        for key,value in P.functionData.items():
            if key in parameterSet:
                print('ERROR: duplication of parameters: "'+key+'" is set BOTH in functionData AND in parameterSet and would be overwritten by functionData')
                print('Computation will be stopped')
                raise ValueError('duplication of parameters')
            setattr(P,key,value)
            # print('get functionData:', key, '=', value) #for debug

    P.episodeSteps = int(P.episodeSteps)
    P.episodeStepsMax = int(P.episodeStepsMax)
    P.evaluationSteps = int(P.evaluationSteps)
    P.totalLearningSteps = int(P.totalLearningSteps)
    P.logLevel = int(P.logLevel)
    P.numActions = int(P.numActions)
    if P.rewardMode == -1: P.rewardThreshold -= 1
    # print('totalLearningSteps',P.totalLearningSteps)
    # print('episodeSteps',P.episodeSteps)
    #++++++++++++++++++++++++++++++++++++++++++++++
    #++++++++++++++++++++++++++++++++++++++++++++++
    #START HERE: create parameterized model, using structure P, which is updated in every computation
    
    RLalgorithmList = ['A2C','PPO','DQN','SAC','TD3','DDPG'] #0,1,2 in settings
    # print('type=', type(P.RLalgorithm))
    if type(P.RLalgorithm) != str:
        P.RLalgorithm = RLalgorithmList[int(P.RLalgorithm)]
    
    #testingSteps = int(P.testingTime/P.stepUpdateTime)

    if (P.computationIndex == 'Ref') and P.verbose != None:
        P.verbose = True
    elif P.verbose == None:
        P.verbose = False

    if P.verbose:
        print('running in verbose mode')

    #compute current loss obtained from logger    
    def ComputeLogs(model, errorList=[]):
        logInfo = model.logger.name_to_value
        entropy_loss = logInfo['train/entropy_loss']
        policy_loss = logInfo['train/policy_loss']
        value_loss = logInfo['train/value_loss']
        actor_loss = logInfo['train/actor_loss']
        critic_loss = logInfo['train/critic_loss']
        ent_coef = logInfo['train/ent_coef']
        ent_coef_loss = logInfo['train/ent_coef_loss']
        
        if P.RLalgorithm == 'DQN':
            loss = logInfo['train/loss'] # for DQN the loss is written just as loss, the other losses may exist but are 0. 
        elif P.RLalgorithm == 'SAC':
            loss = actor_loss + critic_loss + ent_coef * ent_coef_loss
        elif P.RLalgorithm == 'TD3' or 'DDPG':
            loss = actor_loss + critic_loss
        else:
            loss = policy_loss + model.ent_coef * entropy_loss + model.vf_coef * value_loss
        
        lossList = [abs(loss), entropy_loss, policy_loss, value_loss, actor_loss, critic_loss, ent_coef_loss]
   
        return {'totalSteps':model.num_timesteps,
                'rewardStep':model.logger.name_to_value['train/reward'],
                'errorList':errorList, 
                'lossList':lossList, 
                'rewardMean':model.logger.name_to_value['train/reward_mean'],
                'totalEpisodes':model.logger.name_to_value['time/n_episodes'],
                'stepsPerEpisode':model.logger.name_to_value['time/stepsPerEpisode'],
                'rewardPerEpisode':model.logger.name_to_value['train/rewardPerEpisode'],
                #'totalSteps':model.logger.name_to_value['time/log_steps'],
                # 'trainStatus':model.logger.name_to_value['train/status'],
                    }

    def LogData(logData, model, errorList=[]):
        logData += [ComputeLogs(model, errorList)]

    
    #%%++++++++++++++++++++++++++++++++++++++++++++++++++
    #custom callback to add
    class TensorboardCallback(BaseCallback):
        def __init__(self, verbose = 0, meanSampleSize=50, logData=None, logModel=None, 
                     logLevel=2, stopAtEpisodeAfterSteps=1000):
           super(TensorboardCallback, self).__init__(verbose)
           self.meanSampleSize = meanSampleSize
           self.rewardMean = []
           self.logData = logData
           self.logModel = logModel
           self.logSteps = 0 #additional total count
           self.subTrainingSteps = 0 #subcount for trainings
           self.episodesCnt = 0
           self.logLevel = logLevel
           self.stopAtEpisodeAfterSteps = stopAtEpisodeAfterSteps

           #new variables to measure reward and steps in episode
           self.stepsPerEpisode = 0
           self.rewardPerEpisode = 0

        #this function is added to add specific logs in n-link pendulum
        def DoLog(self, episodeFinished):
            currentReward = self.training_env.get_attr('rewardStep')[0]
            self.logger.record('train/reward', currentReward)
            #compute reward mean only at end of episode, otherwise this is erroneous:
            if len(self.rewardMean) < self.meanSampleSize:
                self.rewardMean = np.hstack((self.rewardMean, [currentReward]))
            else:
                self.rewardMean = np.roll(self.rewardMean, -1)
                self.rewardMean[-1] = currentReward

            if len(self.rewardMean) != 0:
                self.logger.record('train/reward_mean', np.mean(self.rewardMean))
            self.logger.record('time/log_steps', self.logSteps)
            
            self.logger.record('time/n_episodes', self.episodesCnt)

            if episodeFinished:
                self.logger.record('time/stepsPerEpisode', abs(self.stepsPerEpisode) )
                self.logger.record('train/rewardPerEpisode', self.rewardPerEpisode)
                

        def _on_training_start(self) -> None: #reset at each training
            if self.logModel.doLogging:
                self.subTrainingSteps = 0
            
        def _on_training_end(self) -> None: #in this case the training ended and logging needs to be done!
            self.DoLog(True) #at end of training, episode has been set to 0, so do not record ...
            self.stepsPerEpisode = -abs(self.stepsPerEpisode)
            

        #stable baselines function called at every training step; used to do special logging            
        def _on_step(self) -> bool: 
            if not self.logModel.doLogging:
                return True

            self.logSteps += 1
            self.subTrainingSteps += 1

            if self.stepsPerEpisode < 0: #this indicates that previous step reached end of episode
                self.stepsPerEpisode = 0
                self.rewardPerEpisode = 0
                
            self.stepsPerEpisode += 1
            self.rewardPerEpisode += self.training_env.get_attr('rewardStep')[0]

            dones = np.sum(self.locals['dones']).item()
            episodeFinished = (dones != 0)
            
            if episodeFinished:
                self.episodesCnt += dones

            doLog = False
            continueLearning = True
            if self.logLevel < 2 or (self.logLevel == 2 and episodeFinished):
                doLog = True
            if episodeFinished and (self.subTrainingSteps >= self.stopAtEpisodeAfterSteps):
                doLog = True
                continueLearning = False

            if doLog:
                self.DoLog(episodeFinished)
                LogData(self.logData, self.logModel, []) #log without error

            #mark self.stepsPerEpisode for next step to be reset!
            #this allows to log episode information even in case that training ends
            if episodeFinished: 
                self.stepsPerEpisode = -abs(self.stepsPerEpisode)
                
            return continueLearning

    #%%++++++++++++++++++++++++++++++++++++++++++++++++++
    seed = 0
    if P.computationIndex != 'Ref':
        seed = int(P.case)
    
    #seed all randomizers with consistent value
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    np.random.seed(seed)
    random.seed(seed)
    # !!!!!!!!!!!!!!
    # 0) in order to make learning independent from evaluation, randomizer needs to be reset after every substep (after evaluation period)
    # 1) test some results, check simulation
    # 2) add friction to revolute joint (Stribeck/coordinate spring damper)
    # 3) test double pendulum
    # !!!!!!!!!!!!!!
        

    #create model and do reinforcement learning
    learningEnv = InvertedNPendulumEnv(nArms=P.nArms, thresholdFactor=P.thresholdFactor,
                                 rewardPositionFactor=P.rewardPositionFactor,
                                 relativeFriction = P.relativeFriction, #train with same friction as evaluated
                                 stepUpdateTime = P.stepUpdateTime, 
                                 cartForce=P.cartForce,
                                 rewardMode = P.rewardMode,
                                 numActions=P.numActions, 
                                 ) 
    learningEnv.randomInitializationValue = P.randomInitializationValue #standard is 0.05; this is the size of disturbance we add at beginning of learning
    
    #test environment and learning environment need to be decoupled! (gives different results!)
    testenv = InvertedNPendulumEnv(nArms=P.nArms, 
                                   massArmFact=P.testMassArmFact, massCartFact=P.testMassCartFact, lengthFact=P.lengthFact, #these factors only for testing!
                                   rewardPositionFactor=P.rewardPositionFactor,
                                   relativeFriction = P.relativeFriction,
                                   stepUpdateTime = P.stepUpdateTime,
                                   cartForce=P.cartForce,
                                   rewardMode = P.rewardMode,
                                   numActions=P.numActions, 
                                   ) 
    #TODO: check if this is too high for learning/testing:
    testenv.randomInitializationValue = P.randomInitializationValue* P.randomInitializationFactorTest
    #testenv=learningEnv
    
    useVecEnv = False #flag to know if vectorized env is used
    if P.nThreadsTraining!=1: #this case only if run without parameter variation to show animated results
        #try parallized training
        torch.set_num_threads(P.nThreadsTraining) #should be number of real cores
        print('using',P.nThreadsTraining,'cores / vectorized envs')
    
        from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
        learningEnv = SubprocVecEnv([InvertedNPendulumEnv for i in range(P.nThreadsTraining)])
        
        useVecEnv = True
    
    policy_kwargs = None
    if P.netarchLayerSize != None and P.netarchNumberOfLayers != None:
        #net_arch = [dict(pi=[64, 64], vf=[64, 64])] #this is default!
        sL = int(P.netarchLayerSize)
        nL = int(P.netarchNumberOfLayers)
        net_arch = [sL]*nL
        policy_kwargs={'net_arch': net_arch}
        
        if P.verbose:
            print('using custom net_arch', net_arch)

    #main learning task; 1e7 steps take 2-3 hours
    if P.RLalgorithm == 'A2C':
        model = A2C('MlpPolicy',
                    env=learningEnv,
                    learning_rate=P.learningRateFactor*1e-3,
                    # n_steps=20,
                    device='cpu',  #usually cpu is faster for this size of networks
                    #device='cuda',  #usually cpu is faster for this size of networks
                    policy_kwargs=policy_kwargs,
                    verbose=P.verbose)
    elif P.RLalgorithm == 'PPO':
        P.episodeSteps = P.episodeStepsMax
        model = PPO('MlpPolicy', 
                    env=learningEnv,
                    learning_rate=P.learningRateFactor*5e-4, # was 5e-4 for DP and SP
                    n_steps=P.episodeSteps, 
                    #vf_coef=1, # не было раньше
                    #ent_coef=0.01, # не было раньше
                    # learning_rate=P.learningRateFactor*3e-4,
                    device='cpu',  #usually cpu is faster for this size of networks
                    policy_kwargs=policy_kwargs,
                    verbose=P.verbose)
    elif P.RLalgorithm == 'DQN':
        model = DQN('MlpPolicy', 
                    env=learningEnv,
                    # learning_rate=P.learningRateFactor*1e-4,
                    learning_rate=P.learningRateFactor*5e-4,
                    device='cpu',  #usually cpu is faster for this size of networks
                    policy_kwargs=policy_kwargs,
                    verbose=P.verbose)
    if P.RLalgorithm == 'SAC':
        model = SAC('MlpPolicy',
                    env=learningEnv,
                    learning_rate=P.learningRateFactor*5e-4,
                    buffer_size=P.bufferSize,
                    learning_starts=P.learningStarts,
                    train_freq=P.trainFrequency,
                    gradient_steps=P.gradientSteps,
                    tau=P.tau,
                    ent_coef='auto',
                    verbose=P.verbose)
    if P.RLalgorithm == 'TD3':
        model = TD3('MlpPolicy',
                    env=learningEnv,
                    learning_rate=P.learningRateFactor*1e-3,
                    buffer_size=P.bufferSize,
                    learning_starts=P.learningStarts,
                    train_freq=(P.trainFrequency, "step"),
                    gradient_steps=P.gradientSteps,
                    target_policy_noise=P.targetPolicyNoise,
                    target_noise_clip=P.targetNoiseClip,
                    verbose=P.verbose)
    if P.RLalgorithm == 'DDPG':
        model = DDPG('MlpPolicy',
                    env=learningEnv,
                    learning_rate=(P.learningRateFactor*1e-3, P.learningRateFactor*1e-3),
                    buffer_size=P.bufferSize,
                    learning_starts=P.learningStarts,
                    train_freq=(P.trainFrequency, "step"),
                    gradient_steps=P.gradientSteps,
                    verbose=P.verbose)
        
    model.doLogging = True #internal logging, turned off during evaluation

    logInterval = int(1e8) #must be high enough, as values are dumped after logInterval and some values are lost
    returnData=[]
    stopAtEpisodeAfterSteps = P.episodeSteps

    logCallback = TensorboardCallback(meanSampleSize=P.meanSampleSize, logData=returnData, 
                                      logModel=model, logLevel=P.logLevel, 
                                      stopAtEpisodeAfterSteps=stopAtEpisodeAfterSteps)

    ts = -time.time()
    # model.learn(total_timesteps=nSteps, log_interval=logInterval, callback=logCallback)
    fileName = P.resultsFile + str(P.computationIndex) + '.txt'
    maxSuccessfulTests = 0 #this stores the max number of successful tests for certain model
    maxSuccessfulTestsSteps = -1 #store steps at which max successful steps occured
    storeModelName = P.storeBestModel + str(P.computationIndex)
    
    with open(fileName, 'w') as resultsFile:
        resultsFile.write('#parameter variation file for learning\n')
        resultsFile.write('#varied parameters:\n')
        for key,value in parameterSet.items():
            resultsFile.write('#'+key+' = '+str(value)+'\n')
        resultsFile.write('#\n') #marks end of header
            
        varStr = ''
        if P.computationIndex != 'Ref':
            varStr = ', var='+str(P.computationIndex).zfill(2)
        
        
        batch_interval = 30000
        batch_count = 1
        nSteps = 0
        nStepsLastEvaluation = 0
        while nSteps <= P.totalLearningSteps:
            model.learn(total_timesteps=P.episodeStepsMax, log_interval=logInterval, 
                        callback=logCallback, reset_num_timesteps=False)
            logs = ComputeLogs(model,[])
            nSteps = logs['totalSteps']
            
            if nSteps >= batch_interval * batch_count:
                        batch_count += 1
                        model.save(storeModelName + '_' + P.RLalgorithm + '_' + str(nSteps))
            
            if P.verbose:
                print('n='+str(nSteps)+varStr+', lo=', logs['lossList'][0], ', re=', logs['rewardStep'] )
            if (logs['rewardStep'] > P.rewardThreshold
                and (nSteps-nStepsLastEvaluation) >= P.evaluationSteps):
                nStepsLastEvaluation = nSteps
                errorList = EvaluateEnv(model, testenv, None, P)
            # errorList = []    
            # if P.RLalgorithm == 'SAC' or P.RLalgorithm == 'TD3' or P.RLalgorithm == 'DDPG':
            #     if (logs['rewardStep'] > P.rewardThreshold and (nSteps-nStepsLastEvaluation) >= P.evaluationSteps):
            #         nStepsLastEvaluation = nSteps
            #         errorList = EvaluateEnv(model, testenv, None, P)
            # else:            
            #     if (logs['lossList'][0] < P.lossThreshold and logs['rewardStep'] > P.rewardThreshold and (nSteps-nStepsLastEvaluation) >= P.evaluationSteps):
            #         nStepsLastEvaluation = nSteps
            #         errorList = EvaluateEnv(model, testenv, None, P)
            
                successfulTests = 0
                sumError = 0
                for er in errorList:
                    if er <= P.testErrorMax: 
                        successfulTests+=1
                    sumError += er # sum of errors to see if model became better as last time
                
                if successfulTests > maxSuccessfulTests:
                    maxSuccessfulTests = successfulTests
                    maxSuccessfulTestsSteps = nSteps

                    if (successfulTests >= min(4,P.numberOfTests) and P.storeBestModel != '' and sumError < P.bestTestError):
                        model.save(storeModelName + '_' + P.RLalgorithm + '_' + str(nSteps))
                        if P.verbose: 
                            print('  saving model {} \t at step {}, err = {}'.format(storeModelName + '_' + P.RLalgorithm, nSteps, sumError))
                    elif (successfulTests >= min(30,P.numberOfTests) and P.storeBestModel != ''):
                        model.save(storeModelName + '_' + P.RLalgorithm + '_' + str(nSteps) + '_' + str(successfulTests) + '_tests')
                        
                if sumError < P.bestTestError: 
                    P.bestTestError = sumError # remember the best results for the test
                if P.verbose:
                    if ts+time.time() < 7200:
                        sTimeSpent = str(round(ts+time.time(),2)) + ' seconds'
                    else:
                        sTimeSpent = str(round((ts+time.time())/3600,4)) + ' hours'
                    print('  successful tests='+str(successfulTests)+varStr+' (', maxSuccessfulTests, ' max, '+sTimeSpent+')', sep='')
                    print('  current test error {}, \t best test error {}'.format(sumError, P.bestTestError))
                
                logs['errorList'] = errorList
                logs['successfulTests'] = successfulTests
                #logs['force'] = force
                
            resultsFile.write(str(logs)[1:-1]+'\n') #write dictionary without { }, making it machine readable

            if P.stopWhenAllTestsSuccess and (maxSuccessfulTests == P.numberOfTests): #no better model will be found!
                #this output may or may not be visible in parameter variation
                print('found max number of successful tests' + varStr + ' ; stop learning')
                break

        if (maxSuccessfulTests < min(4,P.numberOfTests) and P.storeBestModel != ''):
            if P.verbose:
                print('*** less than 4 successful tests, storing failed model' + varStr + ' ***')
            model.save(storeModelName + '_' + P.RLalgorithm + '_fail') #store failed model, as no best model was found
            resultsFile.write('#less than 4 successful tests, storing failed model\n')
    
        resultsFile.write('#maxSuccessfulTests='+str(maxSuccessfulTests)
                          +', maxSuccessfulTestsAtStep='+str(maxSuccessfulTestsSteps)
                          +', timeSpent='+ str(round(ts+time.time(),2)) + '\n'
                          )
        
    if P.verbose: 
        print('*** learning time total =',ts+time.time(),'***')

    return maxSuccessfulTests


#%% 
def getLegendStr(keyList, columns): 
    fcName = ''
    caseName = ''
    for i, key in enumerate(keyList):
        if key in columns: #only parameters which are varied
            value = valueList[i]
            fvalue = float(value)
            if fvalue == int(fvalue): 
                value = int(fvalue)
            else: 
                validDigits = int(np.ceil(-1*np.log10(fvalue))) + 3 # round 3 significant digits
                value = round(fvalue, validDigits) 
					# add shortcuts for some keys
            if key == 'randomInitializationValue': 
                key = 'randInitVal'
                  
            elif key == 'randomInitializationFactorTest': 
                key = 'randInitValTest'
            strPart = key+str(value)
            if key == 'RLalgorithm': # translate numberings of algorithms to strings
                RLalgorithmList = ['A2C','PPO','DQN']
                try: 
                    strPart = RLalgorithmList[int(value)]
                except: 
                    print('Warning: {}{} unknown in function {}'.format(key, value, 'LoadAndVisualizeData'))
            caseName += '' + strPart
            if key != 'case':
                fcName += '_' + strPart
    return caseName
                

#%%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#this function is called after parameter variation
#provide resultsFile name and path: in case of file 'caseA' it will load caseAvar.txt as parameter variation and caseA0.txt, caseA1.txt, etc with the content
#showFigure: display on screen (False: just render and export)
#exportFigure: if True, figures are written to files
#exportFileType: use 'png' or 'pdf' for exported figures
#outputName: some useful name used for file saving figures
#singleFigure: everything into one figure: reward and test error
#showLoss:additionally show loss plo
#showEpisodes: show reward per episode and steps per episode; only makes sense if logLEvel is 2
#caseInFileName: additionally put name of case in file
#caseInLegend: put shortened case name in legend
def LoadAndVisualizeData(resultsFile, showFigure=True, exportFigure = True, showLoss = False, showRewardPerEpisode = False, showStepsPerEpisode=False, showTestError=False, 
                         showReward = True, caseInFileName=True, caseInLegend=False, outputName='nLink', exportFileType = 'png'):

    resultsFileVar = resultsFile + 'Var.txt'
    #define list of markers if many different solutions are created:
    listMarkerStyles = ['.', 'x ', '+ ', '* ', 'd ', 'D ', 's ', 'X ', 'P ', 'v ', '^ ', '< ', '> ', 'o ', 'p ', 'h ', 'H ']

    if not showFigure:
        import matplotlib
        matplotlib.use('Agg') #this turns of drawing into figure, but draws on background and may save file

    from exudyn.plot import ParseOutputFileHeader
    from exudyn.utilities import PlotLineCode
    from exudyn.processing import SingleIndex2SubIndices
    from exudyn.plot import PlotSensor
    #import exudyn.plot as epl
    import matplotlib.pyplot as plt
    
    import ast #for literal_eval
    import scipy.stats #for scipy.stats.norm.ppf(0.975) = 1.959963984540054, 95% confidence
    
    font = {'size'   : 14}
    plt.rc('font', **font)
    convFact = scipy.stats.norm.ppf(0.975)

    resultsFileNameList = [] #this will get a list of files to be loaded

    #try to load variations; if this variations file does not exist, just use single file 'Ref'
    nCasesFile = 1
    columns = []
    variableRanges = []
    totalVariations = 1
    resultsFileNameList = []
    try:
        with open(resultsFileVar) as file:
            lines = file.readlines()
        
        header = ParseOutputFileHeader(lines)
    
        variableRanges = header['variableRanges']
        columns = header['columns']
        if 'case' not in columns:
            print('WARNING: LoadAndVisualizeData: no cases found')
        else:
            nCasesFile = variableRanges[columns.index('case')-2][2]

        variableRangesCnt = []
        for rg in variableRanges:
            variableRangesCnt += [rg[2]]
        totalVariations = int(np.prod(variableRangesCnt)/nCasesFile)
        
        for i in range(totalVariations*nCasesFile):
            resultsFileNameList += [resultsFile+str(i)+'.txt']
    except:
        #no variations file, just single computation with 'Ref.txt' ending
        resultsFileNameList = [resultsFile] # +'Ref.txt'
        print('No valid variations file "'+resultsFileVar+'" found, trying to load single file with Ref number')
        pass
    
    plt.close('all')
    #now load data of sets (variations, without cases; cases go into one plot)
    for nVar in range(totalVariations):#use this to create several plots based on number of variations
        
        for cnt0 in range(nCasesFile):
            cnt = cnt0+nCasesFile*nVar #total file count; case must be last!
            varFile = resultsFileNameList[cnt]
            # print('load data from file:', varFile)
            keyList = [] 
            valueList = [] 
            dataStart = 0

            with open(varFile) as file:
                lines = file.readlines()
                #extract variation parameters from single file
                while lines[dataStart][0] == '#':
                    if '=' in lines[dataStart]:
                        line = lines[dataStart][1:]
                        splitLine = line.split('=')
                        if len(splitLine) == 2: #if not 2, this is some other data
                            key = splitLine[0].strip()
                            value = splitLine[1]
                            if key != 'computationIndex' and key != 'functionData':
                                # print('key=', key, ', value=', value)
                                keyList += [key]
                                valueList += [value]
                    dataStart+=1
        
            #load data from single results files:
            dataList = [] #1 dictionary per list item
            try: 
                for dataStr in lines[dataStart:]:
                    #line has structure: nSteps : {'errorList':[...], 'lossList':[...], ...}
                    #print('line=', dataStr)
                    if dataStr[0] != '#': #leave out comment lines
                        # replace nan with None as it can not be read otherwise
                        dataDir = ast.literal_eval('{'+dataStr.replace('nan', 'None')+'}') 
                        dataList += [dataDir]
                        
                            
            except: 
                print('error in :\n{}'.format(dataStr))
                
            #now create lists for plots; as not all data sets contain all information, there are several x/y lists
            
            xLoss=[]
            yLoss=[]
            yLossMean=[]
            yRewardMean=[]
            yReward=[]
            xError=[]
            yError=[]
            xMeanError=[]
            yMeanError=[]
            yStdError=[]
    
            yStepsPerEpisode=[]
            yRewardPerEpisode=[]
    
            for dataCnt, item in enumerate(dataList):
            
                errorList = item['errorList']
                lossList = item['lossList']
                xLoss += [item['totalSteps']]
                yLoss += [lossList[0]] #0=total loss
                yLossMean += [np.mean(yLoss[-20:])] #track the mean values of last 20 episodes
                yReward += [item['rewardStep']] 
                yRewardMean += [item['rewardMean']]

                yStepsPerEpisode += [item['stepsPerEpisode']] 
                yRewardPerEpisode += [item['rewardPerEpisode']] 
                    
                if len(errorList)!=0:
                    xError += [item['totalSteps']]*len(errorList)
                    yError += errorList
                    xMeanError += [item['totalSteps']]
                    i_eval =  [i for i in range(len(errorList)) if errorList[i] == None]
                    for i in i_eval[::-1]: errorList.pop(i)
                    yMeanError += [np.mean(errorList)]
                    yStdError += [convFact*np.std(errorList)/np.sqrt(len(errorList))]
                #print(xError,yError,len(errorList))
            yStdError = np.array(yStdError)
            
            # cut it off if there are more cases than colors ... 
            kMarker = int((cnt0%nCasesFile) % 16)
            kColor = int((cnt0%nCasesFile))
            
            caseName = str(cnt)
            if caseInLegend: 
                caseName = ''
            
            fcName = ''

            for i, key in enumerate(keyList):
                if key in columns: #only parameters which are varied
                    value = valueList[i]
                    fvalue = float(value)
                    if fvalue == int(fvalue): 
                        value = int(fvalue)
                    else: 
                        validDigits = int(np.ceil(-1*np.log10(fvalue))) + 3 # round 3 significant digits
                        value = round(fvalue, validDigits) 
					# add shortcuts for some keys
                    if key == 'randomInitializationValue': 
                        key = 'randInitVal'
                          
                    elif key == 'randomInitializationFactorTest': 
                        key = 'randInitValTest'
                    strPart = key+str(value)
                    if key == 'RLalgorithm': # translate nubmerings of algorithms to strings
                        RLalgorithmList = ['A2C','PPO','DQN']
                        try: 
                            strPart = RLalgorithmList[int(value)]
                        except: 
                            print('Warning: {}{} unknown in function {}'.format(key, value, 'LoadAndVisualizeData'))
                    caseName += ', ' + strPart
                    if key != 'case':
                        fcName += '_' + strPart
                
            # for i in range(len(variableRanges)):
            #     rg = variableRanges[i]
            #     val = rg[0]+(rg[1]-rg[0])*varList[i]/max(rg[2]-1,1)
            #     val = round(val,5)
            #     if columns != []:
            #         strPart = columns[2+i][:8]+str(val)
            #         caseName += ' ' + strPart
            #         if i < len(variableRanges)-1: #case not included in file
            #             fcName += '_' + strPart

            lcName = ''
            if caseInLegend:
                lcName = caseName
            else:
                if int(cnt0) == cnt0: 
                    cnt0 = int(cnt0) # avoid printing the .0 for int values
                lcName = ', random case '+str(cnt0)
                
            #create and possibly save plots for different values
            logScaleY = True #for loss and error; reward stays linear, if not in single figure
            if True:
                if showReward: 
                    # PlotSensor(None, sensorNumbers=np.array([xLoss,yReward]).T, xLabel='steps', 
                    #             newFigure=False, colorCodeOffset=kColor, lineStyles=[':'], labels=['reward'+caseName],
                    #             markerStyles=[listMarkerStyles[kMarker]], markerDensity=None, logScaleY=True)
                    plt.figure('VariationReward'+str(nVar))
                    # print('VariationReward'+str(nVar))
                    lineStylesReward = []
                    
                    plt.plot(xLoss, yRewardMean)
                if showTestError: 
                    PlotSensor(None, sensorNumbers=np.array([xLoss,yRewardMean]).T, xLabel='steps', yLabel='reward', 
                                newFigure=False, colorCodeOffset=kColor, lineStyles=lineStylesReward, labels=['mean reward'+lcName],
                                markerStyles=[listMarkerStyles[kMarker]], markerDensity=None, 
                                #logScaleY=logScaleY,
                                figureName='VariationReward'+str(nVar),#  flagShow=False #reuse figure
                                )
    
                    
                    
                # print('VariationTest'+str(nVar))
                if True: 
                    yLabel = 'test error / reward'
                    figureName = 'VariationTest'+str(nVar)
                    yLabel = 'test error'
                    ps=PlotSensor(None, sensorNumbers=np.array([xMeanError,yMeanError]).T, xLabel='steps', yLabel=yLabel,
                                newFigure=False, colorCodeOffset=kColor, labels=['test error'+lcName],
                                markerStyles=[listMarkerStyles[kMarker]], markerDensity=None, # flagShow=False,
                                figureName=figureName, 
                                logScaleY=logScaleY, 
                                )
                    ps[2].fill_between(xMeanError, (yMeanError-yStdError), (yMeanError+yStdError), color=PlotLineCode(kColor)[0], alpha=0.1)
                else:
                    PlotSensor(None, sensorNumbers=np.array([xError,yError]).T, xLabel='steps', yLabel=yLabel, # flagShow=False,
                                newFigure=False, colorCodeOffset=kColor, lineStyles=[''], labels=['test error'+lcName],
                                markerStyles=[listMarkerStyles[kMarker]], markerDensity=None, logScaleY=logScaleY)

            if showStepsPerEpisode: 
                plt.figure('EpisodesSteps'+str(nVar))
                PlotSensor(None, sensorNumbers=np.array([xLoss,yStepsPerEpisode]).T, xLabel='steps', yLabel='steps per episode', 
                        newFigure=False, colorCodeOffset=kColor, labels=['steps per episode'+lcName],
                        markerStyles=[listMarkerStyles[kMarker]], markerDensity=None, #  flagShow=False,
                        #logScaleY=logScaleY,
                        figureName='EpisodesSteps'+str(nVar), #reuse figure
                        )
            
            if showRewardPerEpisode:
                plt.figure('EpisodesReward'+str(nVar))
                PlotSensor(None, sensorNumbers=np.array([xLoss,yRewardPerEpisode ]).T, xLabel='steps', yLabel='reward per episode', 
                            newFigure=False, colorCodeOffset=kColor+0*nCasesFile, labels=['reward per episode'+lcName],
                            markerStyles=[listMarkerStyles[kMarker]], markerDensity=None, # flagShow=False,
                            #logScaleY=logScaleY,
                            figureName='EpisodesReward'+str(nVar), #reuse figure
                            )
                
            if showLoss:
                # print('VariationLoss'+str(nVar))
                PlotSensor(None, sensorNumbers=np.array([xLoss,yLossMean]).T, xLabel='steps', yLabel='loss', 
                            newFigure=False, colorCodeOffset=kColor, labels=['loss'+lcName],
                            markerStyles=[listMarkerStyles[kMarker]], markerDensity=None, # flagShow=False, 
                            figureName = 'VariationLoss'+str(nVar), logScaleY=logScaleY)

        if exportFigure:
            if not caseInFileName:
                fcName = str(nVar)
            plt.figure('VariationReward'+str(nVar))
            plt.savefig(outputName+'Reward'+fcName+'.'+exportFileType)

            plt.figure('VariationTest'+str(nVar))
            plt.savefig(outputName+'Test'+fcName+'.'+exportFileType)
            if showLoss:
                plt.figure('VariationLoss'+str(nVar))
                plt.savefig(outputName+'Loss'+fcName+'.'+exportFileType)
            if showStepsPerEpisode:
                plt.figure('EpisodesSteps'+str(nVar))
                plt.savefig(outputName+'EpisodesSteps'+fcName+'.'+exportFileType)
            if showRewardPerEpisode:
                plt.figure('EpisodesReward'+str(nVar))
                plt.savefig(outputName+'EpisodesReward'+fcName+'.'+exportFileType)


#%% 
def ResampleVectorDataNonConstant(xVector, yVector, deltaT):
    numberOfSamples = int(np.int(xVector[-1]/deltaT))+1 # always round down and add one additional point! 
    xVectorResampled = [None]*numberOfSamples
    yVectorResampled = [None]*numberOfSamples        
    iVector = 0
    xMin = np.min(xVector)
    xMax = np.max(xVector)
    for i in range(numberOfSamples):
        xVectorResampled[i] = i*deltaT
        if xVectorResampled[i] < xMin: 
             yVectorResampled[i] = yVector[0]
        elif xVectorResampled[i] > xMax: 
             yVectorResampled[i] = yVector[-1]
        else: 
            while xVectorResampled[i] > xVector[iVector+1]: 
                iVector +=1
            dy = yVector[iVector+1] - yVector[iVector]
            dx = xVector[iVector+1] - xVector[iVector]
            yVectorResampled[i]  = yVector[iVector] + dy/dx * (xVectorResampled[i] - xVector[iVector])
            # if not(xVector[iVector] > xMax): 
                
            # else: 
                    
            
        
        
        # yVectorResampled[i] = InterpolateVectorData(xVector, yVector, i*deltaT)   
    #resampledSig = np.interp(tSim, xVector, yVector)
    return [xVectorResampled, yVectorResampled]

#%%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#this function is called after parameter variation for postprocessing of the data; not finished yet! 
# 
def PostProcessing(resultsFile):
        resultsFileVar = resultsFile + 'Var.txt'
        ## todo: not finished yet!
        from exudyn.plot import ParseOutputFileHeader
        import ast #for literal_eval
        import scipy.stats #for scipy.stats.norm.ppf(0.975) = 1.959963984540054, 95% confidence
        import matplotlib.pyplot as plt
        
        font = {'size'   : 14}
        plt.rc('font', **font)        
        convFact = scipy.stats.norm.ppf(0.975)

        resultsFileNameList = [] #this will get a list of files to be loaded

        #try to load variations; if this variations file does not exist, just use single file 'Ref'
        nCasesFile = 1
        columns = []
        variableRanges = []
        totalVariations = 1
        resultsFileNameList = []
        try:
            with open(resultsFileVar) as file:
                lines = file.readlines()
            
            header = ParseOutputFileHeader(lines)
            
            variableRanges = header['variableRanges']
            columns = header['columns']
            if 'case' not in columns:
                print('WARNING: LoadAndVisualizeData: no cases found')
            else:
                nCasesFile = variableRanges[columns.index('case')-2][2]

            variableRangesCnt = []
            for rg in variableRanges:
                variableRangesCnt += [rg[2]]
            totalVariations = int(np.prod(variableRangesCnt)/nCasesFile)
            
            for i in range(totalVariations*nCasesFile):
                resultsFileNameList += [resultsFile+str(i)+'.txt']
        except:
            #no variations file, just single computation with 'Ref.txt' ending
            resultsFileNameList = [resultsFile+'Ref.txt']
            print('No valid variations file "'+resultsFileVar+'" found, trying to load single file with Ref number')
            pass
        
        # todo: what do we want to know? 
        # number of steps until all tests are successful (convergence)
        # reward per episode
        # {'mean': 0, 'stdDev': 0}
        # plt.close('all')
        plt.figure('TestStatistics')
        cntColor = 0
        nSteps = []
        #now load data of sets (variations, without cases; cases go into one plot)
        for nVar in range(totalVariations):#use this to create several plots based on number of variations
            # plt.figure('Variation' + str(nVar))
            numSuccessfullTests =[] # max num successful tests
            numSuccessfullTestsSteps =[] # when are the max num of successfuls tests
            numSuccessfullTestsMax, numSuccessfullTestsMaxSteps = [], []
            
            # rewardPerEpisode = []
            # stability of training: how to measure if the model becomes worse with more training? 
            
            for cnt0 in range(nCasesFile):
                cnt = cnt0+nCasesFile*nVar #total file count; case must be last!
                varFile = resultsFileNameList[cnt]
                # print('load data from file:', varFile)
                keyList = [] 
                valueList = [] 
                dataStart = 0
                # plt.figure('Variation' + str(nVar))


                with open(varFile) as file:
                    lines = file.readlines()
                    #extract variation parameters from single file
                    while lines[dataStart][0] == '#':
                        if '=' in lines[dataStart]:
                            line = lines[dataStart][1:]
                            splitLine = line.split('=')
                            if len(splitLine) == 2: #if not 2, this is some other data
                                key = splitLine[0].strip()
                                value = splitLine[1]
                                if key != 'computationIndex' and key != 'functionData':
                                    # print('key=', kerrorListey, ', value=', value)
                                    keyList += [key]
                                    valueList += [value]
                                
                        dataStart+=1
                numSuccessfullTests += [[0]]
                numSuccessfullTestsSteps += [[0]]
                #load data from single results files:
                dataList = [] #1 dictionary per list item
                nSteps += [0]
                for dataStr in lines[dataStart:]:
                    #line has structure: nSteps : {'errorList':[...], 'lossList':[...], ...}
                    #print('line=', dataStr)
                    if dataStr[0] != '#': #leave out comment lines
                        dataDir = ast.literal_eval('{'+dataStr+'}')
                        dataList += [dataDir]
                        if 'successfulTests' in dataDir: # and dataDir['successfulTests'] > numSuccessfullTests[-1]: 
                            numSuccessfullTests[-1] += [dataDir['successfulTests']]
                            numSuccessfullTestsSteps[-1] += [dataDir['totalSteps']]
                        nSteps[-1] = dataDir['totalSteps']
                numSuccessfullTestsMax += [max(numSuccessfullTests[-1])]
                numSuccessfullTestsMaxSteps += [max(numSuccessfullTestsSteps[-1])]
                # plt.plot(numSuccessfullTestsSteps[-1], numSuccessfullTests[-1], 'x--')
            # interpolate to calculate mean and min/max values for the cases
            numTestsMean = None
            nTests = len(numSuccessfullTests)
            xC, yC = [], []
            lenMin = 1e38
            for i in range(len(numSuccessfullTests)): 
                try: 
                    xResample, yResample = ResampleVectorDataNonConstant(numSuccessfullTestsSteps[i], numSuccessfullTests[i], 1000)
                except:     
                    print('warning: no successful tests for variation {}, case {}'.format(nVar, i))
                    xResample, yResample = ResampleVectorDataNonConstant([0,nSteps[i]], [0,0], 1000)
                xC += [xResample]
                yC += [yResample]
                # plt.plot(numSuccessfullTestsSteps[i], numSuccessfullTests[i], 'x--')
                
                if len(xResample) < lenMin: 
                    lenMin = len(xResample)
            xMean = xC[0][:lenMin]
            yMean = np.array(yC[0][:lenMin])/nTests
            yMin = yC[0][:lenMin]
            yMax = yC[0][:lenMin]
            # plt.plot(xMean, yMin, label='min iteration {}'.format(0))
            for i in range(1, len(yC)): 
                yMean += np.array(yC[i][:lenMin])/nTests
                # yMin = np.min(np.array([yMin, yC[i][:lenMin]]), axis=0)
                yMin = np.min(np.concatenate([[yMin], [yC[i][:lenMin]]], axis=0), axis=0)
                yMax = np.max(np.concatenate([[yMax], [yC[i][:lenMin]]], axis=0), axis=0)
            plt.plot(xMean, yMean, color=PlotLineCode(cntColor)[0])
            plt.plot(xMean, yMin, '--', color=PlotLineCode(cntColor)[0], linewidth = 0.75)
            plt.plot(xMean, yMax, '--',color=PlotLineCode(cntColor)[0], linewidth = 0.75)
            plt.fill_between(xMean, yMin, yMax, color=PlotLineCode(cntColor)[0], alpha=0.1)           
            
            cntColor += 1
            # break
        plt.grid(True)
        plt.gca().set(xlabel='steps', ylabel='number of successful tests')
            
            # plt.legend()
        return
    
# description: parses the output file from the ParameterVariation to obtain; may not work for all values correctly because of the used regular expressions!
# input: strFileName: a fileName as a string in the format as created  by the exudyn function ParameterVariation
#       flagLegacyV5: in version V5 each test was 1 a variation; now several tests are in one variation; thereby the exact randomized starting states are not written now
# output: 
def PostProcessingTests(strFileName, flagLegacyV5 = True):
    import re
    import ast
    print('loading from file {}'.format(strFileName))
    valueList, iVar, iTest, successes = [],[], [], []
    valueList = [[], [], [], [], []]
    with open(strFileName, 'r') as file: 
        text = file.readlines()
    for lines in text: 
        if '#' in lines:
            if 'globalIndex' in lines:  
                for tmpStr in lines.split(','): 
                    if 'state' in tmpStr: 
                        iVar += [int(tmpStr[-1])]
            else: 
                continue
        else: 
            # print(lines)
            l1, l2 = lines.split('(')
            # iTest += [int(l1[:-2])]
            
            successes += [int(l2[0])]
            buffer = ast.literal_eval('(' +l2)
            
            
            valueList[0] += [buffer[0][0]] # success
            valueList[1] += [buffer[0][1]] # max error
            valueList[2] += [buffer[0][2]] # error coordinates
            if flagLegacyV5: 
                valueList[3] += [buffer[0][3]] # actually used state0 including randomizer
                
            valueList[4] += [list(buffer[1:3])]
            iTest += [buffer[3]]
            if flagLegacyV5: stepNr = buffer[4]
            if not(flagLegacyV5): stepNr = buffer[3]
            if not(stepNr % 20000):  
                    print('process step '+ str(stepNr))
            # values = [int(l2[0])] # success true ==1, false == 0
            # buffer = re.findall(r'\d+\.\d+', l2)
            # for strVal in buffer: 
            #     values +=  [float(strVal)]
            # valueList += [np.array(np.array(buffer, dtype=float)).tolist()]
            # i =
    for i in range(len(valueList)): 
        valueList[i] = np.array(valueList[i])
    # valueList = np.array(valueList)
    state0 = valueList[-1] # [:,-2:]
    if flagLegacyV5: 
        nTestsRand = np.max(iTest) # number of randomized tests for each state combination
    else: 
        successes = valueList[0]
        nTestsRand = 1
    # iVAr = 
    return iTest, iVar, successes, valueList, state0, nTestsRand

# helper function
def setFigSize(fig, fSize=14): 
    for axs in fig.axes: 
        axs.yaxis.get_label().set_fontsize(fSize)
        axs.xaxis.get_label().set_fontsize(fSize)
        axs.title.set_size(fSize)
        axs.tick_params(axis='both', which='major', labelsize=fSize)
        # ax.tick_params(axis='both', which='minor', labelsize=8)
    return

# description: visualizes the output from the Parametervariation in a 2D state space
# input: 
# * states: the list of states; state1 is on the x-axis, state2 on the y-axis
# * successes: a list of the size len(states[0])*len(states[1]*nTestsRnd)
# * iPlot: the numbers of the states plotted as a vector with 2 entries, e.g. [1,2] are the two angles for the double link pendulum on the cart
# * nTestRand: the number of randomized tests for each pixel
# * stateStr: list of strings for labelling the axes
# * if flagRand=False no randomized testing done for the gridpoints
# * pDict: currently unused
# * flag2DPlot: when true using seaborn to make a heatmap; otherwise 3D graphic is used and the the successes are  plotted on the z-axis
def plotMyStates2D(states, successes, iPlot, nTestsRand, stateStr = None, pDict=[], flagRand = False, 
                   flag2DPlot=True, flagDebug=False, trainingSize = None, testSize = None): 
    import matplotlib.pyplot as plt
    if len(states) > 2:     # when plotted after the parametervariation the state
        state1 = states[iPlot[0]]
        state2 = states[iPlot[0]]
    elif len(states) == 2: # when read from the parameter variation file
        state1, state2 = states[0], states[1]
    else: 
        raise ValueError('length of states is not valid for function plotMyStates2D')
    dx = np.diff(state1[0:2])/2
    dy = np.diff(state2[0:2])/2
    color = ['k', 'g']
    k = 0
    if flagRand and not(flag2DPlot): 
        fig, axs = plt.subplots(subplot_kw={"projection": "3d"})
    else: 
        plt.figure()
        axs = plt.gca()
    if flagRand: MatrixVal = np.zeros([len(state2), len(state1)])
    
    for j, x in enumerate(state1): 
        for i, y in enumerate(state2): 
            # color='g'
            if flagRand: 
                # succ =successespDict
                if np.nan in successes[k:k+nTestsRand]: 
                    print('nan!')
                MatrixVal[i,j] = np.sum(successes[k:k+nTestsRand])
                # MatrixVal[i,j] = x*y
                
            else: 
                # coloring areas with sidelength of dx/2...
                xnodes = [x + dx, x - dx, x-dx, x+dx]
                ynodes = [y-dy, y-dy, y+dy, y+dy]
                axs.fill(xnodes,ynodes, color = color[int(successes[k])],label='_nolegend_')
                # print('x: {}, y: {}, state0: {}, state1: {}, success: {}'.format(x, y, \
                #                 pDict['state{}'.format(iPlot[0])][k], pDict['state{}'.format(iPlot[1])][k], successes[k]))
            k+= nTestsRand
    
    # set ticks
    if flagRand: 
        if flag2DPlot: 
            import seaborn as snb
            # print(MatrixVal)
            ticks, ticksPos = [], []
            for state in [state1, state2]: 
                if len(state) < 8: 
                    ticks += [state]
                    ticksPos += [np.linspace(0, len(state), len(ticks[-1])) + 0.5]
                else: 
                    # valTicks = int(len(state)/8)
                    # ticksTxt += [state[0::valTicks]]
                    ticks += [np.linspace(state[0], state[-1], 5)]
                    ticksPos += [np.linspace(0, len(state)-1, 5) + 0.5] #  0.5*0]
                ticks[-1] =  np.round(ticks[-1], 8) # round to avoid ticks like 0.90000000000000001
            if flagDebug: 
                snb.heatmap(MatrixVal, cmap="YlGnBu", annot=True) # , xticklabels=ticks[0], yticklabels=ticks[1])    
            else:
                nTests = np.max(MatrixVal)
                def funcScaling(data, nTests): 
                    return np.log(nTests - data + 1)
                    
                
                snb.heatmap(funcScaling(MatrixVal, nTests), cmap="YlGnBu", annot=False) # , xticklabels=ticks[0], yticklabels=ticks[1])
                # nMax = [0, np.log(np.max(MatrixVal) - MatrixVal +1)] # from 0 to 
                if 1:  # create colorbar for annotation
                    if nTests <= 5: 
                        ticksLabelsCM = [0,1,nTests]
                    elif nTests <= 10: 
                        ticksLabelsCM = [ 0,1,5,nTests]
                        
                    elif nTests <= 50:
                        ticksLabelsCM = [0,1,5,20, nTests]
                    else: 
                        ticksLabelsCM = [0,1,5,20, 50, nTests]
                    ticksCM = funcScaling(nTests-ticksLabelsCM, nTests)
                    ticksLabelsCM = np.array(ticksLabelsCM, dtype=int)
                    plt.gca().collections[0].colorbar.ax.set_yticks(ticksCM, ticksLabelsCM)
                    plt.gca().collections[0].colorbar.ax.set_ylabel('fails')
            plt.xticks(ticksPos[0], ticks[0])
            plt.yticks(ticksPos[1], ticks[1])
            fig = plt.gcf()

        else: 
            from matplotlib import cm
            # Xplt = np.arange(np.min(state[iVar[0]]), np.max(state[iVar[0]]), 0.25)
            # Yplt = np.arange(-np.min(state[iVar[1]]), np.max(state[iVar[1]]), 0.25)
            Xplt, Yplt = np.meshgrid(state1, state2) 
            surf = axs.plot_surface(Xplt, Yplt, MatrixVal, cmap=cm.coolwarm, linewidth=0.1, antialiased=True)
        
    if 0: # do not set minor grid; with finer calculations you can't see anything there
        minorTicksX = 0.5 + np.arange(len(state1))
        minorTicksY = 0.5 + np.arange(len(state2)) # state2
        axs.set_xticks(minorTicksX, minor=True)
        axs.set_yticks(minorTicksY, minor=True)
      
    fig.gca().grid(True, 'major', axis='x')
    fig.gca().grid(True, 'major', axis='y')
    # print('ticksx:\n{}\n, tickxY:{}\n'.format(minorTicksX, minorTicksY))  
    if not(stateStr is None): 
        # get label from iPlot, which are the numbers of the plotted states
        fig.gca().set(xlabel=stateStr[iPlot[0]], ylabel=stateStr[iPlot[1]])
    setFigSize(plt.gcf())
    plt.gca().set_xlim([0, len(state1)])
    plt.gca().set_ylim([0, len(state2)])
    
    if not(trainingSize is None): 
        pass
    return MatrixVal, state1, state2, fig
    
def OutPutAsLatexTable(array): 
    # todo
    print('not implemented yet')
    
    return

# read in the content from the results file to then in
def AppendVersionToResultFile(resultsFile): 
    
    with open(resultsFile, "r") as f: 
        contents = f.readlines()
    
    from torch import __version__ as torchVersion
    from stable_baselines3 import __version__ as stbVersion
    
    # the string containing the version
    strAppend = 'stableBaselines3 version = {}; torch version = {}'.format(stbVersion , torchVersion)
    strAppend = '#{}\n'.format(strAppend)
    iAppend = 1 # if no version string can not be found
    for i, strLine in enumerate(contents): 
        if 'version' in strLine: 
            iAppend = i +1
        if strAppend in strLine: 
            # if strAppend already inserted do not insert it a second time!
            return 
    contents.insert(iAppend, strAppend)
    contents = "".join(contents)
    
    with open(resultsFile, "w") as f:
        f.write(contents)
        # write file again
    return


# set the fontsize and fonttype for the matplotlib library to use the computer modern font used by latex
def SetMatplotlibSettings(): 
    import matplotlib as mpl
    import matplotlib.font_manager as font_manager
    import matplotlib.pyplot as plt
    fsize = 16 # general fontsize
    tsize = 16 # legend
    
    # setting ticks, fontsize
    tdir = 'out' # 'in' or 'out': direction of ticks
    major = 5.0 # length major ticks
    minor = 3.0 # length minor ticks
    lwidth = 0.8 # width of the frame
    lhandle = 2.0 # length legend handle
    plt.style.use('default')
    
    # set font to the locally saved computermodern type
    plt.rcParams['font.family']='serif'
    cmfont = font_manager.FontProperties(fname=mpl.get_data_path() + '/fonts/ttf/cmr10.ttf')
    plt.rcParams['font.serif']=cmfont.get_name()
    plt.rcParams['mathtext.fontset']='cm'
    plt.rcParams['axes.unicode_minus']=False
        
    plt.rcParams['font.size'] = fsize
    plt.rcParams['legend.fontsize'] = tsize
    
    
    plt.rcParams['xtick.direction'] = tdir
    plt.rcParams['ytick.direction'] = tdir
    
    plt.rcParams['xtick.major.size'] = major
    plt.rcParams['xtick.minor.size'] = minor
    plt.rcParams['ytick.major.size'] = major
    plt.rcParams['ytick.minor.size'] = minor
    
    plt.rcParams['axes.linewidth'] = lwidth
    plt.rcParams['legend.handlelength'] = lhandle
    return 

def FormatXAxis(power): 
    import matplotlib.pyplot as plt
    from matplotlib.ticker import ScalarFormatter
    if power == 6: 
        pass
    myFormatter = ScalarFormatter()
    myFormatter.set_powerlimits((power,power))
    myFormatter.useMathText = True # show  with 10^power
    plt.gca().xaxis.set_major_formatter(myFormatter)
    
    
#%% 
if __name__ =='__main__': # for debugging

    # strLoad = 'NArmTest/DoubleA_800000tests2023_04_04_15h52m38s.txt'
    # strLoad = 'NArmTest/DoubleA_100tests2023_04_05_11h46m50sPList.txt'
    # # iTest_, iPlot_, successes_, valueList_, state0_ , nTestsRand_ = PostProcessingTests(strLoad)
    # strLoad = 'NArmTest/SinglePendulumF_1000tests2023_04_14_19h00m23s.txt'
    strLoad =  'stability_zones/SinglePendulum_A2C_r2_v1Model1_A2C200000tests2023_08_16_14h50m08s'
    iTest_, iPlot_, successes_, valueList_, state0_ , nTestsRand_ = PostProcessingTests(strLoad, flagLegacyV5=True)
    # iTest_, iPlot_, successes_, valueList_, state0_ , nTestsRand_= PostProcessingTests(fileName + '.txt')
    state1 = set(np.round(state0_[:,0], 14))
    state2 = set(np.round(state0_[:,1], 14))
      
    states = [list(state1), list(state2)]
    for i in range(2): 
        states[i].sort()
    plotMyStates2D(states, successes_,  iPlot_, nTestsRand_, flagRand=True) # , pDict)

# %%
