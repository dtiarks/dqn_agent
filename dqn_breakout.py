#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 20 20:36:49 2017

@author: daniel
"""

from __future__ import print_function

import gym
import numpy as np
import tensorflow as tf
import time
from collections import deque  
import datetime
import cv2
import os
import sys
from gym import wrappers
import argparse
from memory import ReplayMemory as RPM
from tensorflow.python.client import timeline

class QNet(object):
    def __init__(self,sess,name,params,train=True):
        self.params=params
        self.sess=sess
        self.name=name
        self.input_shape=[None ,params['framesize'],params['framesize'],params['frames']] #add to hyperparamters
        self.images_placeholder = tf.placeholder(tf.float32,shape=self.input_shape)
        self.target_placeholder = tf.placeholder(tf.int32,shape=[None,params['actionsize']])
        self.reward_placeholder = tf.placeholder(tf.float32,shape=[None])
        self.action_placeholder = tf.placeholder(tf.int32,shape=[None])
        self.done_placeholder = tf.placeholder(tf.float32,shape=[None,params['actionsize']])
        self.train=train
        self.buildNet()
        
    def buildNet(self):
        input_layer = self.images_placeholder

        with tf.name_scope(self.name):
            with tf.name_scope('conv1'):
                # 8x8 conv, 4 inputs, 32 outputs, stride=4
                self.W_conv1 = self._weight_variable([8, 8, 4, 32],"W_conv1")
                self.b_conv1 = self._bias_variable([32],"b_conv1")
                h_conv1 = tf.nn.relu(self._conv2d(input_layer, self.W_conv1, 4) + self.b_conv1)
    
            with tf.name_scope('conv2'):
                # 4x4 conv, 32 inputs, 64 outputs, stride=2
                self.W_conv2 = self._weight_variable([4, 4, 32, 64],"W_conv2")
                self.b_conv2 = self._bias_variable([64],"b_conv2")
                h_conv2 = tf.nn.relu(self._conv2d(h_conv1, self.W_conv2, 2) + self.b_conv2)
                
            with tf.name_scope('conv3'):
                # 3x3 conv, 64 inputs, 64 outputs, stride=1
                self.W_conv3 = self._weight_variable([3, 3, 64, 64],"W_conv3")
                self.b_conv3 = self._bias_variable([64],"b_conv3")
                h_conv3 = tf.nn.relu(self._conv2d(h_conv2, self.W_conv3, 1) + self.b_conv3)
            
            dim=h_conv3.get_shape()
            dims=np.array([d.value for d in dim])
            reshaped_dim = np.prod(dims[1:])
            with tf.name_scope('dense1'):
                self.W_fc1 = self._weight_variable([reshaped_dim, 512],"W_fc1")
                self.b_fc1 = self._bias_variable([512],"b_fc1")
    
                h_conv3_flat = tf.reshape(h_conv3, [-1, reshaped_dim])
                h_fc1 = tf.nn.relu(tf.matmul(h_conv3_flat, self.W_fc1) + self.b_fc1)
                
            with tf.name_scope('output'):
                self.W_fc2 = self._weight_variable([512, self.params['actionsize']],"W_fc2")
                self.b_fc2 = self._bias_variable([self.params['actionsize']],"b_fc2")
    
                self.action_logits=tf.add(tf.matmul(h_fc1, self.W_fc2), self.b_fc2,"logits")

        
        self.greedy_actions=tf.argmax(self.action_logits,1)
        return self.action_logits
    
    def estimateActionGreedy(self,state_feed):
        feed=np.expand_dims(state_feed,axis=0)
        s=np.array(feed,dtype=np.float32)/255.
        prediction_index = self.sess.run(self.greedy_actions,
                          feed_dict={self.images_placeholder: s})

        return prediction_index[0]
    
    def meanQ(self,state_feed):
        feed=np.expand_dims(state_feed,axis=0)
        s=np.array(feed,dtype=np.float32)/255.
        q = self.sess.run(self.action_logits,
                          feed_dict={self.images_placeholder: s})
        
        qmean=np.mean(q)

        return qmean
    
    def estimateQGreedy(self):
        lg=self.done_placeholder*self.action_logits
        eval_op=tf.reduce_max(tf.scalar_mul(self.params['discount'],lg),1,keep_dims=False)

        return tf.add(eval_op,self.reward_placeholder) #does this the right thing???
    
    def estimateAction(self):
        oh=tf.one_hot(self.action_placeholder,self.params['actionsize'])
        out=tf.reduce_sum(self.action_logits*oh,1,keep_dims=False)
        
#        gather_indices = tf.range(self.params['batchsize']) * tf.shape(self.action_logits)[1] + self.action_placeholder
#        self.action_predictions = tf.gather(tf.reshape(self.action_logits, [-1]), gather_indices)
        
        return out
#        return self.action_predictions
    
    def getWeights(self):
        return [self.W_conv1,self.b_conv1,self.W_conv2,self.b_conv2,self.W_conv3,self.b_conv3,self.W_fc1,self.b_fc1,self.W_fc2,self.b_fc2]
    
    def updateWeights(self,w):
        holder=[tf.assign(self.W_conv1,w[0]),
                tf.assign(self.b_conv1,w[1]),
                tf.assign(self.W_conv2,w[2]),
                tf.assign(self.b_conv2,w[3]),
                tf.assign(self.W_conv3,w[4]),
                tf.assign(self.b_conv3,w[5]),
                tf.assign(self.W_fc1,w[6]),
                tf.assign(self.b_fc1,w[7]),
                tf.assign(self.W_fc2,w[8]),
                tf.assign(self.b_fc2,w[9])]
        
        self.sess.run(holder)


    def _weight_variable(self,shape,name=None):
        initial = tf.truncated_normal(shape, stddev=0.02)
        return tf.Variable(initial,trainable=self.train,name=name)

    def _bias_variable(self,shape,name=None):
        initial = tf.constant(0., shape=shape)
        return tf.Variable(initial,trainable=self.train,name=name)

    def _conv2d(self,x, W, s):
        return tf.nn.conv2d(x, W, strides=[1, s, s, 1], padding='VALID')

class DQNAgent(object):
    
    def __init__(self,sess,env,params):
        self.params=params
        self.xpsize=params['replaymemory']
        self.cnt=0
        self.env=env
        self.sess=sess
        self.current_loss=0
        
#        self.run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
#        self.run_metadata = tf.RunMetadata()
        
        self.last_reward=tf.Variable(0,name="cum_reward",dtype=tf.float32,trainable=False)
        self.last_q=tf.Variable(0,name="cum_q",dtype=tf.float32,trainable=False)
        self.last_rate=tf.Variable(0,name="rate",dtype=tf.float32,trainable=False)
        self.last_steps=tf.Variable(0,name="episode_steps",dtype=tf.float32,trainable=False)
        self.epoche_reward=tf.Variable(0,name="epoche_reward",dtype=tf.float32,trainable=False)
        self.epoche_value=tf.Variable(0,name="epoche_value",dtype=tf.float32,trainable=False)
        self.epoche_maxreward=tf.Variable(0,name="epoche_max_reward",dtype=tf.float32,trainable=False)
        self.eps=params['initexploration']
        self.q_predict=QNet(sess,"prediction",params)
        self.q_target=QNet(sess,"target",params,train=False)
        
        self.initBuffers()
        self.initTraining()
        self.initSummaries()
        
        self.rpm=RPM(params['replaymemory'])
        
#        os.mkdir(self.params['traindir'])
        subdir=datetime.datetime.now().strftime('%d%m%y_%H%M%S')
        self.traindir=os.path.join(params['traindir'], "run_%s"%subdir)
        os.mkdir(self.traindir)
        self.picdir=os.path.join(self.traindir,"pics")
        os.mkdir(self.picdir)
        checkpoint_dir=os.path.join(self.traindir,self.params['checkpoint_dir'])
        os.mkdir(checkpoint_dir)
        
        self.saver = tf.train.Saver()
        
        if params["latest_run"]:
            self.latest_traindir=os.path.join(params['traindir'], "run_%s"%params["latest_run"])
            latest_checkpoint = tf.train.latest_checkpoint(os.path.join(self.latest_traindir,self.params['checkpoint_dir']))
            if latest_checkpoint:
                print("Loading model checkpoint {}...\n".format(latest_checkpoint))
                self.saver.restore(sess, latest_checkpoint)
        
        self.merged = tf.summary.merge_all()
        self.train_writer = tf.summary.FileWriter(self.traindir,sess.graph)
                
        init = tf.global_variables_initializer()
        
        sess.run(init)
        self.q_target.updateWeights(self.q_predict.getWeights())
#        sess.graph.finalize()
    
    def __del__(self):
        self.train_writer.close()
        
    def initTraining(self):
#        self.optimizer = tf.train.RMSPropOptimizer(self.params['learningrate'],self.params['gradientmomentum'],
#                                                   self.params['mingradientmomentum'],1e-6)
        self.optimizer = tf.train.RMSPropOptimizer(self.params['learningrate'],momentum=0.95, epsilon=0.01)
        
        
        self.global_step = tf.Variable(0, trainable=False)
        self.eps_op=tf.train.polynomial_decay(params['initexploration'], self.global_step,
                                          params['finalexpframe'], params['finalexploration'],
                                          power=1)
        
        qpred=self.q_predict.estimateAction()
        qtarget=self.q_target.estimateQGreedy()
        diff=qtarget-qpred
        
#        self.losses = tf.squared_difference(qtarget, qpred) # (r + g*max a' Q_target(s',a')-Q_predict(s,a))
#        self.loss = tf.reduce_mean(self.losses)
        self.loss = tf.reduce_mean(self.td_error(diff))
        
        self.train = self.optimizer.minimize(self.loss,global_step=self.global_step)

    def initBuffers(self):
        self.reward_buffer=deque([])
        self.frame_buffer=deque([])
        self.frame2_buffer=deque([])
        self.action_buffer=deque([])
        self.done_buffer=deque([])
        
    def variable_summaries(self,var):
        with tf.name_scope('summaries'):
            mean = tf.reduce_mean(var)
            tf.summary.scalar('mean', mean)
            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.summary.scalar('stddev', stddev)
            tf.summary.scalar('max', tf.reduce_max(var))
            tf.summary.scalar('min', tf.reduce_min(var))
            tf.summary.histogram('histogram', var)
            
            
    def initSummaries(self):
        with tf.name_scope("episode_stats"):
            tf.summary.scalar('cum_reward', self.last_reward)
            tf.summary.scalar('steps', self.last_steps)
            tf.summary.scalar('rate', self.last_rate)
        with tf.name_scope("epoche_stats"):
            tf.summary.scalar('epoche_reward', self.epoche_reward)
            tf.summary.scalar('epoche_maxreward', self.epoche_maxreward)
            tf.summary.scalar('epoche_value', self.epoche_value)
        with tf.name_scope("prediction_action"):
            self.variable_summaries(self.q_predict.action_logits)
            tf.summary.histogram('histogram',tf.to_float(self.q_predict.greedy_actions))
        
        with tf.name_scope("target_action"):
            self.variable_summaries(self.q_target.action_logits)
            
        with tf.name_scope("loss"):
            tf.summary.scalar('loss_val',self.loss)
            
        with tf.name_scope("epsilon"):
            tf.summary.scalar('eps_val',self.eps_op)
        
    def addTransition(self,t):
        self.rpm.addTransition(t)
            
    def _sampleTransitionBatch(self,batchsize=32):
        sample=self.rpm.sampleTransition()
        
        return {self.q_predict.images_placeholder: np.array(sample[0],dtype=np.float32)/255.,
                self.q_predict.action_placeholder: sample[1],
                self.q_target.reward_placeholder: np.clip(sample[2],-1,1),
                self.q_target.images_placeholder: np.array(sample[3],dtype=np.float32)/255.,
                self.q_target.done_placeholder: np.array(sample[4],dtype=np.float32)}
        
    def saveStats(self,reward,steps=0,rate=0):
        ops=[self.last_reward.assign(reward),
             self.last_steps.assign(steps),
             self.last_rate.assign(rate)]
        
        self.sess.run(ops)
#        reward_file=os.path.join(self.traindir, 'rewards.dat')
#        np.savetxt(reward_file,np.array(data))

    
    def epocheStats(self,reward,q,rmax):
        ops=[self.epoche_value.assign(q),
             self.epoche_reward.assign(reward),
             self.epoche_maxreward.assign(rmax)]
        
        self.sess.run(ops)
        
    def td_error(self,x):
        if self.params["huberloss"]:
            # Huber loss
            try:
                return tf.select(tf.abs(x) < 1.0, 0.5 * tf.square(x), tf.abs(x) - 0.5)
            except:
                return tf.where(tf.abs(x) < 1.0, 0.5 * tf.square(x), tf.abs(x) - 0.5)
        else:
            return tf.square(x)
            
        
    def takeAction(self,state=None,eps_ext=None):
        self.eps=self.eps_op.eval()
        g=0
        
        actions=range(self.params['actionsize'])
        
        if eps_ext is not None:
            if np.random.random()<eps_ext:
                a=np.random.choice(actions)
            else:
                action_index=self.q_predict.estimateActionGreedy(state)
                a=action_index
                g=1
                return a,g
        else:
            if state==None:
                a=np.random.choice(actions)
            else:
                if np.random.random()<self.eps:
                    a=np.random.choice(actions)
                        
                else:
                    action_index=self.q_predict.estimateActionGreedy(state)
                    a=action_index
                    g=1
            
        return a,g
    
    def getLoss(self):
        xp_feed_dict=self._sampleTransitionBatch(batchsize=self.params['batchsize'])
        l=self.sess.run(self.loss,feed_dict=xp_feed_dict)
        return l
        
    def trainNet(self):
        #Needs frameskipping: every C steps reset target weights!

        xp_feed_dict=self._sampleTransitionBatch(batchsize=self.params['batchsize'])
        
        self.sess.run([self.train],feed_dict=xp_feed_dict)
        
        
        
        # Create the Timeline object, and write it to a json
#        tl = timeline.Timeline(self.run_metadata.step_stats)
#        ctf = tl.generate_chrome_trace_format()
#        with open('timeline.json', 'w') as f:
#            f.write(ctf)
        
        if self.global_step.eval()%self.params['summary_steps']==0:
            l,summary=self.sess.run([self.loss,self.merged],feed_dict=xp_feed_dict)
            self.current_loss=l
            self.train_writer.add_summary(summary, self.global_step.eval())
        
        if self.global_step.eval()%self.params['checkpoint_steps']==0:
            checkpoint_file = os.path.join(self.traindir,self.params['checkpoint_dir'], 'checkpoint')
            name=self.saver.save(self.sess, checkpoint_file, global_step=self.global_step.eval())
            print("Saving checkpoint: %s"%name)
            
        
        return self.current_loss
        
    def resetTarget(self):
        #reset target weights every C steps; put in main loop
        self.q_target.updateWeights(self.q_predict.getWeights())
        

    def _writeFrame(self,frame,episode,timestep,picdir):
        ep_dir=os.path.join(picdir,"episode_%.5d"%episode)
        if not os.path.exists(ep_dir):
            os.mkdir(ep_dir)
        name = os.path.join(ep_dir,"step_%.4d.png"%timestep)
        cv2.imwrite(name,frame)
        
    def writeFrame(self,frame,episode,timestep):
        self._writeFrame(frame,episode,timestep,self.picdir)
        
        
def rescaleFrame(frame):
    ret = np.array(cv2.resize(frame,(84,84)),dtype=np.uint8)
    return ret

def getYChannel(frame):
    xyz = cv2.cvtColor(frame, cv2.COLOR_RGB2XYZ)
    return xyz

if __name__ == '__main__':      
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-E","--env", type=str, help="Atari Environment in Gym, (default: Breakout-v0)",default='Breakout-v0')
    parser.add_argument("-d","--dir", type=str, help="Directory where the relevant training info is stored")
    parser.add_argument("-e","--eval", type=str, help="Evaluation directory. Movies are stored here.")
    parser.add_argument("-c","--checkpoint",type=str, help="Directory of latest checkpoint.")
    args = parser.parse_args()
        
    envname=args.env
    env = gym.make(envname)
    evalenv = gym.make(envname)
    
    params={
            "Env":'Breakout-v0',
            "episodes":1000,
            "epoches":1000,
            "testruns":30,
            "testeps":0.05,
            "testevery":150000,
            "timesteps":20000,#10000,
            "batchsize":32,
            "replaymemory":250000,
            "targetupdate":40000,
            "discount":0.99,
            "learningrate":0.00025,#0.00025,
            "huberloss":True,
            "gradientmomentum":0.99,
            "sqgradientmomentum":0.95,
            "mingradientmomentum":0.00,
            "initexploration":1.0,
            "finalexploration":0.1,
            "finalexpframe":250000,
            "replaystartsize":50000,
            "framesize":84,
            "frames":4,
            "actionsize": env.action_space.n,
            "traindir":"./train_dir",
            "summary_steps":500,
            "skip_episodes": 50,
            "framewrite_episodes":100,
            "checkpoint_dir":'checkpoints',
            "checkpoint_steps":200000,
            "latest_run":args.checkpoint,
            "metricupdate":40
    }
    
    params["Env"]=envname
    
    
    tf.reset_default_graph()
    
    with tf.Session() as sess:
        
        dqa=DQNAgent(sess,env,params)
        
        np.save(os.path.join(dqa.traindir,'params_dict.npy'), params)
        epoche_name=os.path.join(dqa.traindir,"epoche_stats.tsv")
        epoche_fd=open(epoche_name,'w+')
        
        evalenv = wrappers.Monitor(evalenv, os.path.join(dqa.traindir,'monitor'), video_callable=lambda x:x%20==0)
        
        c=0
        epoche_done=False
        t1Frame=0.0001
        t2Frame=0
        for e in xrange(params['epoches']):
            #episode loop
            print("Starting epoche {}".format(e))
            ep_ctr=0
            t1=time.clock()
            for i in xrange(1,params['episodes']):
                if epoche_done:
                    break

                f = env.reset()
                
                action,_ = dqa.takeAction()
                
                obs=np.zeros((84,84,4),dtype=np.uint8)
                
                for k in range(4):
                    f, r, done, _ = env.step(action)
                    
                    rframe=rescaleFrame(f)
                    fframe=np.array(getYChannel(rframe)[:,:,-1]).astype(np.uint8)
                    obs[:,:,k]=fframe
                
                # time steps
                rewards=[]
                ts=[]
                
                
                rcum=0
                for t in xrange(params['timesteps']):
                    done=False
                    
                    obsNew=np.zeros((84,84,4),dtype=np.uint8)
                    
                    if c<params['replaystartsize']:
                        action,g = dqa.takeAction()
                    else:
                        action,g = dqa.takeAction(obs)
                    
                    
                    rcum_steps=0    
                    for k in range(4):
                        f, r, d, _ = env.step(action)
                        rframe=rescaleFrame(f)
                        fframe=getYChannel(rframe)[:,:,-1]
                        obsNew[:,:,k]=fframe
                        
                        c+=1
                        rcum_steps+=r
                        rcum+=r
                        ep_ctr+=1
                        
                        if d:
                            done=True
                    
                    dqa.addTransition([obs,action, rcum_steps,obsNew, np.array(params['actionsize']*[(not done)],dtype=np.bool)])
                    
                    obs=obsNew
                    
                    
                    loss=-1.
                    if c>=params['replaystartsize']:
                        loss=dqa.trainNet()
                        
                    if c%params["testevery"]==0:
                        epoche_done=True
                    
                    if c%params['targetupdate']==0:
                        t1Frame=time.clock()
                        dqa.resetTarget()
                        t2Frame=time.clock()
                    
                    
                    if done: 
                        if i%params["metricupdate"]==0:
                            dtFrame=(t2Frame-t1Frame)
                            t2=time.clock()
                            if t>0:
                                rate=ep_ctr/(t2-t1)
                                print("\r[Epis: {} || it-rate: {} || Loss: {} || db time: {}|| Frame: {}]".format(i,rate,loss,dtFrame,c),end='')
                            
                            sys.stdout.flush()
                            dqa.saveStats(rcum,t,ep_ctr/(t2-t1))
                        break
                    
                
                
            
            testq=[]
            testreward=[]                    
            for s in range(1,params['testruns']):
                f = evalenv.reset()
                
                action,_ = dqa.takeAction()
                
                obs=np.zeros((84,84,4),dtype=np.uint8)
                obsNew=np.zeros((84,84,4),dtype=np.uint8)
                for k in range(4):
                    f, r, done, _ = env.step(action)
                    
                    rframe=rescaleFrame(f)
                    fframe=np.array(getYChannel(rframe)[:,:,-1]).astype(np.uint8)
                    obs[:,:,k]=fframe
                
                rcum=r
                qmean=[]
                done=False
                for t in xrange(params['timesteps']):
                    action,g = dqa.takeAction(obs,params['testeps'])
                    
                    for k in range(4):
                        f, r, d, _ = evalenv.step(action)
                        rframe=rescaleFrame(f)
                        fframe=getYChannel(rframe)[:,:,-1]
                        obsNew[:,:,k]=fframe
                        
                        rcum+=r
                        
                        if d:
                            done=True
                            break
                    
                    q=dqa.q_predict.meanQ(obsNew)
                    qmean.append(q)
                    
                    obs=obsNew
                    
                    if done:
                        testq.append(np.mean(qmean))
                        testreward.append(rcum)
                        if s%10==0:
                            print("[Test: {} || Reward: {} || Mean Q: {}]".format(s,rcum,np.mean(qmean)))
#                        sys.stdout.flush()
                        break
            
            qepoche=np.mean(testq)
            qepoche_std=np.std(testq)
            repoche=np.mean(testreward)
            rmax=np.max(testreward)
            repoche_std=np.std(testreward)
            epoche_fd.write("%d\t%.5f\t%.5f\t%.5f\t%.5f\n"%(e,qepoche,qepoche_std,repoche,repoche_std))
            dqa.epocheStats(repoche,qepoche,rmax)
            print("Test stats after epoche {}: Q: {} ({}) || R: {} ({})".format(e,qepoche,qepoche_std,repoche,repoche_std)) 
            epoche_done=False
                    
                
        epoche_fd.close()
        env.close()
    
    

    
