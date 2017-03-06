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
from frame import FrameBatch
from collections import deque  
import datetime
import cv2
import os
import io
import sys
from gym import wrappers

class QNet(object):
    def __init__(self,sess,name,params,train=True):
        self.params=params
        self.sess=sess
        self.name=name
        self.input_shape=[None ,params['framesize'],params['framesize'],params['frames']] #add to hyperparamters
        self.images_placeholder = tf.placeholder(tf.float32,shape=self.input_shape)
        self.target_placeholder = tf.placeholder(tf.int32,shape=[None,params['actionsize']])
        self.reward_placeholder = tf.placeholder(tf.float32,shape=[None,1])
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
        prediction_index = self.sess.run(self.greedy_actions,
                          feed_dict={self.images_placeholder: feed})

        return prediction_index[0]
    
    def meanQ(self,state_feed):
        feed=np.expand_dims(state_feed,axis=0)
        q = self.sess.run(self.action_logits,
                          feed_dict={self.images_placeholder: feed})
        
        qmean=np.mean(q)

        return qmean
    
    def estimateQGreedy(self):
        lg=self.action_logits*self.done_placeholder
        eval_op=tf.reduce_max(tf.scalar_mul(self.params['discount'],lg),1,keep_dims=False)

        return tf.add(eval_op,self.reward_placeholder) #does this the right thing???
    
    def estimateAction(self):
#        oh=tf.one_hot(self.action_placeholder,self.params['actionsize'])
#        out=self.action_logits*oh
        
        gather_indices = tf.range(self.params['batchsize']) * tf.shape(self.action_logits)[1] + self.action_placeholder
        self.action_predictions = tf.gather(tf.reshape(self.action_logits, [-1]), gather_indices)
        
#        return oh,out
        return self.action_predictions
    
    def getWeights(self):
        return [self.W_conv1,self.b_conv1,self.W_conv2,self.b_conv2,self.W_conv3,self.b_conv3,self.W_fc1,self.b_fc1,self.W_fc2,self.b_fc2]
    
    def updateWeights(self,w):
        tf.assign(self.W_conv1,w[0]).op.run()
        tf.assign(self.b_conv1,w[1]).op.run()
        tf.assign(self.W_conv2,w[2]).op.run()
        tf.assign(self.b_conv2,w[3]).op.run()
        tf.assign(self.W_conv3,w[4]).op.run()
        tf.assign(self.b_conv3,w[5]).op.run()
        tf.assign(self.W_fc1,w[6]).op.run()
        tf.assign(self.b_fc1,w[7]).op.run()
        tf.assign(self.W_fc2,w[8]).op.run()
        tf.assign(self.b_fc2,w[9]).op.run()


    def _weight_variable(self,shape,name=None):
        initial = tf.truncated_normal(shape, stddev=0.05)
        return tf.Variable(initial,trainable=self.train,name=name)

    def _bias_variable(self,shape,name=None):
        initial = tf.constant(0., shape=shape)
        return tf.Variable(initial,trainable=self.train,name=name)

    def _conv2d(self,x, W, s):
        return tf.nn.conv2d(x, W, strides=[1, s, s, 1], padding='SAME')

class DQNAgent(object):
    
    def __init__(self,sess,env,params):
        self.params=params
        self.xpsize=params['replaymemory']
        self.cnt=0
        self.env=env
        self.sess=sess
        
        self.last_reward=tf.Variable(0,name="cum_reward",dtype=tf.float32,trainable=False)
        self.last_q=tf.Variable(0,name="cum_q",dtype=tf.float32,trainable=False)
        self.last_steps=tf.Variable(0,name="episode_steps",dtype=tf.float32,trainable=False)
        self.epoche_reward=tf.Variable(0,name="epoche_reward",dtype=tf.float32,trainable=False)
        self.epoche_value=tf.Variable(0,name="epoche_value",dtype=tf.float32,trainable=False)
        self.eps=params['initexploration']
        self.q_predict=QNet(sess,"prediction",params)
        self.q_target=QNet(sess,"target",params,train=False)
        
        self.initBuffers()
        self.initTraining()
        self.initSummaries()
        
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
        #Summaries: Max Q (from action request) for episode (need reset func called after each episode), timesteps per episode
        #avg q per episode (laser 20 or so), cumm reward per episode, avg last 20 reward per episode, loss,eps,gifs?
    
    def __del__(self):
        self.train_writer.close()
        
    def initTraining(self):
        self.optimizer = tf.train.RMSPropOptimizer(self.params['learningrate'],self.params['gradientmomentum'],
                                                   self.params['mingradientmomentum'],1e-6)
        
        self.global_step = tf.Variable(0, trainable=False)
        self.eps_op=tf.train.polynomial_decay(params['initexploration'], self.global_step,
                                          params['finalexpframe'], params['finalexploration'],
                                          power=1)
        
        qpred=self.q_predict.estimateAction()
        qtarget=self.q_target.estimateQGreedy()
        
        self.losses = tf.squared_difference(qtarget, qpred) # (r + g*max a' Q_target(s',a')-Q_predict(s,a))
        self.loss = tf.reduce_mean(self.losses)
        
        self.train = self.optimizer.minimize(self.loss,global_step=self.global_step)
        print("b logits")

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
            tf.summary.scalar('sum', tf.reduce_sum(var))
#            tf.summary.scalar('maxq', tf.argmax(var))
            tf.summary.scalar('min', tf.reduce_min(var))
            tf.summary.histogram('histogram', var)
            tf.summary.histogram('histogram_sum', tf.reduce_sum(var,1))
            
            
    def initSummaries(self):
        with tf.name_scope("episode_stats"):
            tf.summary.scalar('cum_reward', self.last_reward)
            tf.summary.scalar('steps', self.last_steps)
        with tf.name_scope("epoche_stats"):
            tf.summary.scalar('epoche_reward', self.epoche_reward)
            tf.summary.scalar('epoche_value', self.epoche_value)
        with tf.name_scope("prediction_action"):
            self.variable_summaries(self.q_predict.action_logits)
        
        with tf.name_scope("target_action"):
            self.variable_summaries(self.q_target.action_logits)
            
        with tf.name_scope("loss"):
            tf.summary.scalar('loss_val',self.loss)
            
        with tf.name_scope("epsilon"):
            tf.summary.scalar('eps_val',self.eps_op)
        
    def addTransition(self,t):
        self.frame_buffer.appendleft(t[0])
        self.action_buffer.appendleft(t[1])
        self.reward_buffer.appendleft(t[2])
        self.frame2_buffer.appendleft(t[3])
        self.done_buffer.appendleft(t[4])
        
        
        if len(self.frame_buffer) > self.xpsize:
            self.frame_buffer.pop()
            self.frame2_buffer.pop()
            self.action_buffer.pop()
            self.reward_buffer.pop()
            self.done_buffer.pop()
            
    def _sampleTransitionBatch(self,batchsize=32):
        idx=np.random.randint(0,len(self.frame_buffer),batchsize)
        frame_batch=[]
        frame2_batch=[]
        reward_batch=[]
        action_batch=[]
        done_batch=[]
        
        for j in idx:
            frame_batch.append(np.array(self.frame_buffer[j],dtype=np.float32)/255.)
            frame2_batch.append(np.array(self.frame2_buffer[j],dtype=np.float32)/255.)
            reward_batch.append(np.clip(np.array(self.reward_buffer[j]),-1,1))
            action_batch.append(np.array(self.action_buffer[j]))
            done_batch.append(np.array(self.done_buffer[j]))
        
        return {self.q_predict.images_placeholder: frame_batch,
                self.q_predict.action_placeholder: action_batch,
                self.q_target.reward_placeholder: reward_batch,
                self.q_target.images_placeholder: frame2_batch,
                self.q_target.done_placeholder: done_batch}
        
    def saveRewards(self,data,steps=0):
        self.last_reward.assign(data[-1]).op.run()
        self.last_steps.assign(steps).op.run()
        reward_file=os.path.join(self.traindir, 'rewards.dat')
        np.savetxt(reward_file,np.array(data))
        
    def epocheStats(self,reward,q):
        self.epoche_value.assign(q).op.run()
        self.epoche_reward.assign(reward).op.run()
            
        
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
        
        l,_,summary=self.sess.run([self.loss,self.train,self.merged],feed_dict=xp_feed_dict)
        
        
        if self.global_step.eval()%self.params['summary_steps']==0:
            self.train_writer.add_summary(summary, self.global_step.eval())
        
        if self.global_step.eval()%self.params['checkpoint_steps']==0:
            checkpoint_file = os.path.join(self.traindir,self.params['checkpoint_dir'], 'checkpoint')
            name=self.saver.save(self.sess, checkpoint_file, global_step=self.global_step.eval())
            print("Saving checkpoint: %s"%name)
        
        return l
        
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

if __name__ == '__main__':      
    if len(sys.argv) == 1:
        train_dir="./train_dir"
    else:
        train_dir=sys.argv[1]
    
    env = gym.make('Breakout-v0')
    evalenv = gym.make('Breakout-v0')
    
    params={
            "episodes":1000,
            "epoches":1000,
            "testruns":30,
            "testeps":0.05,
            "timesteps":10000,#10000,
            "batchsize":32,
            "replaymemory":1000000,
            "targetupdate":10000,
            "discount":0.99,
            "learningrate":0.00025,#0.00025,
            "gradientmomentum":0.99,
            "sqgradientmomentum":0.95,
            "mingradientmomentum":0.00,
            "initexploration":1.0,
            "finalexploration":0.1,
            "finalexpframe":1000000,
            "replaystartsize":50000,
            "framesize":84,
            "frames":4,
            "actionsize": env.action_space.n,
            "traindir":train_dir,
            "summary_steps":50,
            "skip_episodes": 50,
            "framewrite_episodes":100,
            "checkpoint_dir":'checkpoints',
            "checkpoint_steps":200000,
            "latest_run":None
    }
    
    tf.reset_default_graph()
    
    

    
    #add gif suppoert to frame class, add every and at the end of the episode save a gif (or every nth episode) with matplotlib?

    with tf.Session() as sess:
        
        dqa=DQNAgent(sess,env,params)
        
        evalenv = wrappers.Monitor(evalenv, os.path.join(dqa.traindir,'monitor'), video_callable=lambda x:x%20==0)

    	epoche_name=os.path.join(dqa.traindir,"epoche_stats.tsv")
        che_fd=open(epoche_name,'w+')
        
        c=0
        
        for e in xrange(params['epoches']):
            #episode loop
            cumRewards=[]
            
            for i in xrange(1,params['episodes']):
                f = env.reset()
                fb_init=FrameBatch(sess)
                
                action,_ = dqa.takeAction()
                
                while fb_init.addFrame(f) is not True:
                    f, r, done, _ = env.step(action)
                    
                obs=fb_init.getNextBatch()
                
                # time steps
                
                rewards=[]
                ts=[]
                done=False
                for t in xrange(params['timesteps']):
                    t1=time.clock()
                    fb=FrameBatch(sess)
                    if c<params['replaystartsize']:
                        action,g = dqa.takeAction()
                    else:
                        action,g = dqa.takeAction(obs)
                        
    
                    rmax=0.
                    while fb.addFrame(f) is not True:
    #                    env.render()
                        f, r, d, _ = env.step(action)   
                        if r>rmax:
                            rmax=r
                        if (i>params['skip_episodes']) and (i%params['framewrite_episodes']==0):
                            dqa.writeFrame(f,i,t)
                        c+=1
                        rewards.append(r)
                        if d:
                            done=True
                    obsNew=fb.getNextBatch()
                    dqa.addTransition([obs,action, [rmax],obsNew, params["actionsize"]*[float((not done))]])
                    
                    loss=-1.
                    if c>=params['replaystartsize']:
                        loss=dqa.trainNet()
                    
                    curr_xp=len(dqa.frame_buffer)
                    if t%40==0:
                        print("\r[Epis: {} || Action: {} ({}) || Loss: {} || Replaybuffer: {}|| Frame: {}]".format(i,action,g,loss,curr_xp,c),end='')
                        sys.stdout.flush()
                        
                    obs=obsNew
                    
                    if c%params['targetupdate']==0: #check this
                               dqa.resetTarget()
                    if done:
                        rSum=np.sum(rewards)
                        cumRewards.append(rSum)
                        dqa.saveRewards(cumRewards,t)
                        break
            
            testq=[]
            testreward=[]                    
            for s in range(1,params['testruns']):
                f = env.reset()
                fb_init=FrameBatch(sess)
                
                action,_ = dqa.takeAction()
                
                while fb_init.addFrame(f) is not True:
                    f, r, done, _ = env.step(action)
                    
                obs=fb_init.getNextBatch()
                
                rcum=r
                qmean=0
                done=False
                for t in xrange(params['timesteps']):
                    fb=FrameBatch(sess)
                    
                    action,g = dqa.takeAction(obs,params['testeps'])
                    q=dqa.q_predict.meanQ(obs)
                    
                    rmax=0.
                    while fb.addFrame(f) is not True:
    #                    env.render()
                        f, r, d, _ = env.step(action)   
                        rcum+=r
                        if d:
                            done=True
                    
                    obs=fb.getNextBatch()
                    q=dqa.q_predict.meanQ(obs)
                    qmean+=q
                    
                    if done:
                        qmean=qmean/s
                        testq.append(qmean)
                        testreward.append(rcum)
                        if s%10==0:
                            print("\r[Test: {} || Reward: {} || Mean Q: {}]".format(s,rcum,qmean),end='')
                        sys.stdout.flush()
                        break
            
            qepoche=np.mean(testq)
            qepoche_std=np.mean(testq)
            repoche=np.mean(testreward)
            repoche_std=np.std(testreward)
            che_fd.write("%d\t%.5f\t%.5f\t%.5f\t%.5f\n"%(e,qepoche,qepoche_std,repoche,repoche_std))
            dqa.epocheStats(repoche,qepoche)
            print("Test stats after epoche {}: Q: {} ({}) || R: {} ({})".format(e,qepoche,qepoche_std,repoche,repoche_std)) 
                    
                
    epoche_fd.close()
    env.close()
    
    

    
