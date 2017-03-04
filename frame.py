#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 20 23:18:10 2017

@author: daniel
"""

import numpy as np
import cv2

class FrameBatch(object):
    
    def __init__(self,sess,size=[84,84],frames=4,makeGif=False):
        self.size=size
        self.frames=4
        self.sess=sess
        self.batchReady=False
        self.frameLst=[]
        self.cnt=0
        self.shape=[size[0],size[1],frames]
        self.makeGif=makeGif
        
    def _rescaleFrame(self,frame):
        ret = cv2.resize(frame,(84,84))
        return ret

    def _getYChannel(self,frame):
        xyz = cv2.cvtColor(frame, cv2.COLOR_RGB2XYZ)
        return xyz
    
    def addFrame(self,frame):
        """
            This function is supposed to be used in a while loop to create
            new batches of frames
        """
        
        rframe=self._rescaleFrame(frame)
        
        fframe=np.array(self._getYChannel(rframe)[:,:,-1]).astype(np.uint8)
        self.frameLst.append(fframe)
        self.cnt+=1
        
        if self.cnt>self.frames:
            print "Error in preprocessing: more than %d frames in batch!"%self.frames
            return False
        elif self.cnt==self.frames:
            self.finalBatch=np.array(self.frameLst)
            return True
        else:
            return False
    
    def getNextBatch(self):
        if self.makeGif:
            print "making gif"
        else:
            del self.frameLst[:]
        return np.transpose(self.finalBatch,(1,2,0))
