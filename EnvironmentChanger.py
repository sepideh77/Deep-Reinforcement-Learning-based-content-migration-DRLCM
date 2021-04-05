#-*- coding: utf-8 -*-
import numpy as np
import random
from random import randint




class EnvironmentChanger(object):
    args =  get_all_config_parameters()
    def __init__(self, args):
    
        self.ContentSize = contentSize
        self.mincontentsetLength = mincontentsetLength
        self.maxcontentsetLength = maxcontentsetLength
        self.numcontents = numDescriptors
        self.serviceLength = np.zeros(self.batchSize,  dtype='int32')
        self.Length_plc_chain = Length_plc_chain
        self.num_rsu = num_rsu
        self.num_Nomadic_Caches = num_Nomadic_Caches
        self.num_Plcd_NS_contents = num_Plcd_NS_contents
        self.num_s_contents = num_s_contents
        self.Maximum_numreq_handled_CacheNode = Maximum_numreq_handled_CacheNode


    def getNewState(self):
        """ Generate new batch of service chain """

        # Clean attributes
        num_total_cache_nodes = self.num_rsu + self.num_Nomadic_Caches
        self.serviceLength = np.zeros(self.batchSize,  dtype='int32')
        self.serviceLength[:] = self.Length_plc_chain
        self.state = np.zeros((self.batchSize, self.Length_plc_chain),  dtype='int32')

        # Compute random services

        for batch in range(self.batchSize):
            sub_chain_part1 = []  # Specifying placement of non-safety contents on nomadic caches
            sub_chain_part2 = []  # Specifying placement of safety contents on RSUs.
            sub_chain_part3 = []  # specifying number of req for non_safety content from each region of nomadic caches to be replyed by each cache node

            sub_chain_part1 = np.random.permutation(range(self.num_Plcd_NS_contents))
            sub_chain_part2 = np.random.permutation(range(self.num_Plcd_NS_contents,self.num_Plcd_NS_contents+self.num_s_contents))
            remaining_elements_lenght = self.Length_plc_chain - len(sub_chain_part1) - len(sub_chain_part2)
            for i in range(remaining_elements_lenght):
              element = randint(0, self.Maximum_numreq_handled_CacheNode)
              sub_chain_part3.append(element)
            self.state[batch] =  [*sub_chain_part1, *sub_chain_part2, *sub_chain_part3]

            #self.serviceLength[batch] = [np.asarray(list(d))]


if __name__ == "__main__":

    # Define generator
    batch_size = 5
    minServiceLength = 2
    maxServiceLength = 6
    numDescriptors = 8

    env = ServiceBatchGenerator(batch_size, minServiceLength, maxServiceLength, numDescriptors)
    env.getNewState()



def get_all_config_parameters():
    # costomize this function so that your selected input features changes in the environment
    pass