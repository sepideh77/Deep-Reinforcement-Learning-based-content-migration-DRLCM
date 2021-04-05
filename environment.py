
import numpy as np
import matplotlib.pyplot as plt
from numpy  import array
import math
from itertools import accumulate, chain, repeat, tee




def extract(text, sub1, sub2):

    pos1 = text.lower().find(sub1) + len(sub1)
    pos2 = text.lower().find(sub2)
    result = str(text[pos1:pos2])
    return result



#input path for locations of vehicles: GET FROM sumo simulator#
input_file_path = 'C:/Users/umroot/PycharmProjects/JournalWork/DRLCM/Traces/sumoTrace.xml'


def get_V_locations(self):

    Vehicles_Positions = {}
    lines = []
    with open(input_file_path, "r") as f:
        lines = f.readlines()
        Current_timeslot = extract(lines[22], '"', '">')
        Trace = lines[23:40]
        for line in Trace:
            Vehicle_id = extract(line, 'vehicle id="', '" ')
            lane = extract(line, 'lane="', '" slope')
            Vehicles_Positions[Vehicle_id] = lane
    with open(input_file_path, "w") as f:
        for line in lines[0:21]:
            f.write(str(line))
            if str(line) != '\n':
               f.write('\n')
        for line in lines[42:]:
            f.write(str(line))
            if str(line) != '\n':
               f.write('\n')










    return Vehicles_Positions






def xrange(x):

    return iter(range(x))



def chunks(l, n, m):
    """Yield successive n-sized chunks from l."""
    i = n
    list =[]
    while i < m:
        list.append(l[i])
        i = i+1
    return list

def chunks_evenly(xs, n):
        assert n > 0
        L = len(xs)
        s, r = divmod(L, n)
        widths = chain(repeat(s + 1, r), repeat(s, n - r))
        offsets = accumulate(chain((0,), widths))
        b, e = tee(offsets)
        next(e)
        return [xs[s] for s in map(slice, b, e)]


def Convert_Placement(self,Plc):
    sub1_Plc_chain = chunks(Plc, 0, self.num_Plcd_NS_contents)
    sub2_Plc_chain = chunks(Plc, self.num_Plcd_NS_contents, self.num_s_contents + self.num_Plcd_NS_contents)
    sub3_Plc_chain = chunks(Plc, self.num_s_contents + self.num_Plcd_NS_contents, self.serviceLength)

    max_sub1 = max(sub1_Plc_chain)
    min_sub1 = min(sub1_Plc_chain)

    max_sub2 = max(sub2_Plc_chain)
    min_sub2 = min(sub2_Plc_chain)

    max_sub3 = max(sub3_Plc_chain)
    min_sub3 = min(sub3_Plc_chain)

    Total_cache_nodes = self.num_rsu + self.num_Nomadic_Caches
    converted_subplc1 = []
    converted_subplc2 = []
    converted_subplc3 = []

    for element in sub1_Plc_chain:
        conv_element = int((Total_cache_nodes-1 - 0)*((element - min_sub1)/(max_sub1 - min_sub1)))
        converted_subplc1.append(conv_element)

    for element in sub2_Plc_chain:
        a = self.num_Nomadic_Caches
        b = self.num_Nomadic_Caches+self.num_rsu -1
        try:
            conv_element = int((b-a) * ((element - min_sub2) / (max_sub2 - min_sub2))+ a)
        except:
            conv_element = 0
        converted_subplc2.append(conv_element)

    for element in sub3_Plc_chain:
        conv_element = int((self.Capacity_RSUs - 0) * ((element - min_sub3) / (max_sub3 - min_sub3)))
        converted_subplc3.append(conv_element)

    converted_plc = converted_subplc1 + converted_subplc2 + converted_subplc3
    return converted_plc





       # else:
         #if i < self.num_Plcd_NS_contents + self.num_s_contents:



    pass


def create_cells(self):
    Lists =[]
    i = 0
    while  i< self.num_Nomadic_Caches:
        row = np.empty(self.Capacity_nomadic_caches)
        row [:] = np.nan
        Lists.append( row )
        i = i+1
    i = 0
    while i < self.num_rsu:
        row = np.empty(self.Capacity_RSUs)
        row[:] = np.nan
        Lists.append( row )
        i = i+ 1
    return Lists


def Compute_Y(self, m, i):
    if isinstance(self, Environment):
        Cells = self.cells
        cache_node = i
        NS_content = m
        if NS_content in Cells[cache_node]:
            return 1
        else:
            return 0
    else:
        Compute_Y_pervt()

def Compute_X(self, m_prime, r):
    Cells = self.cells
    RSU = r
    S_content = m_prime
    if S_content in Cells[RSU]:
        return 1
    else:
        return 0

def Compute_C(self,PLC_chunck3,m,j,i):

    NScontentsLists = chunks_evenly (PLC_chunck3,self.num_Plcd_NS_contents)
    list = NScontentsLists[m]
    Regions = chunks_evenly (list,self.num_Nomadic_Caches)
    try:
        Region = Regions [j]
    except:
        Region = 0
    values = chunks_evenly (Regions[j],self.num_Nomadic_Caches + self.num_rsu)
    value = values[i]
    return value[0]


def Compute_Y_pervt(self, Initial_Plc, m, i):
    Init_plc_cells = array(create_cells(self))
    Init_service = list(range(self.num_s_contents +self.num_Plcd_NS_contents))
    placement = Initial_Plc
    for i in range(self.num_descriptors):
        content = Init_service[i]
        suggested_cache = int(placement[i])
        row = Init_plc_cells[suggested_cache]
        for i in range(len(row)):
            if math.isnan(row[i]):
                row[i] = content
                Init_plc_cells[suggested_cache] =row
                break
    cache_node = i
    NS_content = m
    try :
      if NS_content in Init_plc_cells[cache_node]:
        return 1
      else:
        return 0
    except:
        return 0


def Compute_Plus_braket(a, b):
    value = a - b
    if value > 0:
        return 1
    else:
        return 0




def Compue_Cmig_PartialCost1(self, placement, CN, M, V_m, g, Initial_Plc):
    Cmig_PartialCost1 = 0
    for i in range(CN):
        for m in range(M):
            Y_t = Compute_Y(self, m, i)
            Y_perv_t = Compute_Y_pervt(self,Initial_Plc, m, i)
            Plus_braket = Compute_Plus_braket(Y_perv_t,Y_t)
            Cmig_PartialCost1 = Cmig_PartialCost1 + V_m * g * Plus_braket
    return Cmig_PartialCost1


def Compue_Cmig_PartialCost2(self, placement, CN, M, V_m, p, Initial_Plc):
    Cmig_PartialCost2 = 0
    for i in range(CN):
        for m in range(M):
            Y_t = Compute_Y(self, m, i)
            Y_perv_t = Compute_Y_pervt(self, Initial_Plc, m, i)
            Plus_braket = Compute_Plus_braket(Y_t,Y_perv_t)
            Cmig_PartialCost2 = Cmig_PartialCost2 + V_m * p * Plus_braket
    return Cmig_PartialCost2


def Compute_loct(self, Initial_Plc, m, j):
    pass


def Get_Cachenode_Type(self,i):
    if i in range(0,self.num_Nomadic_Caches-1):
       return 'Nomadic Cache'
    else:
       return 'RSU'


def Compute_distance(self, i, j, placement, Initial_Plc):
    #test if i is RSU:
    Type_i = Get_Cachenode_Type(self,i)
    Type_j = Get_Cachenode_Type(self,j)
    if Type_i == 'RSU' and Type_j == 'RSU': # both cache nodes are RSUs
        if i == j:
            return 0
        else:
            return  10000
    if Type_i != 'RSU' and Type_j != 'RSU': # both cache nodes are Nomadic Caches
        if i == j:
            return 0.166
        else:
            return abs(i - j)*4
    if Type_i == 'RSU' and Type_j == 'Nomadic Cache':
        return abs(i - j)
    if Type_i == 'Nomadic Cache' and Type_j == 'RSU':
        return abs(i - j)





def Compue_Cmig_PartialCost3(self, placement, CN, M, V_m, O, Initial_Plc):
    Cmig_PartialCost3 = 0
    for i in range(CN):
        for j in range(CN):
            if i == j:
                break
            else:
                 for m in range(M):
                     Yi_t = Compute_Y(self, m, i)
                     Yi_perv_t = Compute_Y_pervt(self, Initial_Plc, m, i)
                     Yj_t = Compute_Y(self, m, j)
                     Yj_perv_t = Compute_Y_pervt(self, Initial_Plc, m, j)
                     plus_braket_i = Compute_Plus_braket(Yi_perv_t,Yi_t)
                     plus_braket_j = Compute_Plus_braket(Yj_t,Yj_perv_t)
                     Distance_iprvt_jt = Compute_distance(self,i,j,placement,Initial_Plc)
                     #---------------------

                     #---------------------
                     Cmig_PartialCost3 = Cmig_PartialCost3 + V_m * O * Distance_iprvt_jt * plus_braket_i * plus_braket_j

    return Cmig_PartialCost3





def Compute_CnsACCDelay(self, placement, CN, M, V_m, o, Initial_Plc):
    d_epsilon = self.small_transfering_delay
    d_C = self.Cloud_transfering_delay
    d = self.transfering_delay
    Q_j_m = self.num_Users_per_Nomadic_caches
    CnsACCDelay = 0
    for i in range(CN):
        for j in range(CN):
            if i == j:
                break
            else:
                for m in range(M):
                    S_j_m = Compute_C(self,placement,m,j,i)
                    C_j_i_m = Compute_C(self,placement,m,j,i)
                    r_j_m = Compute_C(self,placement,m,j,i)
                    Y_j_m = Compute_Y(self,m,j)
                    Y_i_m = Compute_Y(self,m,i)
                    CnsACCDelay = CnsACCDelay + ((S_j_m * Y_j_m * d_epsilon + C_j_i_m * Y_i_m * d*Compute_distance(self,i,j,placement,Initial_Plc)+r_j_m*d_C)/Q_j_m)
    return   CnsACCDelay


def Compute_t_Rem_i_r(self, placement, sgn, i, r, vel_i,L_r):
    A = L_r /2
    B = sgn*abs(i-r)
    D = Compute_Plus_braket(A,B)
    t_Rem_i_r = D / vel_i
    return t_Rem_i_r


def Compute_v_dl_mprime_i_r(self, placement, V_M_prime, t_Rem_i_r, X_mprime_r, tb_i_r):
    A = X_mprime_r * t_Rem_i_r / tb_i_r
    B = V_M_prime
    v_dl_mprime_i_r = min (A,B)
    return v_dl_mprime_i_r


def Compute_t_RSU_dl_mprime_i_r(v_dl_mprime_i_r,tb_i_r):
    t_RSU_dl_mprime_i_r = v_dl_mprime_i_r * tb_i_r
    return t_RSU_dl_mprime_i_r

def Compute_t_c_dl_mprime_i(self,placement,v_dl_mprime_i_r, t_c, V_M_prime,t_Rem_i_r,X_mprime_r,tb_i_r):
    RSU = self.num_rsu
    A = 0
    for r in range(RSU):
        A = A + Compute_v_dl_mprime_i_r(self,placement,V_M_prime,t_Rem_i_r,X_mprime_r,tb_i_r)
    B = V_M_prime
    t_c_dl_mprime_i = Compute_Plus_braket(A,B)* t_c
    return t_c_dl_mprime_i


def Compute_CsDownloadDelay(self, placement, NC_zegond, M_prime, V_M_prime, L_r, Initial_Plc):
   CsDownloadDelay = 0
   t_c = self.Cloud_transfering_delay
   tb_i_r = self.transfering_delay
   vel_i = self.Nomadic_Caches_Velocity
   sgn = +1
   t_RSU_dl_mprime_i_r = 0
   v_dl_mprime_i_r =0
   t_Rem_i_r = 0
   X_mprime_r = 0
   for i in range(NC_zegond):
       for m_prime in range(M_prime):
           for r in range(self.num_rsu):
               if i == r:
                   break
               else:
                    t_Rem_i_r = Compute_t_Rem_i_r(self,placement,sgn,i,r,vel_i,L_r)
                    X_mprime_r = Compute_X(self,m_prime,r)
                    v_dl_mprime_i_r = Compute_v_dl_mprime_i_r(self,placement,V_M_prime,t_Rem_i_r,X_mprime_r,tb_i_r)
                    t_RSU_dl_mprime_i_r = t_RSU_dl_mprime_i_r + Compute_t_RSU_dl_mprime_i_r(v_dl_mprime_i_r,tb_i_r)
           t_c_dl_mprime_i = Compute_t_c_dl_mprime_i(self,placement,v_dl_mprime_i_r, t_c, V_M_prime,t_Rem_i_r,X_mprime_r,tb_i_r)
       CsDownloadDelay = CsDownloadDelay + t_c_dl_mprime_i + t_RSU_dl_mprime_i_r
   return CsDownloadDelay


def times_so_far(ls):
   return [ls[:i].count(ls[i]) for i in xrange(len(ls))]


class Environment(object):

    def __init__(self,Capacity_RSUs, Capacity_nomadic_caches, Cloud_transfering_delay, Diameter_rsus, G, Maximum_numreq_handled_CacheNode, Nomadic_Caches_Velocity, O, P, num_Nomadic_Caches, num_Plcd_NS_contents, num_Targeted_Nomadic_caches, num_Users_per_Nomadic_caches, num_descriptors, num_rsu, num_s_contents, size_of_contents, small_transfering_delay, transfering_delay, w1, w2, w3, w4, w5, w6,servicelenght,Initial_Placement):


        # Environment properties
        self.Capacity_RSUs = Capacity_RSUs
        self.Capacity_nomadic_caches = Capacity_nomadic_caches
        self.Diameter_rsus = Diameter_rsus
        self.G = G
        self.Maximum_numreq_handled_CacheNode = Maximum_numreq_handled_CacheNode
        self.Nomadic_Caches_Velocity = Nomadic_Caches_Velocity
        self.O = O
        self.P = P
        self.num_Nomadic_Caches = num_Nomadic_Caches
        self.num_Plcd_NS_contents = num_Plcd_NS_contents
        self.num_Targeted_Nomadic_caches = num_Targeted_Nomadic_caches
        self.num_Users_per_Nomadic_caches = num_Users_per_Nomadic_caches
        self.num_descriptors = num_descriptors
        self.num_rsu = num_rsu
        self.num_s_contents = num_s_contents
        self.size_of_contents = size_of_contents
        self.small_transfering_delay = small_transfering_delay
        self.transfering_delay = transfering_delay
        self.Cloud_transfering_delay = Cloud_transfering_delay
        self.w1 = w1
        self.w2 = w2
        self.w3 = w3
        self.w4 = w4
        self.w5 = w5
        self.w6 = w6

        num_Cache_Nodes = num_Nomadic_Caches + num_rsu
        num_Non_Safety_Contents = num_Plcd_NS_contents
        self.Fixed_Initaial_Placement = np.nan
        self.cells = np.empty((num_Cache_Nodes, self.Capacity_nomadic_caches))
        self.cells[:] = np.nan
        self.service_properties = [{"size": 1} for _ in range(self.num_descriptors)]

        # Placement properties
        self.serviceLength = servicelenght
        self.service = None
        self.placement = None
        self.first_slots = None
        self.reward = 1
        self.invalidPlacement = False
        self.Initial_Placement = Initial_Placement

        # Assign ns properties within the environment
        self._get_service_propertieses()


    def _get_service_propertieses(self):
        """ Packet properties """
        # By default the size of each package in that environment is 1, should be modified here.
        for i in range(len(self.service_properties)):
            self.service_properties[i]["size"] = self.size_of_contents

    def _placeSubPakcet(self, cache_node, content):
        """ Place subPacket """

        occupied_slot = None
        for slot in range(len(self.cells[cache_node])):
            if np.isnan(self.cells[cache_node][slot]):
                self.cells[cache_node][slot] = content
                occupied_slot = slot
                break
            elif slot == len(self.cells[cache_node])-1:
                self.invalidPlacement = True
                occupied_slot = -1      # No space available
                break
            else:
                pass                    # Look for next slot

        return occupied_slot

    def _placePacket(self, i, cache_node, content):
        """ Place Packet """

        for slot in range(self.service_properties[content]["size"]):
            occupied_slot = self._placeSubPakcet(cache_node, content)

            # Anotate first slot used by the Packet
            if slot == 0:
                self.first_slots[i] = occupied_slot

    def _computeReward(self, placement):




        """ Compute reward """
        CN = self.num_Nomadic_Caches +self.num_rsu
        NC_prime = self.num_Targeted_Nomadic_caches
        NC_zegond =  NC_prime
        M = self.num_Plcd_NS_contents
        M_prime = self.num_s_contents
        V_m = self.size_of_contents
        V_M_prime = self.size_of_contents
        L_r = self.Diameter_rsus
        g = self.G
        p = self.P
        o = self.O
        Initial_Plc = self.Initial_Placement
        w1 = self.w1
        w2 = self.w2
        w3 = self.w3
        w4 = self.w4
        w5 = self.w5
        w6 = self.w6
        """ Compute locations of the current time slot """
        V_Locations =[]
        V_Locations = get_V_locations(self)









        Cmig_PartialCost1 = Compue_Cmig_PartialCost1 (self,placement,CN,M,V_m,g,Initial_Plc)
        Cmig_PartialCost2 = Compue_Cmig_PartialCost2 (self,placement,CN,M,V_m,p,Initial_Plc)
        Cmig_PartialCost3 = Compue_Cmig_PartialCost3 (self,placement,CN,M,V_m,o,Initial_Plc)
        Cmigration = w1 * Cmig_PartialCost1 + w2 * Cmig_PartialCost2 + w3 * Cmig_PartialCost3


        try:
            CnsACCDelay = Compute_CnsACCDelay(self,placement,CN,M,V_m,o,Initial_Plc)
        except:
            CnsACCDelay = 1000
        CsDownloadDelay = Compute_CsDownloadDelay(self,placement,NC_zegond,M_prime,V_M_prime,L_r,Initial_Plc)

        CTotal = w4 * CnsACCDelay + w5 * Cmigration + w6 * CsDownloadDelay
        reward =  -  CTotal
        return reward

    def step(self, placement, service, length):
        """ Place service """
        self.service = service
        self.serviceLength = length
        placement = Convert_Placement(self, placement)

        self.first_slots = np.zeros(length, dtype='int32')

        sub1_Service_chain = chunks(self.service, 0, self.num_Plcd_NS_contents)
        sub2_Service_chain = chunks(self.service, self.num_Plcd_NS_contents, self.num_s_contents+ self.num_Plcd_NS_contents )
        sub3_Service_chain = chunks(self.service, self.num_s_contents+ self.num_Plcd_NS_contents, self.serviceLength )


        for i in range(self.num_descriptors):
            self._placePacket(i, placement[i], service[i])

        """ Compute reward """
        if self.invalidPlacement == True:
            self.reward = 1
        else:
            self.reward = self._computeReward(placement)

    def clear(self):
        """ Clean environment """
        num_Cache_Nodes = self.num_Nomadic_Caches +self.num_rsu
        num_Non_Safety_Contents = self.num_Plcd_NS_contents
        self.cells = array(create_cells(self))
        self.serviceLength = 0
        self.service = None
        self.placement = None
        self.first_slots = None
        self.reward = 1
        self.invalidPlacement = False

    def render(self, epoch=0):
        """ Render environment using Matplotlib """

        # Creates just a figure and only one subplot
        fig, ax = plt.subplots()
        ax.set_title(f'Environment {epoch}\nreward: {self.reward/1000}')

        margin = 3
        margin_ext = 6
        xlim = 200
        ylim = 200


        numBins = self.num_rsu + self.num_Nomadic_Caches
        numSlots_Nomadic_caches = self. Capacity_nomadic_caches
        numSlots_RSUs = self.Capacity_RSUs
        numSlots = max(numSlots_Nomadic_caches,numSlots_RSUs)

        # Set drawing limits
        plt.xlim(0, xlim)
        plt.ylim(-ylim, 0)

        # Set hight and width for the box
        high = np.floor((ylim - 2 * margin_ext - margin * (numBins - 1)) / numBins)
        wide = np.floor((xlim - 2 * margin_ext - margin * (numSlots - 1)) / numSlots)

        # Plot slot labels
        for slot in range(numSlots):
            x = wide * slot + slot * margin + margin_ext
            plt.text(x + 0.5 * wide, -3, "     ".format(slot), ha="center", family='sans-serif', size=8)

        # Plot bin labels & place empty boxes
        for bin in range(numBins):
            y = -high * (bin + 1) - (bin) * margin - margin_ext
            if bin < self.num_Nomadic_Caches:
                plt.text(0, y + 0.5 * high, "Nomadic Cache{}:".format(bin), ha="center", family='sans-serif', size=8)
            else:
                plt.text(0, y + 0.5 * high, "RSU {}:   ".format(bin - self.num_Nomadic_Caches), ha="center", family='sans-serif', size=8)
            for slot in range(numSlots):
                x = wide * slot + slot * margin + margin_ext
                #rectangle = mpatches.Rectangle((x, y), wide, high, linewidth=1, edgecolor='black', facecolor='none')
                #ax.add_patch(rectangle)

        # Select serviceLength colors from a colormap
        cmap = plt.cm.get_cmap('hot')
        colormap = [cmap(np.float32(i+1)/(self.serviceLength+1)) for i in range(self.serviceLength)]

        # Plot service boxes
        repeated_cache_node = []
        for i in range(self.num_descriptors):
            content = self.service[i]
            # find which cache node contains the content
            for i in range(self.cells.size):
                if content in self.cells[i]:
                    cache_node = i
                    repeated_cache_node.append(cache_node)
            pkt = content
            bin = cache_node
            count = repeated_cache_node.count(cache_node)
            first_slot = count*2
            idx = i
            for k in range(self.service_properties[pkt]["size"]):
                slot = first_slot + k
                x = wide * slot + slot * margin + margin_ext
                y = -high * (bin + 1) - bin * margin - margin_ext
                #rectangle = mpatches.Rectangle((x, y), wide, high, linewidth=0, alpha=.9)
                #ax.add_patch(rectangle)
                if content < self.num_Plcd_NS_contents :
                     plt.text(x + 0.5 * wide, y + 0.5 * high, "      NS{}   ".format(pkt), ha="center", family='sans-serif', size=8)
                     print (self.cells)
                else:
                    plt.text(x + 0.5 * wide, y + 0.5 * high, "      S{}     ".format(pkt), ha="center",   family='sans-serif', size=8)
                    print(self.cells)


        plt.axis('off')
        plt.show()


if __name__ == "__main__":

    # Define environment
    numBins = 0
    numSlots = 0
    numDescriptors = 0
    env = Environment(numBins, numSlots, numDescriptors)

    # Allocate service in the environment
    servicelength = 0 # number of placement to be considered
    ns = []#service
    placement = []
    env.step(placement, ns, servicelength)
    env.render()
    env.clear()
