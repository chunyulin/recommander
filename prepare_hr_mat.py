#!/usr/bin/python
import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

### Require:
### 1) Only need -- IP_ID, Resource_ID, Timestamp(can be in binary for speed/space)
### 2) We may need a permanent ID table for IP, resource

##F_RD = "resource_download_n.csv"  # rid,resource,download_n
##F_RC = "resource_connect_n.csv"   # resource_S,resource_D,rid_S,rid_D,connect_n
F_HR = "host_resource_list.csv"   # host,rid,resource
F_RLIST = "resource_list.csv"      # rid,resource

def build_rid_tbl(): 
    ### build resource ID table
    resource = None
    with open(F_RLIST, mode='r') as infile:
        reader = csv.reader(infile, delimiter=',')
        next(reader)
        resource = {int(rows[0]):rows[1] for rows in reader} 
    return resource    

def build_hid_tbl(): 
    ### build host ID table from D/L list -- need to remove duplicate IP
    host = Counter()
    with open(F_HR, mode='r') as infile:
        reader = csv.reader(infile, delimiter=',')
        next(reader)
        for rows in reader:
            host[rows[0]] += 1
    
    ### build host table
    i = 0
    host_tbl = {}   ## use normal dict here, 'cause counter cannot start from 0
    for key in host.keys():
        host_tbl.update({key:i})
        i += 1

    return host_tbl
  
def build_hr_mat(host): 
    """
    Return (hid, rid, # d/l)
    """
    ### build host ID table from D/L list -- need to remove duplicate IP
    host_res = Counter()
    with open(F_HR, mode='r') as infile:
        reader = csv.reader(infile, delimiter=',')
        next(reader)
        for rows in reader:
            host_res[(host[rows[0]], int(rows[1]))] +=1    ## hid, rid, # d/l

    return (host_res) 

  
if __name__ == '__main__':
        
    #resource = build_rid_tbl()
    host     = build_hid_tbl()
    hr_mat = build_hr_mat(host) 
    
    ### written in sparse matrix
    I = [i[0] for i in hr_mat.keys()]
    J = [i[1] for i in hr_mat.keys()]
    D = [i    for i in hr_mat.values()]
    
    with open('hr_mat.dat', 'w') as f:
        for i in range(len(I)):
            f.write("%d,%d,%d\n" % (I[i],J[i],D[i] ) )
        
        
