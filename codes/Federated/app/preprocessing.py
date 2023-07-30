import numpy as np
import pandas as pd
from pathlib import Path
import random
import math
from matplotlib import pyplot
from pathlib import Path

def shuffle_data_and_build_dict_for_indexes_of_labels(y_data, seed, amount):
    y_data=pd.DataFrame(y_data,columns=["label"])
    y_data["index"]=np.arange(len(y_data))
    label_dict = {}
    for i in range(10):
        var_name = f"label_{i}"
        label_info=y_data[y_data["label"]==i]
        np.random.seed(seed)
        label_info=np.random.permutation(label_info)
        label_info=label_info[0:amount]
        label_info=pd.DataFrame(label_info, columns=["label","index"])
        label_dict.update({var_name: label_info })
    return label_dict 
# labeller ve her labelden amount qeder indexlerin goturub, labeller shuffle edirik ve amount qeder label secirik

def creating_client_data_sets_for_each_client_with_shuffled_data(label_dict, number_of_clients, amount): 
    clients_data_dicts= dict()
    label_amount =int(math.floor(amount/number_of_clients))
    for i in range(number_of_clients):
        client_data_name="client_data_"+str(i)
        dumb=pd.DataFrame()
        for j in range(10):
            label_name=str("label_")+str(j)
            a=label_dict[label_name][i*label_amount:(i+1)*label_amount]
            dumb=pd.concat([dumb,a], axis=0)
        dumb.reset_index(drop=True, inplace=True)    
        clients_data_dicts.update({client_data_name: dumb}) 
    return clients_data_dicts 

# number of client_datas qeder client_data build edirik ver he client_datada her labelden var label_amount qeder


def creating_final_datasets_with_combining_client_datas(clients_data_dicts, x_data, y_data, x_name, y_name, number_of_clients):
    x_data_dict= dict()
    y_data_dict= dict()
    for i in range(number_of_clients): 
        xname= x_name + '_' + str(i)
        yname= y_name + '_' + str(i)
        client_data_name = "client_data_" + str(i)
        indexes=np.sort(np.array(clients_data_dicts[client_data_name]["index"]))
        x_info= x_data[indexes]
        x_data_dict.update({xname : x_info})
        y_info= y_data[indexes]
        y_data_dict.update({yname : y_info})
    return x_data_dict, y_data_dict 

# client_data-dictda labeller var ama x-data saxlamadiq, burda x-data ile hemin labelleri biryere yigiriq
 
def data_shuffler_and_distributor_for_client(x_data, y_data, x_name, y_name, amount, number_of_clients):
    label_dict=shuffle_data_and_build_dict_for_indexes_of_labels(y_data, 1, amount)
    clients_data_dicts = creating_client_data_sets_for_each_client_with_shuffled_data(label_dict, number_of_clients, amount)
    x_dict, y_dict = creating_final_datasets_with_combining_client_datas(clients_data_dicts, x_data, y_data, x_name, y_name, number_of_clients)
    return x_dict, y_dict

