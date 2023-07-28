import numpy as np
import pandas as pd
import torch.nn.functional as F
from torch import nn
from torch import optim
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import torch
from modelling import net2nn

def create_model_optimizer_criterion_dict(number_of_clients, learning_rate, momentum):
    model_dict = dict()
    optimizer_dict= dict()
    criterion_dict = dict()
    
    for i in range(number_of_clients):
        model_name=f"client_{i}_model"
        model_info=net2nn()
        model_dict.update({model_name : model_info })
        
        optimizer_name="optimizer"+str(i)
        optimizer_info = torch.optim.SGD(model_info.parameters(), lr=learning_rate, momentum=momentum)
        optimizer_dict.update({optimizer_name : optimizer_info })
        
        criterion_name = "criterion"+str(i)
        criterion_info = nn.CrossEntropyLoss()
        criterion_dict.update({criterion_name : criterion_info})
        
    return model_dict, optimizer_dict, criterion_dict 

def get_averaged_weights(model_dict, number_of_clients, name_of_models):
   
    fc1_mean_weight = torch.zeros(size=model_dict[name_of_models[0]].fc1.weight.shape)
    fc1_mean_bias = torch.zeros(size=model_dict[name_of_models[0]].fc1.bias.shape)
    
    fc2_mean_weight = torch.zeros(size=model_dict[name_of_models[0]].fc2.weight.shape)
    fc2_mean_bias = torch.zeros(size=model_dict[name_of_models[0]].fc2.bias.shape)
    
    fc3_mean_weight = torch.zeros(size=model_dict[name_of_models[0]].fc3.weight.shape)
    fc3_mean_bias = torch.zeros(size=model_dict[name_of_models[0]].fc3.bias.shape)
    
    with torch.no_grad():
        for i in range(number_of_clients):
            fc1_mean_weight += model_dict[name_of_models[i]].fc1.weight.data.clone()
            fc1_mean_bias += model_dict[name_of_models[i]].fc1.bias.data.clone()
        
            fc2_mean_weight += model_dict[name_of_models[i]].fc2.weight.data.clone()
            fc2_mean_bias += model_dict[name_of_models[i]].fc2.bias.data.clone()
        
            fc3_mean_weight += model_dict[name_of_models[i]].fc3.weight.data.clone()
            fc3_mean_bias += model_dict[name_of_models[i]].fc3.bias.data.clone()

        
        fc1_mean_weight =fc1_mean_weight/number_of_clients
        fc1_mean_bias = fc1_mean_bias/ number_of_clients
        
        fc2_mean_weight =fc2_mean_weight/number_of_clients
        fc2_mean_bias = fc2_mean_bias/ number_of_clients
    
        fc3_mean_weight =fc3_mean_weight/number_of_clients
        fc3_mean_bias = fc3_mean_bias/ number_of_clients
    
    return fc1_mean_weight, fc1_mean_bias, fc2_mean_weight, fc2_mean_bias, fc3_mean_weight, fc3_mean_bias


def set_averaged_weights_as_main_model_weights_and_update_main_model(main_model, model_dict, number_of_clients, name_of_models):
    fc1_mean_weight, fc1_mean_bias, fc2_mean_weight, fc2_mean_bias, fc3_mean_weight, fc3_mean_bias = get_averaged_weights(model_dict, number_of_clients, name_of_models)
    with torch.no_grad():
        main_model.fc1.weight.data = fc1_mean_weight.data.clone()
        main_model.fc2.weight.data = fc2_mean_weight.data.clone()
        main_model.fc3.weight.data = fc3_mean_weight.data.clone()

        main_model.fc1.bias.data = fc1_mean_bias.data.clone()
        main_model.fc2.bias.data = fc2_mean_bias.data.clone()
        main_model.fc3.bias.data = fc3_mean_bias.data.clone() 
    return main_model


def send_main_model_to_nodes_and_update_model_dict(main_model, model_dict, number_of_clients,name_of_models):
    with torch.no_grad():
        for i in range(number_of_clients):
            model_dict[name_of_models[i]].fc1.weight.data =main_model.fc1.weight.data.clone()
            model_dict[name_of_models[i]].fc2.weight.data =main_model.fc2.weight.data.clone()
            model_dict[name_of_models[i]].fc3.weight.data =main_model.fc3.weight.data.clone() 
            model_dict[name_of_models[i]].fc1.bias.data =main_model.fc1.bias.data.clone()
            model_dict[name_of_models[i]].fc2.bias.data =main_model.fc2.bias.data.clone()
            model_dict[name_of_models[i]].fc3.bias.data =main_model.fc3.bias.data.clone() 
    return model_dict


def start_train_end_node_process(number_of_clients, model_dict, name_of_criterions, name_of_optimizers, criterion_dict, name_of_models, optimizer_dict, x_train_dict, y_train_dict, x_test_dict, y_test_dict, batch_size, numEpoch):
    for i in range (number_of_clients): 
        train_ds = TensorDataset(x_train_dict[list(x_train_dict.keys())[i]], y_train_dict[list(y_train_dict.keys())[i]])
        train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        
        test_ds = TensorDataset(x_test_dict[list(x_test_dict.keys())[i]], y_test_dict[list(y_test_dict.keys())[i]])
        test_dl = DataLoader(test_ds, batch_size= batch_size * 2)
    
        model=model_dict[name_of_models[i]]
        criterion=criterion_dict[name_of_criterions[i]]
        optimizer=optimizer_dict[name_of_optimizers[i]]
    
        # print("Client: " ,i)
        for epoch in range(numEpoch):        
            train_loss, train_accuracy = model.train_model(train_dl, criterion, optimizer)
            test_loss, test_accuracy =   model.validate_model(test_dl, criterion)
    
            # print("epoch: {:3.0f}".format(epoch+1) + " | train accuracy: {:7.5f}".format(train_accuracy) + " | test accuracy: {:7.5f}".format(test_accuracy))


def compare_local_and_merged_model_performance(number_of_clients, model_dict, name_of_criterions, name_of_optimizers, criterion_dict, name_of_models, optimizer_dict, main_model, main_criterion,  x_test_dict, y_test_dict, batch_size):
    accuracy_table=pd.DataFrame(data=np.zeros((number_of_clients,3)), columns=["Client", "local_ind_model", "merged_main_model"])
    for i in range (number_of_clients):
    
        test_ds = TensorDataset(x_test_dict[list(x_test_dict.keys())[i]], y_test_dict[list(y_test_dict.keys())[i]])
        test_dl = DataLoader(test_ds, batch_size=batch_size * 2)
    
        model=model_dict[name_of_models[i]]
        criterion=criterion_dict[name_of_criterions[i]]
        optimizer=optimizer_dict[name_of_optimizers[i]]
    
        individual_loss, individual_accuracy = model.validate_model(test_dl, criterion)
        main_loss, main_accuracy =model.validate_model( test_dl, main_criterion )
    
        accuracy_table.loc[i, "Client"]="Client "+str(i)
        accuracy_table.loc[i, "local_ind_model"] = individual_accuracy
        accuracy_table.loc[i, "merged_main_model"] = main_accuracy

    return accuracy_table