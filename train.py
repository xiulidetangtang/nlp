import io
import torch
from torch import optim
import torch.nn as nn
from models.model import BaseModel, ModelSmall
from torch.utils.data import DataLoader, TensorDataset
from data.load_data import createDataLoader, createEmbeddingDataset
from utils.utils import *


def mcrmse(predict, actual):
    """
    MCRMSE loss:
    """

    return torch.mean(torch.sqrt(torch.mean(
        (predict - actual)**2, dim = 1
    )))

def valid_loss(model, valid_loader, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for _, (x,y) in enumerate(valid_loader):
            x = x.to(device)
            y = y.to(device)
            predict = model(x)
            total_loss += mcrmse(torch.exp(predict), y).item()
        model.train()
    return total_loss / len(valid_loader)

def valid_loss_basemodel(model, valid_loader, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for _, (x,mask, y) in enumerate(valid_loader):
            x = x.to(device)
            y = y.to(device)
            mask = mask.to(device)
            predict = model(x, mask)
            total_loss  += mcrmse(predict, y).item()
    model.train()
    return total_loss / len(valid_loader)

def train_basemodel(n_epochs, model_path = '/root/autodl-tmp/cache/mymodels/',seed = 1234):
    torch.random.manual_seed(seed)
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        print("Use device: cuda:0")
    else:
        device = torch.device('cpu')
        print("Use device: cpu")

    print("Preparing data...")
    train_loader, valid_loader = createDataLoader(device, batch_size=12, valid_size=64)
    model = BaseModel()

    #Use multiple GPU if possible
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    optimizer = optim.Adam(model.parameters(), lr = 1e-4)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones = [2, 3, 4], gamma = 0.5)
    #criterion = nn.MSELoss() or MSE loss
    model.train()
    model = model.to(device)

    saveto(model, model_path + 'raw.pickle')
    torch.save(model.state_dict(), model_path + 'model.pt')

    loss_list = []
    loss_list_epoch = []
    loss_list_valid = []

    valid_loss_min = +torch.inf
    for epoch in range(n_epochs):
        total_loss = 0
        for count, (x,mask,  y) in enumerate(train_loader):
            x = x.to(device)
            mask = mask.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            predict = model(x, mask)
            loss = mcrmse(predict, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            loss_list.append(loss.item())

            if (count + 1) % 10 == 0:
                print(f"Epoch:{epoch + 1}/{n_epochs}, iteration:{count + 1}, loss = {loss.item()}")
            #check if it reached a best valid loss every 20 iters
            if (count + 1) % 20 == 0:
                loss_v = valid_loss_basemodel(model, valid_loader, device)
                loss_list_valid.append(loss_v)
                print(f"Validation... loss = {loss_v}")
                if loss_v < valid_loss_min:
                    print(f"New best valid loss:{valid_loss_min} --> {loss_v}")
                    print(f"save the best model to path {model_path}")
                    saveto(model, model_path + 'raw.pickle')
                    torch.save(model.state_dict(), model_path + 'model.pt')
                    valid_loss_min = loss_v
        
        scheduler.step()
        #valid loss
        loss_v = valid_loss_basemodel(model, valid_loader, device)
        print(f"Epoch:{epoch + 1}/{n_epochs}, loss = {total_loss / len(train_loader)}, valid loss = {loss_v}")
        loss_list_epoch.append(total_loss / len(train_loader))

        if loss_v < valid_loss_min:
            print(f"New best valid loss:{valid_loss_min} --> {loss_v}")
            print(f"save the best model to path {model_path}")
            saveto(model, model_path + 'raw.pickle')
            torch.save(model.state_dict(), model_path + 'model.pt')
            valid_loss_min = loss_v
        else:
            print(f"Not the best, best valid loss is {valid_loss_min}")
        saveto(loss_list, model_path + 'loss_list.pickle')
        saveto(loss_list_epoch, model_path + 'loss_list_epoch.pickle')
        saveto(loss_list_valid, model_path + 'loss_list_valid.pickle')
    
    print(f"Training done! Best valid loss:{valid_loss_min}, saving loss info...")
    saveto(loss_list, model_path + 'loss_list6.pickle')
    saveto(loss_list_epoch, model_path + 'loss_list_epoch.pickle')
    saveto(loss_list_valid, model_path + 'loss_list_valid.pickle')
    return model

#train for small model
def train(n_epochs, seed = 1):
    torch.random.manual_seed(seed)
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        print("Use device: cuda:0")
    else:
        device = torch.device('cpu')
        print("Use device: cpu")
    print("Preparing data...")
    train_set = loadfrom('./data/trainset.pickle')
    valid_set = loadfrom('./data/validset.pickle')
    train_loader = DataLoader(train_set, batch_size=16, shuffle=True)
    valid_loader = DataLoader(valid_set, batch_size=16, shuffle=True)
    model = ModelSmall()

    optimizer = optim.Adam(model.parameters(), lr = 1e-4)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones = [3,  10, 20], gamma = 0.5)
    model.train()
    model = model.to(device)

    valid_loss_min = +torch.inf
    for epoch in range(n_epochs):
        total_loss = 0
        for count, (x, y) in enumerate(train_loader):
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            predict = model(x)
            loss = mcrmse(predict, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if (count + 1) % 10 == 0:
                print(f"Epoch:{epoch + 1}/{n_epochs}, iteration:{count + 1}, loss = {loss.item()}")
        
        scheduler.step()
        #valid loss
        loss_v = valid_loss(model, valid_loader, device)
        print(f"Epoch:{epoch + 1}/{n_epochs}, loss = {total_loss / len(train_loader)}, valid loss = {loss_v}")

        if loss_v < valid_loss_min:
            print(f"New best valid loss:{valid_loss_min} -> {loss_v}")
            valid_loss_min = loss_v
    
    return model




if __name__ == '__main__':
    model = train_basemodel(15)
