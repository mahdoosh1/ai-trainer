import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn import MSELoss
import os


class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super().__init__()
        self.sizes = len(hidden_sizes) + 1
        self.linears = nn.ModuleList()
        if self.sizes > 3:
            for i in range(-1, self.sizes):
                if i == -1:
                    self.linears.append(
                        module=nn.Linear(input_size, hidden_sizes[i+1])
                    )
                elif i == self.sizes - 1:
                    self.linears.append(
                        module=nn.Linear(hidden_sizes[i-1], output_size)
                    )
                else:
                    self.linears.append(
                        module=nn.Linear(hidden_sizes[i], hidden_sizes[i+1])
                    )
        else:
            self.linears.append(module=nn.Linear(input_size, hidden_sizes[0]))
            self.linears.append(module=nn.Linear(hidden_sizes[0], output_size))

    def forward(self, x):
        for layer in self.linears:
            x = F.relu(layer(x))
        return x

    def save(self, file_name='model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)


class QTrainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def train_step(self, inputs):
        """
        trains predefined model

        Args:
          inputs : a list where elements are
              inputs, outputs, reward, next_inputs (not used if short_train is true), short_train respectively

        Returns:
            None
        """
        temp = [[],[],[],[],[]]
        for i in inputs:
            for idx,j in enumerate(i):
                temp[idx].append(j)
        for i in range(5):
            temp[i] = torch.tensor(temp[i], dtype=torch.float)
        # (n, x)
        if len(temp[0].shape) == 1:
            temp2 = []
            for i in temp:
                if i != inputs[4]:
                    temp2.append(torch.unsqueeze(i, 0))
                else:
                    break
            temp2.append(temp[4])
            temp = temp2
        # 1: predicted Q values with current input
        # print(temp)
        pred = self.model(temp[0])  # input -> output

        target = pred.clone()  # output initialise
        for idx in range(len(temp[4])):  # iterate done
            weight = temp[2][idx]
            if temp[4][idx]:
                weight += torch.tensor(0, dtype=torch.float)
            else:
                weight += self.gamma * torch.max(self.model(temp[3][idx]))
            # AI at each "state" will have a "reward" chance to take "action"
            target[idx][torch.argmax(temp[1][idx]).item()] = weight

        # 2: Q_new = reward + gamma * max(next_predicted Q value) -> only do this if not done
        # pred.clone()
        # preds[argmax(action)] = Q_new
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()

        self.optimizer.step()
        return loss
    def train(self, train_inputs, epochs, wanted_loss=0.001, tinputs_updating=[]):
        for i in range(epochs):
            inp = train_inputs if tinputs_updating == [] else tinputs_updating[i]
            loss = self.train_step(inp)
            print(f"Epoch: {i+1} - Loss: {loss}")
            if loss < wanted_loss:
                print(f"Reached wanted loss {wanted_loss}, breaking")
                break

def data_to_train(inputs, outputs, reward, done):
    tin = []
    for i in range(len(inputs) - 1):
        tin.append([(inputs[i]),
                    (outputs[i],),
                    (reward,),
                    (inputs[i + 1],),
                    (done,)
                    ])
    return tin
