import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleCritic(nn.Module):

    def __init__(self, num_layer, dimS, nA, hidden1, hidden2, hidden3):
        super(DoubleCritic, self).__init__()
        self.num_layer = num_layer
        self.fc1 = nn.Linear(dimS, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)

        self.fc5 = nn.Linear(dimS, hidden1)
        self.fc6 = nn.Linear(hidden1, hidden2)

        if self.num_layer==2:
            self.fc3 = nn.Linear(hidden2, nA)
            self.fc7 = nn.Linear(hidden2, nA)
        else:
            self.fc3 = nn.Linear(hidden2, hidden3)
            self.fc4 = nn.Linear(hidden3, nA)

            self.fc7 = nn.Linear(hidden2, hidden3)
            self.fc8 = nn.Linear(hidden3, nA)

    def forward(self, state):
        x1 = F.relu(self.fc1(state))
        x1 = F.relu(self.fc2(x1))

        x2 = F.relu(self.fc5(state))
        x2 = F.relu(self.fc6(x2))

        if self.num_layer==2:
            x1 = self.fc3(x1)
            x2 = self.fc7(x2)
        else:
            x1 = F.relu(self.fc3(x1))
            x1 = self.fc4(x1)

            x2 = F.relu(self.fc7(x2))
            x2 = self.fc8(x2)

        return x1, x2

    def Q1(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        if self.num_layer == 2:
            x = self.fc3(x)
        else:
            x = F.relu(self.fc3(x))
            x = self.fc4(x)
        return x




class DoubleCritics(nn.Module):
    def __init__(self, num_layer, dimS, nA, hidden1, hidden2, hidden3):
        super(DoubleCritics, self).__init__()
        self.fc1 = nn.Linear(dimS, hidden1)
        self.fc5 = nn.Linear(dimS, hidden1)
        self.num_layer= num_layer

        if self.num_layer==2:
            self.fc2 = nn.Linear(hidden1, hidden2)
            self.fc2_adv = nn.Linear(hidden1, hidden2)
            self.fc3 = nn.Linear(hidden2, 1)
            self.fc3_adv = nn.Linear(hidden2, nA)

            self.fc6 = nn.Linear(hidden1, hidden2)
            self.fc6_adv = nn.Linear(hidden1, hidden2)
            self.fc7 = nn.Linear(hidden2, 1)
            self.fc7_adv = nn.Linear(hidden2, nA)
        else:
            self.fc2 = nn.Linear(hidden1, hidden2)
            self.fc6 = nn.Linear(hidden1, hidden2)

            self.fc3 = nn.Linear(hidden2, hidden3)
            self.fc4 = nn.Linear(hidden3, 1)
            self.fc3_adv = nn.Linear(hidden2, hidden3)
            self.fc4_adv = nn.Linear(hidden3, nA)

            self.fc7 = nn.Linear(hidden2, hidden3)
            self.fc8 = nn.Linear(hidden3, 1)
            self.fc7_adv = nn.Linear(hidden2, hidden3)
            self.fc8_adv = nn.Linear(hidden3, nA)

    def forward(self, state):
        x1 = F.relu(self.fc1(state))
        x2 = F.relu(self.fc5(state))


        if self.num_layer==2:
            x1_adv = F.relu(self.fc2_adv(x1))
            x1 = F.relu(self.fc2(x1))
            x1_adv = self.fc3_adv(x1_adv)
            x1 = self.fc3(x1)

            x2_adv = F.relu(self.fc6_adv(x2))
            x2 = F.relu(self.fc6(x2))
            x2_adv = self.fc7_adv(x2_adv)
            x2 = self.fc7(x2)

        else:
            x1 = F.relu(self.fc2(x1))
            x2 = F.relu(self.fc6(x2))

            x1_adv = F.relu(self.fc3_adv(x1)) #advantage
            x1 = F.relu(self.fc3(x1)) # value

            x1_adv = self.fc4_adv(x1_adv) #advantage
            x1 = self.fc4(x1) # value

            x2_adv = F.relu(self.fc7_adv(x2))
            x2 = F.relu(self.fc7(x2))

            x2_adv = self.fc8_adv(x2_adv)
            x2 = self.fc8(x2)

        return x1 + x1_adv - x1_adv.mean(), x2 + x2_adv - x2_adv.mean()



class Critic_QR(nn.Module):
    """
    implementation of a critic network Q(s, a)
    """
    def __init__(self, state_dim, num_action, hidden_size1, hidden_size2, hidden_size3, num_layer, dueling, atoms):
        super(Critic, self).__init__()
        self.atoms = atoms
        #self.dropout = nn.Dropout(1.0)
        if num_layer == 2:
            self.fc1 = nn.Linear(state_dim, hidden_size1)
            self.fc2 = nn.Linear(hidden_size1, hidden_size2)
            self.fc3 = nn.Linear(hidden_size2, num_action)


            if dueling == 1:
                self.fc3 = nn.Linear(hidden_size2, self.atoms)
                self.fc2_adv = nn.Linear(hidden_size1, hidden_size2)
                self.fc3_adv = nn.Linear(hidden_size2, num_action * self.atoms)

        if num_layer == 3:
            self.fc1 = nn.Linear(state_dim, hidden_size1)
            self.fc2 = nn.Linear(hidden_size1, hidden_size2)
            self.fc3 = nn.Linear(hidden_size2, hidden_size3)
            self.fc4 = nn.Linear(hidden_size3, num_action)



            if dueling == 1:
                self.fc4 = nn.Linear(hidden_size3, self.atoms)
                self.fc3_adv = nn.Linear(hidden_size2, hidden_size3)
                self.fc4_adv = nn.Linear(hidden_size3, num_action * self.atoms)



        self.dueling = dueling
        self.num_action = num_action
        self.num_layer = num_layer

    def forward(self, state):
        # given a state s, the network returns a vector Q(s,) of length |A|
        # network with two hidden layers
        if self.num_layer == 2:
            x1 = F.relu(self.fc1(state))
            x1_ = F.relu(self.fc1(state))

            x2 = F.relu(self.fc2(x1))
            x2_ = F.relu(self.fc2(x1))
            if self.dueling == 1:
                x2_adv = F.relu(self.fc2_adv(x1))
                x3_adv = self.fc3_adv(x2_adv)
                x3 = self.fc3(x2)

                v, a = x3.view(-1, 1, self.atoms), x3_adv.view(-1, self.num_action, self.atoms)
                q = v + a - a.mean(1, keepdim=True)  # Combine streams
                # q = F.softmax(q, dim=2)  # Probabilities with action over second dimension
                return q
            else:
                return self.fc3(x2)



