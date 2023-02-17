import torch.nn as nn
import torch.nn.functional as F


class Critic(nn.Module):
    """
    implementation of a critic network Q(s, a)
    """
    def __init__(self, state_dim, num_action, hidden_size1, hidden_size2, hidden_size3, num_layer, dueling, atoms):
        super(Critic, self).__init__()
        self.atoms = atoms
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
            x2 = F.relu(self.fc2(x1))
            if self.dueling == 1:
                x2_adv = F.relu(self.fc2_adv(x1))
                x3_adv = self.fc3_adv(x2_adv)
                x3 = self.fc3(x2)

                v, a = x3.view(-1, 1, self.atoms), x3_adv.view(-1, self.num_action, self.atoms)
                q = v + a - a.mean(1, keepdim=True)  # Combine streams
                #print(v.mean(dim=2))
                # q = F.softmax(q, dim=2)  # Probabilities with action over second dimension
                return q
            else:
                return self.fc3(x2)

        if self.num_layer == 3:
            x1 = F.relu(self.fc1(state))
            #x1 = self.dropout(x1)
            x2 = F.relu(self.fc2(x1))
            #x2 = self.dropout(x2)

            x3 = F.relu(self.fc3(x2))
            #x3 = self.dropout(x3)
            if self.dueling == 1:
                x3_adv = F.relu(self.fc3_adv(x2))
                #x3_adv = self.dropout(x3_adv)
                x4_adv = self.fc4_adv(x3_adv)
                x4 = self.fc4(x3)

                v, a = x4.view(-1, 1, self.atoms), x4_adv.view(-1, self.num_action, self.atoms)
                q = v + a - a.mean(1, keepdim=True)  # Combine streams
                #q = F.softmax(q, dim=2)  # Probabilities with action over second dimension
                return q
            else:
                return self.fc4(x3)






class Critic_target(nn.Module):
    """
    implementation of a critic network Q(s, a)
    """
    def __init__(self, state_dim, num_action, hidden_size1, hidden_size2, hidden_size3, num_layer, dueling):
        super(Critic_target, self).__init__()
        if num_layer == 2:
            self.fc1 = nn.Linear(state_dim, hidden_size1)
            self.fc2 = nn.Linear(hidden_size1, hidden_size2)
            self.fc3 = nn.Linear(hidden_size2, num_action)

            if dueling == 1:
                self.fc3 = nn.Linear(hidden_size2, 1)
                self.fc2_adv = nn.Linear(hidden_size1, hidden_size2)
                self.fc3_adv = nn.Linear(hidden_size2, num_action)

        if num_layer == 3:
            self.fc1 = nn.Linear(state_dim, hidden_size1)
            self.fc2 = nn.Linear(hidden_size1, hidden_size2)
            self.fc3 = nn.Linear(hidden_size2, hidden_size3)
            self.fc4 = nn.Linear(hidden_size3, num_action)



            if dueling == 1:
                self.fc4 = nn.Linear(hidden_size3, 1)
                self.fc3_adv = nn.Linear(hidden_size2, hidden_size3)
                self.fc4_adv = nn.Linear(hidden_size3, num_action * self.N)



        self.dueling = dueling
        self.num_action = num_action
        self.num_layer = num_layer

    def forward(self, state):
        # given a state s, the network returns a vector Q(s,) of length |A|
        # network with two hidden layers
        if self.num_layer == 2:
            x1 = F.relu(self.fc1(state))
            x2 = F.relu(self.fc2(x1))
            if self.dueling == 1:
                x2_adv = F.relu(self.fc2_adv(x1))
                x3_adv = self.fc3_adv(x2_adv)
                x3 = self.fc3(x2)
                return x3 + x3_adv - x3_adv.mean()
            else:
                return self.fc3(x2)

        if self.num_layer == 3:
            x1 = F.relu(self.fc1(state))
            x2 = F.relu(self.fc2(x1))
            x3 = F.relu(self.fc3(x2))

            if self.dueling == 1:
                x3_adv = F.relu(self.fc3_adv(x2))
                x4_adv = self.fc4_adv(x3_adv)
                x4 = self.fc4(x3)
                return x4 + x4_adv - x4_adv.mean()
            else:
                return self.fc4(x3)


#
# class Critic(nn.Module):
#     """
#     implementation of a critic network Q(s, a)
#     """
#     def __init__(self, state_dim, num_action, hidden_size1, hidden_size2):
#         super(Critic, self).__init__()
#         self.fc1 = nn.Linear(state_dim, 64)
#         self.fc2 = nn.Linear(64, 64)
#         self.fc3 = nn.Linear(64, 64)
#         self.fc4 = nn.Linear(64, num_action)
#
#     def forward(self, state):
#         # given a state s, the network returns a vector Q(s,) of length |A|
#         # network with two hidden layers
#         x = F.relu(self.fc1(state))
#         x = F.relu(self.fc2(x))
#         x = F.relu(self.fc3(x))
#
#         return self.fc4(x)

#
#
# class DoubleCritic(nn.Module):
#     """
#     implementation of a critic network Q(s, a)
#     """
#     def __init__(self, state_dim, num_action, hidden_size1, hidden_size2):
#         super(DoubleCritic, self).__init__()
#         self.fc1 = nn.Linear(state_dim, hidden_size1)
#         self.fc2 = nn.Linear(hidden_size1, hidden_size2)
#         self.fc3 = nn.Linear(hidden_size2, num_action)
#
#         self.fc4 = nn.Linear(state_dim, hidden_size1)
#         self.fc5 = nn.Linear(hidden_size1, hidden_size2)
#         self.fc6 = nn.Linear(hidden_size2, num_action)
#
#     def forward(self, state):
#         # given a state s, the network returns a vector Q(s,) of length |A|
#         # network with two hidden layers
#         x1 = F.relu(self.fc1(state))
#         x1 = F.relu(self.fc2(x1))
#         x1 = self.fc3(x1)
#
#         x2 = F.relu(self.fc1(state))
#         x2 = F.relu(self.fc2(x2))
#         x2 = self.fc3(x2)
#
#         return x1, x2
#
#     def Q1(self, state):
#         x = F.relu(self.fc1(state))
#         x = F.relu(self.fc2(x))
#
#         return self.fc3(x)
