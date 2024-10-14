import torch
import torch.nn as nn
from torch.utils.data import Dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class RNN(nn.Module):
    #  TODO:  Implémenter comme décrit dans la question 1
    def __init__(self, dim, latent, output):
        super(RNN, self).__init__()
        # self.dim = dim
        # self.latent = latent
        # self.output = output
        self.Wi = nn.Linear(dim, latent)
        self.Wh = nn.Linear(latent, latent)
        self.Wd = nn.Linear(latent, output)

        self.decoder = nn.Sigmoid()
        self.activation = nn.Tanh()
    
    def one_step(self, x, h):
        """
        x: batch*dim
        h: batch*latent
        h_next: batch*latent 
        """
        h_next = self.activation(self.Wi(x) + self.Wh(h))
        return h_next
    
    def forward(self, x, h):
        """
        x: length*batch*dim
        h: battch*latent
        h_seq: length*batch*latent
        """
        length = x.size(0)
        h_seq = []

        for i in range(length):
            h = self.one_step(x[i],h)
            h_seq.append(h)
        
        return torch.stack(h_seq, dim=0)
    
    def decode(self, h):
        """
        h: batch*latent
        y: batch*output
        """
        y = self.decoder(self.Wd(h))
        return y


class SampleMetroDataset(Dataset):
    def __init__(self, data,length=20,stations_max=None):
        """
            * data : tenseur des données au format  Nb_days x Nb_slots x Nb_Stations x {In,Out}
            * length : longueur des séquences d'exemple
            * stations_max : normalisation à appliquer
        """
        self.data, self.length= data, length
        ## Si pas de normalisation passée en entrée, calcul du max du flux entrant/sortant
        self.stations_max = stations_max if stations_max is not None else torch.max(self.data.view(-1,self.data.size(2),self.data.size(3)),0)[0]
        ## Normalisation des données
        self.data = self.data / self.stations_max
        self.nb_days, self.nb_timeslots, self.classes = self.data.size(0), self.data.size(1), self.data.size(2)

    def __len__(self):
        ## longueur en fonction de la longueur considérée des séquences
        return self.classes*self.nb_days*(self.nb_timeslots - self.length)

    def __getitem__(self,i):
        ## transformation de l'index 1d vers une indexation 3d
        ## renvoie une séquence de longueur length et l'id de la station.
        station = i // ((self.nb_timeslots-self.length) * self.nb_days)
        i = i % ((self.nb_timeslots-self.length) * self.nb_days)
        timeslot = i // self.nb_days
        day = i % self.nb_days
        return self.data[day,timeslot:(timeslot+self.length),station],station

class ForecastMetroDataset(Dataset):
    def __init__(self, data,length=20,stations_max=None):
        """
            * data : tenseur des données au format  Nb_days x Nb_slots x Nb_Stations x {In,Out}
            * length : longueur des séquences d'exemple
            * stations_max : normalisation à appliquer
        """
        self.data, self.length= data,length
        ## Si pas de normalisation passée en entrée, calcul du max du flux entrant/sortant
        self.stations_max = stations_max if stations_max is not None else torch.max(self.data.view(-1,self.data.size(2),self.data.size(3)),0)[0]
        ## Normalisation des données
        self.data = self.data / self.stations_max
        self.nb_days, self.nb_timeslots, self.classes = self.data.size(0), self.data.size(1), self.data.size(2)

    def __len__(self):
        ## longueur en fonction de la longueur considérée des séquences
        return self.nb_days*(self.nb_timeslots - self.length - 1)

    def __getitem__(self,i):
        ## Transformation de l'indexation 1d vers indexation 2d
        ## renvoie x[d,t:t+length-1,:,:], x[d,t+1:t+length,:,:]
        # 通过整除和取余操作，分别计算出当前样本所属的时间片和天数
        timeslot = i // self.nb_days
        day = i % self.nb_days
        return self.data[day,timeslot:(timeslot+self.length)],self.data[day,(timeslot+2):(timeslot+self.length+2)]

