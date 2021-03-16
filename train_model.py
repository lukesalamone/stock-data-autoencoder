from model import StonksNet
import torch
from torch import Tensor, optim
import torch.nn as nn
import numpy as np
import pandas as pd
import os

STOCK_PATH = 'stock_data'

def train_stonksnet(prices:Tensor):
    num_days, num_components = tuple(prices.size())
    model = StonksNet(size=num_components)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    criterion = nn.MSELoss()

    for epoch in range(20):
        # train day by day
        losses = []
        for day in prices:
            optimizer.zero_grad()
            output = model(day)
            loss = criterion(output, day)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        print(f'epoch {epoch} avg loss: {np.mean(losses)}')

    return model

def get_all_components():
    files = os.listdir(STOCK_PATH)
    files = [x[:-4] for x in files if x[-4:] == '.csv']

    components = []
    for name in files:
        df = pd.read_csv(os.path.join(STOCK_PATH, f'{name}.csv'))
        df = df.to_numpy()[:,0]

        components.append(df)

    return components, files

if __name__ == '__main__':
    components, symbols = get_all_components()

    components = [c.astype(np.double) for c in components]
    components = torch.transpose(torch.tensor(components, dtype=torch.float), dim0=0, dim1=1)

    train_stonksnet(components)