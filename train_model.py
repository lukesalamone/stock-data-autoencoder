import matplotlib.pyplot as plt
import torch
from torch import Tensor, optim
from itertools import count
import torch.nn as nn
import numpy as np
import pandas as pd
import os

from model import StonksNet

STOCK_PATH = 'stock_data'

EPSILON = 1

def calc_loss(valid:Tensor, model:StonksNet, criterion:nn.MSELoss):
    losses = []
    for v in valid:
        loss = criterion(v, model(v))
        losses.append(loss.item())
    return np.mean(losses)


def train_stonksnet(train:Tensor, valid:Tensor):
    def stop(losses, epoch):
        if epoch < 50:
            return False

        if len(losses) < 5:
            return
        if max(losses) - min(losses) < EPSILON:
            return True
        return False

    num_days, num_components = tuple(train.size())
    model = StonksNet(size=num_components)
    optimizer = optim.Adam(model.parameters(), lr=0.00001)
    criterion = nn.MSELoss()

    train_losses = []
    valid_losses = []

    for epoch in count():
        # train day by day
        losses = []
        for day in train:
            optimizer.zero_grad()
            output = model(day)
            loss = criterion(output, day)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        train_loss = np.mean(losses)
        train_losses.append(train_loss)
        valid_loss = calc_loss(valid, model, criterion)
        valid_losses.append(valid_loss)
        print(f'epoch {epoch} avg validation loss: {valid_loss}')

        if stop(train_losses[-5:], epoch):
            break

    return model, valid_losses


def train_test_split(prices:Tensor):
    lengths = [int(x*len(prices)) for x in [0.8, 0.1, 0.1]]
    lengths[0] += len(prices) - sum(lengths)
    a,b,c = lengths

    return prices[0:a], prices[a:a+b], prices[a+b:]


def get_all_components():
    files = os.listdir(STOCK_PATH)
    files = [x[:-4] for x in files if x[-4:] == '.csv']

    components = []
    for name in files:
        df = pd.read_csv(os.path.join(STOCK_PATH, f'{name}.csv'))
        df = df.to_numpy()[:,0]

        components.append(df)

    return components, files


def rank_components(model:StonksNet, prices:Tensor, symbols:list):
    inputs = torch.transpose(prices, dim0=0, dim1=1)
    outputs = [model(p).detach().numpy() for p in prices]
    outputs = torch.transpose(torch.tensor(outputs), dim0=0, dim1=1)
    criterion = nn.MSELoss()

    results = {}

    for symbol, x, y in zip(symbols, inputs, outputs):
        loss = criterion(x, y)
        results[symbol] = loss.item()

    results = [x for x in results.items()]
    results = sorted(results, key=lambda x: x[1])

    print('\nbest 5 results: ')
    for (sym, val) in results[0:5]:
        print(f'{sym}: {val:.2f}')

    print('\nworst 5 results: ')
    for (sym, val) in results[-5:]:
        print(f'{sym}: {val:.2f}')


if __name__ == '__main__':
    components, symbols = get_all_components()

    components = [c.astype(np.double) for c in components]
    components = torch.transpose(torch.tensor(components, dtype=torch.float), dim0=0, dim1=1)

    train, valid, test = train_test_split(components)

    model, losses = train_stonksnet(train, valid)
    torch.save(model, 'trained_model.pt')

    x_axis = list(range(len(losses)))
    plt.plot(x_axis, losses, label='losses', color='#08d')
    plt.legend()
    plt.title('validation losses')
    plt.xlabel('epoch', fontsize=12)
    plt.ylabel('mean squared error', fontsize=12)
    plt.savefig('training_losses.png')
    plt.show()

    # you can load a previously trained model here
    # model = torch.load('trained_model.pt')

    rank_components(model, test, symbols)