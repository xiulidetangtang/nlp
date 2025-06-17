import pickle

def as_integer(x : list):
    return [float(i) for i in x]

def saveto(object, path):
    with open(path, 'wb') as f:
        pickle.dump(object, f)

def loadfrom(path):
    with open(path, 'rb') as f:
        return pickle.load(f)