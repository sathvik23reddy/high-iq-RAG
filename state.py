shared_state = {
    "index": None,  
    "updated": False,  
}

def setIndex(index):
    shared_state['index'] = index

def setFlag(val):
    shared_state['updated'] = val

def getIndex():
    return shared_state['index']

def getFlag():
    return shared_state['updated']