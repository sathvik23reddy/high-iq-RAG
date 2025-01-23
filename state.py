shared_state = {
    "index": None,  
    "updated": False,  
    "file_map": set()
}

def setIndex(index):
    shared_state['index'] = index

def setFlag(val):
    shared_state['updated'] = val

def getIndex():
    return shared_state['index']

def getFlag():
    return shared_state['updated']

def insertFileMap(file):
    shared_state['file_map'].add(file)

def isInFileMap(file):
    return file in shared_state['file_map']

def isFileMapEmpty():
    return True if len(shared_state['file_map'])==0 else False