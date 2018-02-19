class FaceInfo:

    def __init__(self) -> None:
        self.timestamp = 0
        self.info["result"].possibility = 0.0
        self.info["result"].name = ''
        self.info["result"].x1   = 0
        self.info["result"].y1   = 0
        self.info["result"].x2   = 0
        self.info["result"].y2   = 0



    def __repr__(self) -> str:
        return 'timestamp        : ['+ str(self.timestamp)              \
             + ']\ntimestamp     : ['+ str(self.info["result"].possibility) \
             + ']\nname          : ['+ str(self.info["result"].name)        \
             + ']\nx1            : ['+ str(self.info["result"].x1)          \
             + ']\ny1            : ['+ str(self.info["result"].y1)          \
             + ']\nx2            : ['+ str(self.info["result"].x2)          \
             + ']\ny2            : ['+ str(self.info["result"].y2)          \
             + ']'

