class FaceInfo:
    def __init__(self) -> None:
        self.timestamp = 0
        self.result = set
        # self.result.possibility = 0.0
        # self.result.name = ''
        # self.result.x1 = 0
        # self.result.y1 = 0
        # self.result.x2 = 0
        # self.result.y2 = 0

    def __repr__(self) -> str:
        info = 'timestamp         : [' + str(self.timestamp)

        for item in self.result:
            info += ']\ntimestamp : [' + str(item.possibility) \
                  + ']\nname      : [' + str(item.name)        \
                  + ']\nx1        : [' + str(item.x1)          \
                  + ']\ny1        : [' + str(item.y1)          \
                  + ']\nx2        : [' + str(item.x2)          \
                  + ']\ny2        : [' + str(item.y2)          \
                  + ']'

        return info
