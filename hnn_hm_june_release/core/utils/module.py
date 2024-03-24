class Node(object):
    def __init__(self, degree, x, y):
        self.degree = degree
        self.x = x
        self.y = y

    def __str__(self):
        return "degree = {}, x = {}, y = {}".format(self.degree, self.x, self.y)


class VesselSegment(object):
    def __init__(self, node1, node2, vessel_centerline):
        self.node1 = node1
        self.node2 = node2
        self.vessel_centerline = vessel_centerline
        self.vessel_mask = None
        self.vessel_class = None