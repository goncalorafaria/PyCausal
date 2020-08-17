
class Named:

    def __init__(self,
                 name):
        self.name = name

    def __str__(self):
        return self.name

    def uname(self):
        return self.name.split("/")[1]
