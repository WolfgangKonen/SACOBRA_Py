# just some tests for Enum.
# Not directly related to SACOBRA.

from enum import Enum

class RSTYPE2(Enum):
    SIGMOID = 1
    CONSTANT = 2

class ISAoptions:
    def __init__(self,
                 isa_ver=1,
                 RStype2=RSTYPE2.CONSTANT,  # .CONSTANT,  .SIGMOID
                 ):
        self.isa_ver = isa_ver
        self.RStype2 = RStype2

def enum_demo():
    io = ISAoptions()
    if io.RStype2 == RSTYPE2.CONSTANT:
        print(io.RStype2.name)

    io = ISAoptions(RStype2=RSTYPE2.SIGMOID)
    if io.RStype2 == RSTYPE2.CONSTANT:
        print(io.RStype2.name)
    if io.RStype2 == RSTYPE2.SIGMOID:
        print(f"SIGM = {io.RStype2.name}")

    for x in list(RSTYPE2): print(x.value, x.name)

enum_demo()