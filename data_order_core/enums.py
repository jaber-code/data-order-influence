from enum import Enum

class Incremental_Type(Enum):
    ALPH = "alph"
    RAND = "random"
    MOST_DISS_1ST = "most_diss_first"
    MOST_SIM_1ST = "most_sim_first"
    HYBRID_DISS_SIM = "hybrid_diss_sim"
    HYBRID_DISS_SEQ = "hybrid_diss_seq"


class Incremental_Decay_Type(Enum):
    STATIC = "static"
    DEC = "dec"
    INC = "inc"