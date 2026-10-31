import bchlib


BCH_POLYNOMIAL = 137
BCH_BITS = 5


def create_bch():
    return bchlib.BCH(BCH_BITS, BCH_POLYNOMIAL)


def decode_inplace(bch, data, ecc):
    bitflips = bch.decode(data=data, recv_ecc=ecc)
    if bitflips >= 0:
        bch.correct(data=data, ecc=ecc)
    return bitflips
