import sys, os
sys.path.insert(1, os.path.realpath(os.path.pardir))
from specs import *
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import t


rawDir = "../../../data/"
pn1Dir = rawDir + "P1_SN1222/"
pn2Dir = rawDir + "P2_SN1215/"
pn1files0 = [f for f in listdir(pn1Dir) if isfile(join(pn1Dir, f)) and f.endswith('mca')]
pn2files0 = [f for f in listdir(pn2Dir) if isfile(join(pn2Dir, f)) and f.endswith('mca')]

pn1files = [None] * len(pn1files0)
pn2files = [None] * len(pn2files0)

for f1 in pn1files0:
    ddd = (f1[1:-4].split("_"))[-1]
    if ddd == "BKG":
        pn1files[0] = pn1Dir+f1
    else :
        q = int(ddd.split("-")[-1])
        pn1files[q] = pn1Dir+f1
for f2 in pn2files0:
    ddd = (f2[1:-4].split("_"))[-1]
    if ddd == "BKG":
        pn2files[0] = pn2Dir+f2
    else :
        q = int(ddd.split("-")[-1])
        pn2files[q] = pn2Dir+f2


Log =[
    [0, ["Bkg"], 0, 0],                            # 0
    [1, ["Am241",], 80, 420],                      # 1
    [2, ["Am241",], 180, 320],                     # 2
    [3, ["Am241",], 280, 220],                     # 3
    [4, ["Am241",], 460, 40],                      # 4
    [5, ["Cs137",], 70, 430],                      # 5
    [6, ["Cs137",], 130, 370],                     # 6
    [7, ["Cs137",], 300, 200],                     # 7
    [8, ["Cs137",], 390, 110],                     # 8
    [9, ["Co60",], 60, 440],                       # 9
    [10, ["Co60",], 120, 380],                      # 10
    [11, ["Co60",], 310, 190],                      # 11
    [12, ["Co60",], 470, 30],                       # 12
    [13, ["Co60", "Cs137"], 136, 359],              # 13
    [14, ["Am241", "Co60"], 287, 208],              # 14
    [15, ["Co60", "Cs137"], 361, 134],              # 15
    [16, ["Am241", "Cs137"], 163, 332],
    [17, ["Am241", "Co60", "Cs137"], 309, 186],
    [18, ["Am241", "Co60",], 210, 285, 153, 342],
    [19, ["Co60", "Cs137"], 285, 210, 367, 128],
    [20, ["Am241", "Cs137"], 85, 410, 173, 322],
    [21, ["Co60", "Cs137"], 316, 179, 223, 272],
    [22, [None,], ],
]

speckeys = [
    "bkg" ,
    "am_80" ,
    "am_180" ,
    "am_280" ,
    "am_460" ,
    "cs_70" ,
    "cs_130" ,
    "cs_300" ,
    "cs_390" ,
    "co_60" ,
    "co_120" ,
    "co_310" ,
    "co_470" ,
    "co_cs_136",
    "am_co_287" ,
    "co_cs_361" ,
    "am_cs_163" ,
    "am_co_cs_309" ,
    "am_210_co_153" ,
    "co_285_cs_367" ,
    "am_85_cs_173" ,
    "co_223_cs_316" ]


def get_header_info():
    """
    Read every *.mca files and arrange experiment informations to dataframe.
    """
    HeaderInfo={}
    for path in pn1files0:
        lines = [l.strip() for l in open(os.path.join(pn1Dir,path), encoding='latin-1').read().splitlines()]
        qs = (path.split("_")[-1]).split(".")[0]
        if qs == "BKG":
            id = 0
        else :
            id = int(qs.split("-")[-1])
        
        def hdr(key):
            for l in lines:
                if l.startswith(key):
                    return l.split('-', 1)[1].strip()
            return ''
        def status(key):
            for l in lines:
                if l.strip().startswith(key):
                    return l.split(':', 1)[1].strip()
            return ''
        def cfg(key):
            for l in lines:
                if l.startswith(key + '='):
                    return l.split('=', 1)[1].split(';')[0]
            return ''
        if id < 22 :
            HeaderInfo[speckeys[id]] ={
                "LIVE_TIME" : hdr('LIVE_TIME'), 
                "REAL_TIME" : hdr('REAL_TIME'), 
                "START_TIME" : hdr('START_TIME'),
                "Fast Count" : status('Fast Count'), 
                "Slow Count" : status('Slow Count'), 
                "Dead Time" : status('Dead Time'),
                "MCAC" : cfg('MCAC'), 
                "GAIN" : cfg('GAIN'), 
                "THFA" : cfg('THFA'), 
                "THSL" : cfg('THSL')
            }
    return HeaderInfo
 
def read_data(initial_index :int = 3, livetime_norm=True):   
    """
    Read spectrum from *.mca files
    """

    ni = initial_index
    Specs1 = pd.DataFrame( {
        "bkg" : read_mca(pn1files[0])[ni:], 
        "am_80" : read_mca(pn1files[1])[ni:],
        "cs_70" : read_mca(pn1files[5])[ni:],
        "cs_130" : read_mca(pn1files[6])[ni:],
        "co_120" : read_mca(pn1files[10])[ni:],
        "co_60" :  read_mca(pn1files[9])[ni:],
        "co_cs_136" : read_mca(pn1files[13])[ni:],
        "am_co_287" : read_mca(pn1files[14])[ni:],
        "co_cs_361" : read_mca(pn1files[15])[ni:],
        "am_cs_163" : read_mca(pn1files[16])[ni:],
        "am_co_cs_309" : read_mca(pn1files[17])[ni:],
        "am_210_co_153" : read_mca(pn1files[18])[ni:],
        "co_285_cs_367" : read_mca(pn1files[19])[ni:],
        "am_85_cs_173" : read_mca(pn1files[20])[ni:],
        "co_316_cs_223" : read_mca(pn1files[21])[ni:]
            
    })

    # PN2 data
    Specs2 = pd.DataFrame( {
        "bkg" : read_mca(pn2files[0])[ni:], 
        "am_80" : read_mca(pn2files[1])[ni:],
        "cs_70" : read_mca(pn2files[5])[ni:],
        "cs_130" : read_mca(pn2files[6])[ni:],
        "co_120" : read_mca(pn2files[10])[ni:],
        "co_60" :  read_mca(pn2files[9])[ni:],
        "co_cs_136" : read_mca(pn2files[13])[ni:],
        "am_co_287" : read_mca(pn2files[14])[ni:],
        "co_cs_361" : read_mca(pn2files[15])[ni:],
        "am_cs_163" : read_mca(pn2files[16])[ni:],
        "am_co_cs_309" : read_mca(pn2files[17])[ni:],
        "am_210_co_153" : read_mca(pn2files[18])[ni:],
        "co_285_cs_367" : read_mca(pn2files[19])[ni:],
        "am_85_cs_173" : read_mca(pn2files[20])[ni:],
        "co_316_cs_223" : read_mca(pn2files[21])[ni:]
            
    })
    
    if livetime_norm :
        HeaderInfo = get_header_info()
        for kk in Specs1.keys():
            Specs1[kk] = Specs1[kk]/float(HeaderInfo[kk]["LIVE_TIME"])
    return Specs1, Specs2

def read_refs(initial_index :int = 3, livetime_norm : bool=True, background_subtration :bool =True): 
    ni = initial_index
    SpecsRef = pd.DataFrame( {
    "bkg" : read_mca(pn1files[0])[ni:], 
    "am_80" : read_mca(pn1files[1])[ni:],
    "am_180" : read_mca(pn1files[2])[ni:],
    "am_280" : read_mca(pn1files[3])[ni:],
    "am_460" : read_mca(pn1files[4])[ni:],
    "co_60" :  read_mca(pn1files[9])[ni:],
    "co_120" :  read_mca(pn1files[10])[ni:],
    "co_310" :  read_mca(pn1files[11])[ni:],
    "co_470" :  read_mca(pn1files[12])[ni:],
    "cs_70" :  read_mca(pn1files[5])[ni:],
    "cs_130" :  read_mca(pn1files[6])[ni:],
    "cs_300" :  read_mca(pn1files[7])[ni:],
    "cs_390" :  read_mca(pn1files[8])[ni:],
    })
    
    if livetime_norm :
        
        HeaderInfo = get_header_info()
        for kk in SpecsRef.keys():
            SpecsRef[kk] = SpecsRef[kk]/float(HeaderInfo[kk]["LIVE_TIME"])

    if background_subtration :
        _rr = {}
        for kk in SpecsRef.keys():
            if kk != 'bkg':
                _rr[kk] = SpecsRef[kk] - SpecsRef['bkg']
        SpecsRefBkg = pd.DataFrame(_rr)
        return SpecsRefBkg
    else :
        return SpecsRef
