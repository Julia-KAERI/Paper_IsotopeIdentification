import numpy as np
import struct, os, sys
import matplotlib.pyplot as plt
import csv
import pandas as pd
import scipy.optimize as optimization
from datetime import datetime
from os import listdir
from os.path import isfile, join

dir0 = "/Users/jiyong/data/spectrum_analyzer_202507/v03/reports/"
dir1 = "/Users/jiyong/data/spectrum_analyzer_202507/v03/results/"


def bccs(data):
    """array 를 Backward cumulative channel sum (BCCS) 형태로 변환하여 반환합니다.
    """
    return np.array([sum(data[i:]) for i in range(0, len(data))])


class Specs:
    def __init__(self):
        df = get_dataframe()
        self.df = df[df.time1 > datetime(2024, 10, 16, 00, 00, 00)]
        self.df.reset_index(drop=True, inplace=True)
        self.df.insert(7, "Am241", np.nan)
        self.df.insert(8, "Co60", np.nan)
        self.df.insert(9, "Cs137", np.nan)
        self.ref = get_ref_data()
        self.current_ref = self.ref['kwangyang_20241016']
        self.initparam = (0.01, 0.01, 0.01)
        
        
    def __len__(self):
        return len(self.df)
    
    def get_spec(self, id:int):
        return pd.read_csv(self.df.iloc[id].specs, sep = ","+" ", skiprows=9, engine='python')

    def get_ref(self):
        return self.ref
    
    def save2xlsx(self, fn:str):
        self.df.to_excel(fn)
        
        
    def specfit(self, id:int, detector = 1, i0=3, params = (0.5, 0.5, 0.5), plot=False):
        """Fit the spectrum to a linear combination of reference spectra.
        """ 
        spec = self.get_spec(id)
        
        if detector == 1 :
            spectrum = spec["Main"] - spec["MainBkg"]
            
        elif detector == 2:
            spectrum = spec["Sub"] - spec["SubBkg"]
        else :
            raise IndexError("Invalide detector index : %d"%(detector))
        
        pnd = "PN%1d"%(detector)
        
        amspec = self.current_ref[pnd]["Am241"] - self.current_ref[pnd]["BKG"]
        cospec = self.current_ref[pnd]["Co60"] - self.current_ref[pnd]["BKG"]
        csspec = self.current_ref[pnd]["Cs137"] - self.current_ref[pnd]["BKG"]
        
        def detfunc(params, xdata, ydata):
            a, b, c = params[0:3]
            return ydata[i0:] - (abs(a)*amspec[i0:]+abs(b)*cospec[i0:]+abs(c)*csspec[i0:])

        fr1, _ = optimization.leastsq(detfunc, params, args=(np.arange(len(spectrum)), spectrum))
                
        if plot:
            plt.figure(figsize=(12, 6))
            plt.plot(spectrum, label="Spectrum", color="black")
            plt.plot(fr1[0]*amspec, label="Am-241", color="blue")
            plt.plot(fr1[1]*cospec, label="Co-60", color="red")
            plt.plot(fr1[2]*csspec, label="Cs-137", color="green")
            plt.plot(fr1[0]*amspec + fr1[1]*cospec + fr1[2]*csspec, label="Fitted", color="violet")

            plt.yscale("log", base=10)
            plt.ylim(1.0, spectrum[0]*2)
            plt.legend()
            plt.title("Spectrum Fit")
            plt.show()
        return fr1
    
    def bccsfit(self, id:int, detector= 1, i0=3, i1 = None, params=(0.5, 0.5, 0.5, 0.1), plot=False, ylog=True):
        """Fit the spectrum to a linear combination of reference spectra.
        """ 
        
        spec = self.get_spec(id)
        pnd = "PN%1d"%(detector)
        Mchan = len(self.current_ref[pnd]["Am241"])
        if i1 == None or i1 > Mchan or i1 <= i0 :
            i1 = len(self.current_ref[pnd]["Am241"])
        
        
        
        if detector == 1 :
            spectrum = bccs(spec["Main"][i0:i1] - spec["MainBkg"][i0:i1])
            
        elif detector == 2:
            spectrum = bccs(spec["Sub"][i0:i1] - spec["SubBkg"][i0:i1])
        else :
            raise IndexError("Invalide detector index : %d"%(detector))
        
        
        
        amspec0 = self.current_ref[pnd]["Am241"][i0:i1] - self.current_ref[pnd]["BKG"][i0:i1]
        cospec0 = self.current_ref[pnd]["Co60"][i0:i1] - self.current_ref[pnd]["BKG"][i0:i1]
        csspec0 = self.current_ref[pnd]["Cs137"][i0:i1]- self.current_ref[pnd]["BKG"][i0:i1]
        
        nchan = np.arange(i0, i1)
        
        amspec = bccs(amspec0)
        cospec = bccs(cospec0)
        csspec = bccs(csspec0)
                
        def detfunc(params, xdata, ydata):
            a, b, c, d = params[0:4]
            return np.abs(ydata - (abs(a)*amspec+abs(b)*cospec+abs(c)*csspec+ d))

        fr1, _ = optimization.leastsq(detfunc, params, args=(np.arange(len(spectrum)), spectrum))
        
        if plot:
            plt.figure(figsize=(12, 6))
            plt.plot(nchan, spectrum, label="Spectrum", color="black")
            plt.plot(nchan, fr1[0]*amspec, label="Am-241", color="blue")
            plt.plot(nchan, fr1[1]*cospec, label="Co-60", color="red")
            plt.plot(nchan, fr1[2]*csspec, label="Cs-137", color="green")
            plt.plot(nchan, fr1[0]*amspec + fr1[1]*cospec + fr1[2]*csspec, label="Fitted", color="violet")
            if ylog :
                plt.yscale("log", base=10)
            plt.ylim(1.0, spectrum[0]*2)
            plt.legend()
            plt.title("Spectrum Fit")
            plt.show()
        return fr1 
        
    def bccsfit2(self, id:int, bgid:int, detector= 1, i0=3, i1 = None, params=(0.5, 0.5, 0.5, 0.1), plot=False, ylog=True):
        """Fit the spectrum to a linear combination of reference spectra.
        """ 
        
        spec = self.get_spec(id)
        bgspec = self.get_spec(bgid)
        pnd = "PN%1d"%(detector)
        Mchan = len(self.current_ref[pnd]["Am241"])
        if i1 == None or i1 > Mchan or i1 <= i0 :
            i1 = len(self.current_ref[pnd]["Am241"])
        
        
        
        if detector == 1 :
            spectrum0 = bccs(spec["Main"][i0:i1] - spec["MainBkg"][i0:i1])
            bgspectrum =bccs(bgspec["Main"][i0:i1] - bgspec["MainBkg"][i0:i1])
        elif detector == 2:
            spectrum0 = bccs(spec["Sub"][i0:i1] - spec["SubBkg"][i0:i1])
            bgspectrum =bccs(bgspec["Main"][i0:i1] - bgspec["MainBkg"][i0:i1])
        else :
            raise IndexError("Invalide detector index : %d"%(detector))
        
        spectrum = spectrum0 - bgspectrum
        
        amspec0 = self.current_ref[pnd]["Am241"][i0:i1] - self.current_ref[pnd]["BKG"][i0:i1]
        cospec0 = self.current_ref[pnd]["Co60"][i0:i1] - self.current_ref[pnd]["BKG"][i0:i1]
        csspec0 = self.current_ref[pnd]["Cs137"][i0:i1]- self.current_ref[pnd]["BKG"][i0:i1]
        
        nchan = np.arange(i0, i1)
        
        amspec = bccs(amspec0)
        cospec = bccs(cospec0)
        csspec = bccs(csspec0)
                
        def detfunc(params, xdata, ydata):
            a, b, c, d = params[0:4]
            return ydata - (abs(a)*amspec+abs(b)*cospec+abs(c)*csspec+ d)

        fr1, _ = optimization.leastsq(detfunc, params, args=(np.arange(len(spectrum)), spectrum))
        
        if plot:
            plt.figure(figsize=(12, 6))
            plt.plot(nchan, spectrum, label="Spectrum", color="black")
            plt.plot(nchan, fr1[0]*amspec, label="Am-241", color="blue")
            plt.plot(nchan, fr1[1]*cospec, label="Co-60", color="red")
            plt.plot(nchan, fr1[2]*csspec, label="Cs-137", color="green")
            plt.plot(nchan, fr1[0]*amspec + fr1[1]*cospec + fr1[2]*csspec, label="Fitted", color="violet")
            if ylog :
                plt.yscale("log", base=10)
            plt.ylim(1.0, spectrum[0]*2)
            plt.legend()
            plt.title("Spectrum Fit")
            plt.show()
        return fr1 
        

def get_dataframe():

    onlyfiles = [f for f in listdir(dir0) if isfile(join(dir0, f)) and f.endswith('carentry.bin')]
    
    time1, time2, nus, gammaLevel, neutronLevel, nuclides, fns1, fns2 =[], [], [], [], [], [], [], []
    for of in onlyfiles :
        if os.path.getsize(join(dir0, of)) == 0:
            continue
        f1 = open(join(dir0, of), 'rb')
        header = (struct.unpack("BBBBH", f1.read(6)))
        if header[0:4] == (0xf5, 0xfa, 0x12, 0x01) :
            readmore = True
        else :
            readmore = False
        while readmore:
            t1 = struct.unpack("HBBHHHH", f1.read(12))
            t2 = struct.unpack("HBBHHHH", f1.read(12))
            mm = header[-1]-24
            nu = ''.join([chr(x) for x in struct.unpack("B"*(mm-1), f1.read(mm-1)) if x !=0])
            alarmlevel = struct.unpack("B", f1.read(1))
            # gammaLevel = alarmlevel[0] %16
            # neutronLevel = alarmlevel[0] // 16
            
            q1 = "%4d%02d%02d_%02d%02d%02d"%(t1[0], t1[1], t1[2], t1[3], t1[4], t1[5])
            #nu.append(str(alarmlevel))
            if isfile(os.path.join(dir1, q1+"_spec.txt")) and os.path.getsize(os.path.join(dir1, q1+"_cps.txt")) :

                time1.append(datetime(*(t1[0:6])))
                time2.append(datetime(*(t2[0:6])))
                nus.append(nu)
                fns1.append(os.path.join(dir1, q1+"_spec.txt"))
                fns2.append(os.path.join(dir1, q1+"_cps.txt"))
                gammaLevel.append(alarmlevel[0] % 16)
                neutronLevel.append(alarmlevel[0] // 16)
            else:
                # print(os.path.join(dir1, q1+"_spec.txt"), " not found")
                pass
            
            q = f1.read(6)
            if not q:
                readmore = False
            else:
                try:
                    header = struct.unpack("BBBBH", q)
                except:
                    readmore = False
                
                #print("Header = ", header)
                if header[0:4] == (0xf5, 0xfa, 0x12, 0x01) :
                    readmore = True
                else :
                    readmore = False
    print("time1 time : ", type(time1))
    dfs = pd.DataFrame({'time1': time1, 'time2': time2, 'glevel':gammaLevel, 'nlevel': neutronLevel,  'nuclides': nus, 'specs': fns1, 'cps': fns2})
    dfs.sort_values(by='time1', inplace=True, ignore_index=True)
    return dfs

def get_ref_data():
    """ isotope reference data 를 읽고 이를 dictionary 형태로 반환합니다.
    """
    dir0 = "/Users/jiyong/data/spectrum_analyzer_202507/v03/isotope_data/"


    isotope_dir = {"daejeon": dir0 + "Isotope_sample",
                "kwangyang_20241016" : dir0 + "kwangyang_20241016",
                "kwangyang_20241017" : dir0 + "kwangyang_20241017",
                "kwangyang_20241111" : dir0 + "kwangyang_20241111",}

    isotope_data = {}
    for key, val in isotope_dir.items():
        isotope_data[key] = {"PN1":{}, "PN2": {}}
        for f in os.listdir(val)  :
            if f.startswith("PN1_") and f.count("Am241") > 0:
                isotope_data[key]["PN1"]["Am241"] = read_mca(os.path.join(val, f))
            elif f.startswith("PN2_") and f.count("Am241") > 0:
                isotope_data[key]["PN2"]["Am241"] = read_mca(os.path.join(val, f))
            elif f.startswith("PN1_") and f.count("Co60") > 0:
                isotope_data[key]["PN1"]["Co60"] = read_mca(os.path.join(val, f))
            elif f.startswith("PN2_") and f.count("Co60") > 0:
                isotope_data[key]["PN2"]["Co60"] = read_mca(os.path.join(val, f))
            elif f.startswith("PN1_") and f.count("Cs137") > 0:
                isotope_data[key]["PN1"]["Cs137"] = read_mca(os.path.join(val, f))
            elif f.startswith("PN2_") and f.count("Cs137") > 0:
                isotope_data[key]["PN2"]["Cs137"] = read_mca(os.path.join(val, f))
            elif f.startswith("PN1_") and f.count("BKG") > 0:
                isotope_data[key]["PN1"]["BKG"] = read_mca(os.path.join(val, f))
            elif f.startswith("PN2_") and f.count("BKG") > 0:
                isotope_data[key]["PN2"]["BKG"] = read_mca(os.path.join(val, f))
    return isotope_data

    
def read_mca(fp):
    """ mca 파일을 읽어 numpy array로 반환합니다.
    """
    if os.path.isfile(fp) and os.path.getsize(fp) > 10 :
        return np.genfromtxt(fp, skip_header=12, skip_footer=68, encoding='latin-1')
    else:
        return None
    




def specfit(params, spectrum, refspecs, i0=3, plot=False):
    """Fit the spectrum to a linear combination of reference spectra.
    """ 
    amspec = refspecs["Am241"] - refspecs["Bkg"]
    cospec = refspecs["Co60"] - refspecs["Bkg"]
    csspec = refspecs["Cs137"] - refspecs["Bkg"]
    def detfunc(params, xdata, ydata):
        a, b, c = params[0:3]
        return ydata[i0:] - (abs(a)*amspec[i0:]+abs(b)*cospec[i0:]+abs(c)*csspec[i0:])

    fr1, _ = optimization.leastsq(detfunc, (0.5, 0.5, 0.5), args=(np.arange(len(spectrum)), spectrum))
    if plot:
        plt.figure(figsize=(12, 6))
        plt.plot(spectrum, label="Spectrum", color="black")
        plt.plot(fr1[0]*amspec, label="Am-241", color="blue")
        plt.plot(fr1[1]*cospec, label="Co-60", color="red")
        plt.plot(fr1[2]*csspec, label="Cs-137", color="green")
        plt.plot(fr1[0]*amspec + fr1[1]*cospec + fr1[2]*csspec, label="Fitted", color="violet")

        plt.yscale("log", base=10)
        plt.ylim(1.0, spectrum[0]*2)
        plt.legend()
        plt.title("Spectrum Fit")
        plt.show()
    return fr1

def bccsfit(params, spectrum, refspecs, i0=3, plot=False):
    """Fit the spectrum to a linear combination of reference spectra.
    """ 
    amspec = bccs(refspecs["Am241"] - refspecs["Bkg"])
    cospec = bccs(refspecs["Co60"] - refspecs["Bkg"])
    csspec = bccs(refspecs["Cs137"] - refspecs["Bkg"])
    def detfunc(params, xdata, ydata):
        a, b, c = params[0:3]
        return ydata[i0:] - (abs(a)*amspec[i0:]+abs(b)*cospec[i0:]+abs(c)*csspec[i0:])

    fr1, _ = optimization.leastsq(detfunc, (0.5, 0.5, 0.5), args=(np.arange(len(spectrum)), spectrum))
    if plot:
        plt.figure(figsize=(12, 6))
        plt.plot(spectrum, label="Spectrum", color="black")
        plt.plot(fr1[0]*amspec, label="Am-241", color="blue")
        plt.plot(fr1[1]*cospec, label="Co-60", color="red")
        plt.plot(fr1[2]*csspec, label="Cs-137", color="green")
        plt.plot(fr1[0]*amspec + fr1[1]*cospec + fr1[2]*csspec, label="Fitted", color="violet")
        plt.yscale("log", base=10)
        plt.ylim(1.0, spectrum[0]*2)
        plt.legend()
        plt.title("Spectrum Fit")
        plt.show()
    return fr1