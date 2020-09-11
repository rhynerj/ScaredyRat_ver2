import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display, HTML
import math
import time

def animal_read(inpath, filename, sheet):
    '''
    Returns corresponding animals for sheets
    Reads data into dataframe with long header removed.
    Uses correct labels as column labels.
    Uses time as row indexes.
    '''
    filepath = os.path.join(inpath,filename)
    try:
        df = pd.read_excel(filepath,sheet_name=sheet,skipfooter=35,index_col=0)
    except:
        return(-1,-1,-1)

    anim_loc = df.index.get_loc('Animal ID')
    animal = df.iloc[anim_loc][0]

    ctx_loc = df.index.get_loc('Trial Control settings')
    context = df.iloc[ctx_loc][0]

    skiprows = df.index.get_loc('Trial time') + 1
    print("{} {} is {} in {}".format(filename, sheet, animal, context))

    df = pd.read_excel(filepath,sheet_name=sheet,skiprows=skiprows,index_col=0,headers=0)
    df = df[1:]
    df.replace(to_replace='-',value=0,inplace=True)
    return(animal,context,df)
    
def find_tone_vels(df,i):
    '''
    '''
    tone = pd.DataFrame()
    num = str(i+1)
    
    try:
        label = 'Tone ' + num
        tone = pd.DataFrame(df[df[label] == 1])
    except:
        label = 'Tone' + num
        tone = pd.DataFrame(df[df[label] == 1])
    
    tone = tone['Velocity']
    return(tone)

def find_shock_vels(df, i):
    '''
    '''
    sresponse = pd.DataFrame()
    num = str(i+1)
    try:
        label = 'Tone ' + num
        tone = df[df[label] == 1] #Tone dataframe
    except:
        label = 'Tone' + num
        tone = df[df[label] == 1] #Tone dataframe
    toneend = tone.iloc[-1] #Time for tone end

    starttime = math.floor(toneend['Recording time']) #Time for sresponse start
    endtime = math.floor(toneend['Recording time'] + 5) #Time for sresponse end
    itime = df.index.get_loc(starttime,method='bfill') #Index for sresponse start
    etime = df.index.get_loc(endtime,method='bfill') #Index for sresponse end

    sresponse = df.iloc[itime:etime] #df for sresponse
    sresponse = sresponse['Velocity']

    return(sresponse)
    
def find_delim_segment(df, ndelim, delim):
    '''
    Creates dictionary of each delimited segment. Values are
    dataframes for times at which each delimiter == 1.
    Takes in the dataframe (df), the number of delimiters (ndelim)
    and the base delimiter string (delim).  It is assumed
    that each delimiter is either labeled as delim + ' ' + num
    or delim + num.
    
    Output is a dataframe (datarray) which is made up of
    delimited periods
    '''
    datarray = {}
    i = 0
    while i < ndelim: # number of delimiters
        num = str(i+1)
        try:
            label = delim + num
            dat = pd.DataFrame(df[df[label] == 1])
        except:
            label = delim.strip() + num
            dat = pd.DataFrame(df[df[label] == 1])
        datarray[i] = dat
        i += 1
    return(datarray)
    
def find_timeperiod(df, nperiods, starttime, endtime):
    '''
    Creates a dictionary of the given timeperiods
    of the recording in order to calculate
    behaviors.
    '''
    datarray = {}
    label = 'Recording time'
    i = 0
    while i < nperiods:
        stime = starttime[i]
        stime_idx= df.index.get_loc(stime,method='bfill')
        
        etime = endtime[i]
        etime_idx = df.index.get_loc(etime,method='bfill')
        
        dat = df.iloc[stime_idx:etime_idx]
        datarray[i] = dat
        i += 1
    return(datarray)

def find_delim_based_time(df, ndelim, delim, startidx, startoffset, stopoffset):
    '''
    Creates dictionary of each pretone. Values
    are dataframes for 30s before tone == 1
    '''
    datarray = {}
    i = 0
    while i< ndelim:
        num = str(i+1)
        try:
            label = delim.strip() + num
            predat = pd.DataFrame(df[df[label] == 1])
        except:
            label = delim + num
            predat = pd.DataFrame(df[df[label] == 1])
            
        delimidx = predat.iloc[startidx] #Time for tone start

        starttime = math.floor(delimidx['Recording time']+startoffset) #Time for pretone start
        endtime = math.floor(delimidx['Recording time']+stopoffset) #Time for pretone end
        itime = df.index.get_loc(starttime,method='bfill') #Index for pretone start
        etime = df.index.get_loc(endtime,method='bfill') #Index for pretone end

        dat = df.iloc[itime:etime] #df for pretone
        datarray[i] = dat #dictionary for all pretones
        i += 1
    return(datarray)
    
def get_baseline_freezing(datadict, freezingThreshold, binSecs):
    freezing = pd.DataFrame()
    freezingSecs = 0
    nonfreezingSecs = 0
    freezingTimes = []


    toneLabel = 'Baseline Freezing'

    epoch = datadict
    vels = epoch['Velocity']

    startSec = int(round(vels.index[0],0))
    endSec = int(round(vels.index[-1],0))
    # print('\nNumber of indices - ', vels.index.size,'\nLost frames leading - ',(vels.index[0] - startSec),'\nLost frames after - ',(vels.index[-1] - endSec))
    counter = 0
    for n in range(startSec, endSec-1, binSecs):
        startOfSecond = vels.index.get_loc(n, method='bfill')
        endOfSecond = vels.index.get_loc(n+(binSecs-0.1), method='bfill')
        velsInSec = []
        #counter += endOfSecond-startOfSecond
        for frame in range(startOfSecond, endOfSecond):
            velocity = float(vels.iloc[frame])
            velsInSec.append(velocity)
            counter += 1
        if np.mean(velsInSec) < freezingThreshold:
            freezingSecs += 1
            freezingTimes.append([n,n+binSecs])
        else:
            nonfreezingSecs += 1
    # print('\nCounter - ', counter)
    percentFreezing = 100.0 * round(freezingSecs/(freezingSecs + nonfreezingSecs),3)
    toneFreezing = pd.DataFrame({toneLabel: [freezingSecs, nonfreezingSecs, percentFreezing]},index=['Freezing', 'Nonfreezing','Percent Freezing']).T
    freezing = pd.concat([freezing, toneFreezing])
    freezingSecs = 0
    nonfreezingSecs = 0
    return(freezing, freezingTimes)

def get_freezing(datadict, ntones, freezingThreshold, binSecs):
    freezing = pd.DataFrame()
    freezingSecs = 0
    nonfreezingSecs = 0
    freezingTimes = []

    i = 0
    while i < ntones:
        toneLabel = 'Tone {}'.format(str(i))

        epoch = datadict[i]
        vels = epoch['Velocity']

        startSec = int(round(vels.index[0],0))
        endSec = int(round(vels.index[-1],0))
        # print('\nNumber of indices - ', vels.index.size,'\nLost frames leading - ',(vels.index[0] - startSec),'\nLost frames after - ',(vels.index[-1] - endSec))
        counter = 0
        for n in range(startSec, endSec-1, binSecs):
            startOfSecond = vels.index.get_loc(n, method='bfill')
            endOfSecond = vels.index.get_loc(n+(binSecs-0.1), method='bfill')
            velsInSec = []
            #counter += endOfSecond-startOfSecond
            for frame in range(startOfSecond, endOfSecond):
                velocity = float(vels.iloc[frame])
                velsInSec.append(velocity)
                counter += 1
            if np.mean(velsInSec) < freezingThreshold:
                freezingSecs += 1
                freezingTimes.append([n,n+binSecs])
            else:
                nonfreezingSecs += 1
        # print('\nCounter - ', counter)
        percentFreezing = 100.0 * round(freezingSecs/(freezingSecs + nonfreezingSecs),3)
        toneFreezing = pd.DataFrame({toneLabel: [freezingSecs, nonfreezingSecs, percentFreezing]},index=['Freezing', 'Nonfreezing','Percent Freezing']).T
        freezing = pd.concat([freezing, toneFreezing])
        freezingSecs = 0
        nonfreezingSecs = 0
        i += 1
    return(freezing, freezingTimes)

def get_darting(datadict, ntones, dartThreshold, binSecs):
    darting = pd.DataFrame()
    dartingTimes = []
    nDarts = 0

    i = 0
    while i < ntones:
        toneLabel = 'Tone {}'.format(str(i))

        epoch = datadict[i]
        vels = epoch['Velocity']

        startSec = int(round(vels.index[0],0))
        endSec = int(round(vels.index[-1],0))

        for n in range(startSec, endSec-1, binSecs):
            startOfSecond = vels.index.get_loc(n, method='bfill')
            endOfSecond = vels.index.get_loc(n+(binSecs-0.1), method='bfill')
            velsInSec = []
            for frame in range(startOfSecond, endOfSecond):
                velocity = float(vels.iloc[frame])
                velsInSec.append(velocity)
            for v in velsInSec:
                if v > dartThreshold:
                    nDarts += 1
                    dartingTimes.append([n,n+binSecs])
                    break

        toneDarting = pd.DataFrame({toneLabel: nDarts},index=['Darts']).T
        darting = pd.concat([darting, toneDarting])
        nDarts = 0
        i += 1
    return(darting, dartingTimes)

def plot_outputs(anim, ID, trialTypeFull, outpath, trialType, ntones, FTs, DTs):
    ## plot stuff
    vels = pd.DataFrame(anim['Velocity'])

    plt.style.use('seaborn-white')
    plt.figure(figsize=(16,8),facecolor='white',edgecolor='white')
    plt.axhline(linewidth=2, color='black')
    plt.axvline(linewidth=2, color='black')
    parameters = {'axes.labelsize': 22,
            'axes.titlesize': 35,
            'xtick.labelsize': 18,
            'ytick.labelsize':18,
            'legend.fontsize':18}
    plt.rcParams.update(parameters)
    # Plots main velocity in black
    line1, = plt.plot(vels,color='k',linewidth=0.1,label='ITI')

    # Loops through tones, plots each one in cyan
    i = 0
    hasTone = False
    while i < ntones:            
        tone = find_tone_vels(anim,i)
        line2, = plt.plot(tone,color='c',linewidth=0.5,label='Tone')            
        i += 1
        hasTone = True

    # Loops through shocks, plots each one in magenta
    i = 0
    hasShock = False
    while i < ntones:
        sresponse = find_shock_vels(anim,i)
        line3, = plt.plot(sresponse,color='m',linewidth=0.5,label='Shock')
        i += 1
        hasShock = True

    # Loops through freezing bins, plots each below the x-axis
    for timebin in FTs:
        plt.plot([timebin[0],timebin[1]],[-0.3,-0.3],color='#ff4f38',linewidth=3)

    # Loops through darting bins, plots each below the x-axis
    # print(DTs)
    for timebin in DTs:
        plt.plot([timebin[0],timebin[1]],[-0.7,-0.7],color='#167512',linewidth=3)
        
    plt.ylim(-1,35)

    sns.despine(left=True, bottom=True, right=True)
    plt.title(ID + " " + trialTypeFull)
    if(hasTone & hasShock):
        plt.legend(handles=[line1,line2,line3])
    elif(hasTone):
        plt.legend(handles=[line1,line2])
    else:
        plt.legend(handles=[line1])
    
    plt.ylabel('Velocity (cm/s)')
    plt.xlabel('Trial time (s)')

    ## define where to save the fig
    fname = outpath + '/' + trialType + '-plot-{}'
    fname = fname.format(ID)

    plt.savefig(fname, dpi=300)
    plt.savefig(fname+'eps', format='eps',dpi=300)

    # plt.show()
    plt.close()
    return()
    
def get_means(datadict,timebin, ntones):
    '''
    Returns a dataframe of mean velocity of timebin at each tone.
    '''
    meanlist = []
    i = 0
    while i < ntones:
        epoch = datadict[i]
        vels = epoch['Velocity']
        mean = round(np.mean(vels),3)
        meanlist.append(mean)
        i += 1
    means = pd.DataFrame(meanlist,columns=[timebin + ' Mean Velocity'])
    means.index = np.arange(1, len(meanlist) + 1)
    return(means)

def get_meds(datadict,timebin, ntones):
    '''
    Returns a dataframe of median velocity of timebin at each tone.
    '''
    medlist = []
    i = 0
    while i < ntones:
        epoch = datadict[i]
        vels = epoch['Velocity']
        med = round(np.median(vels),3)
        medlist.append(med)
        i += 1
    meds = pd.DataFrame(medlist,columns=[timebin + ' Median Velocity'])
    meds.index = np.arange(1, len(meds) + 1)
    return(meds)

def get_SEMs(datadict,timebin, ntones):
    '''
    Returns a dataframe of median velocity of timebin at each tone.
    '''
    SEMlist = []
    i = 0
    while i < ntones:
        epoch = datadict[i]
        vels = epoch['Velocity']
        SEM = round(np.std(vels),3)
        SEMlist.append(SEM)
        i += 1
    SEMs = pd.DataFrame(SEMlist,columns=[timebin + 'SEM'])
    SEMs.index = np.arange(1, len(SEMs) + 1)
    return(SEMs)

def get_vels(df, ntones):
    '''
    Creates data of all velocities from original dataframe for plotting.
    '''
    tonevels = {}
    i = 0
    while i < ntones: # number of tones
        vels = []
        num = str(i)
        try:
            label = 'Tone ' + num
            tone = pd.DataFrame(df[df[label] == 1])
        except:
            label = 'Tone' + num
            tone = pd.DataFrame(df[df[label] == 1])
        vels.append(tone['Velocity'])
        tonevels[i] = vels
        i += 1
    return(tonevels)

def get_top_vels(datadict,nmax, ntones):
    '''
    Returns dataframe of nmax (int) maximum velocities for a timebin.
    The second section adds a column for an average of the maxima.
    '''
    nmaxes = pd.DataFrame()
    i = 0
    while i < ntones:
        epoch = datadict[i]
        vels = epoch['Velocity']
        vlist = vels.tolist()
        vlist.sort()
        if(nmax==1):
            topvels = pd.DataFrame([vlist[-1]])
        else:
            topvels = pd.DataFrame([vlist[-nmax:-1]])
        nmaxes = nmaxes.append(topvels)
        i += 1

    nmaxes.index = np.arange(1, nmaxes.shape[0] + 1)
    nmaxes.columns = np.arange(1, nmaxes.shape[1] + 1)

    nmaxes['Mean'] = nmaxes.mean(axis = 1)

    return(nmaxes)
def scaredy_find_csvs(csv_dir, prefix):
    csvlist = []

    for root, dirs, names in os.walk(csv_dir):
        for file in names:
            if file.startswith(prefix):
                f = os.path.join(root, file)
                csvlist.append(f)

    return(csvlist)

def concat_data(means, SEMs, meds, ntones):
    allData = pd.DataFrame()
    ix = []
    for n in range(ntones):
        allData = allData.append(means.iloc[:,n])
        ix.append('Tone {} Mean'.format(n+1))
    for n in range(ntones): 
        allData = allData.append(SEMs.iloc[:,n])
        ix.append('Tone {} SEM'.format(n+1))
    for n in range(ntones):     
        allData = allData.append(meds.iloc[:,n])       
        ix.append('Tone {} Median'.format(n+1))

    allData.index = ix
    allData = allData.transpose()

    return(allData)

def concat_all_freezing(csvlist, tbin):
    freezing = pd.DataFrame()
    for csv in csvlist:
        anim = get_anim(csv,-2)
        df = pd.read_csv(csv, index_col=0).T
        loc = (tbin * 3) + 2
        percentF = pd.DataFrame([df.iloc[loc]], index=[anim])
        freezing = pd.concat([freezing, percentF])

    return(freezing)

def concat_all_max(csvlist):
    maxes = pd.DataFrame()

    for csv in csvlist:
        anim = get_anim(csv,-2)
        df = pd.read_csv(csv,index_col=0)
        meanMax = pd.DataFrame({anim: df['Mean']}).T
        maxes = pd.concat([maxes, meanMax])

    return(maxes)

def concat_all_darting(csvlist, loc):
    freezing = pd.DataFrame()
    for csv in csvlist:
        anim = get_anim(csv,-2)
        df = pd.read_csv(csv, index_col=0).T
        percentF = pd.DataFrame([df.iloc[loc]], index=[anim])
        freezing = pd.concat([freezing, percentF])

    return(freezing)
    
def compress_data(csvlist,tbin):

    allanims = pd.DataFrame()
    for csv in csvlist:
        anim = get_anim(csv,-2)
        df = pd.read_csv(csv,index_col=0).transpose()

        tonevels = pd.DataFrame(df.iloc[tbin]).transpose()
        tonevels.set_index([[anim]],inplace=True)

        allanims = pd.concat([allanims,tonevels])

    return(allanims)
    
def compile_SR(trialType, num_dEpoch,dEpoch_list, behavior, inpath, outpath):
    summaryCSVs = scaredy_find_csvs(inpath,trialType + '-' + behavior)
    meanCSVs = scaredy_find_csvs(inpath,trialType + '-mean')
    medCSVs = scaredy_find_csvs(inpath,trialType + '-med')
    SEMCSVs = scaredy_find_csvs(inpath,trialType + '-SEM')
    for i in range(0,num_dEpoch):
        summaryData = concat_all_darting(summaryCSVs,i)
        outfile = os.path.join(outpath, projName + '-All-'+ trialType + '-' +dEpoch_list[i] + '-' + behavior + '.csv' )
        summaryData.to_csv(outfile)
        
        means = compress_data(meancsvs,i)
        meds = compress_data(medcsvs,i)
        SEMs = compress_data(SEMcsvs,i)
        allData = concat_data(means,SEMs,meds,nFCtones)
        outfile = os.path.join(outpath,projName + '-All-'+ trialType + '-' +dEpoch_list[i] + '-data.csv')
        allData.to_csv(outfile)
    
    