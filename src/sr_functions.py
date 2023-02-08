import os, sys
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import math
import time
import PySimpleGUI as sg

# print = sg.Print # Set print to go to a window rather than the terminal

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
    animal = animal.strip()

    ctx_loc = df.index.get_loc('Trial Control settings')
    context = str(df.iloc[ctx_loc][0])
    context=context.strip();

    skiprows = df.index.get_loc('Trial time') + 1
    print("\n{} {} is {} in {}".format(filename, sheet, animal, context))

    df = pd.read_excel(filepath,sheet_name=sheet,skiprows=skiprows,index_col=0,header=0)
    df = df[1:]
    df.replace(to_replace='-',value=0,inplace=True)
    return(animal,context,df)
    
def find_tone_vels(df,i, epochLabel):
    '''
    '''
    tone = pd.DataFrame()
    num = str(i+1)
    # print(epochLabel)
    try:
        label = epochLabel + num
        # print(label)
        tone = pd.DataFrame(df[df[label] == 1])
    except:
        label = epochLabel.strip() + num
        # print(label)
        tone = pd.DataFrame(df[df[label] == 1])
    
    tone = tone['Velocity']
    return(tone)

def find_delim_vels(df,ndelim,epoch,delim_times):  #(df, i):
    '''
    '''
    sresponse = pd.DataFrame()
    num = str(ndelim+1)
    try:
        label = epoch + num
        tone = df[df[label] == 1] #Tone dataframe
    except:
        label = epoch.strip() + num
        tone = df[df[label] == 1] #Tone dataframe
    # print(delim_times)
    # print(' ')
    delimidx = tone.iloc[int(delim_times[0])] #Time for tone start
    #print(delim_times)
    sto = int(delim_times[1])
    eto = int(delim_times[2])
    starttime = math.floor(delimidx['Recording time']+sto) #Time for pretone start
    endtime = math.floor(delimidx['Recording time']+eto) #Time for pretone end
    itime = df.index.get_loc(starttime,method='bfill') #Index for pretone start
    etime = df.index.get_loc(endtime,method='bfill') #Index for pretone end

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
    # print(delim)
    while i < ndelim: # number of delimiters
        num = str(i+1)
        try:
            label = delim + num
            # print(label)
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
# def find_delim_vels(df,ndelim,epoch,delim_times):
#     full_df = find_delim_based_time(df,ndelim,epoch,delim_times[0],delim_times[1],delim_times[2])
#     print(full_df)
#     return(full_df('Velocity'))
    
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
            label = delim + num
            predat = pd.DataFrame(df[df[label] == 1])
        except:
            label = delim.strip() + num
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
    toneFreezing = pd.DataFrame({toneLabel: [freezingSecs, nonfreezingSecs, percentFreezing]},index=['Baseline Freezing (Time Bins)', 'Baseline Nonfreezing (Time Bins)','Baseline Percent Freezing']).T
    freezing = pd.concat([freezing, toneFreezing])
    freezingSecs = 0
    nonfreezingSecs = 0
    return(freezing, freezingTimes)

def get_freezing(datadict, ntones, freezingThreshold, scopeName, binSecs):
    freezing = pd.DataFrame()
    freezingSecs = 0
    nonfreezingSecs = 0
    freezingTimes = []

    i = 0
    while i < ntones:
        toneLabel = scopeName + ' {}'.format(str(i+1))
        #print(toneLabel)
        epoch = datadict[i]
        vels = epoch['Velocity']

        startSec = round(vels.index[0],2)
        endSec = round(vels.index[-1],2)
        print('\nNumber of indices - ', vels.index.size,'\nStart - ',startSec,'\nEnd - ',endSec)
        counter = 0
        for n in np.arange(startSec, endSec, binSecs):
            startOfSecond = vels.index.get_loc(n, method='nearest')
            endOfSecond = vels.index.get_loc(round(n+(binSecs-0.1),2), method='nearest')
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
        print('\n fs: ',freezingSecs,'\t nfs: ', nonfreezingSecs)
        percentFreezing = 100.0 * round(freezingSecs/(freezingSecs + nonfreezingSecs),3)
        toneFreezing = pd.DataFrame({toneLabel: [freezingSecs, nonfreezingSecs, percentFreezing]},index=['Freezing  (Time Bins)', 'Nonfreezing  (Time Bins)','Percent Freezing']).T
        freezing = pd.concat([freezing, toneFreezing])
        freezingSecs = 0
        nonfreezingSecs = 0
        i += 1
    return(freezing, freezingTimes)

def get_darting(datadict, ntones, dartThreshold, scopeName, binSecs):
    darting = pd.DataFrame()
    dartingTimes = []
    nDarts = 0

    i = 0
    while i < ntones:
        toneLabel = scopeName + ' {}'.format(str(i+1))

        epoch = datadict[i]
        vels = epoch['Velocity']

        startSec = (round(vels.index[0],2))
        endSec = (round(vels.index[-1],2))

        for n in np.arange(startSec, endSec, binSecs):
            startOfSecond = vels.index.get_loc(n, method='nearest')
            endOfSecond = vels.index.get_loc(round(n+(binSecs-0.1),2), method='nearest')
            velsInSec = []
            for frame in range(startOfSecond, endOfSecond):
                velocity = float(vels.iloc[frame])
                velsInSec.append(velocity)
            for v in velsInSec:
                if v > dartThreshold:
                    nDarts += 1
                    dartingTimes.append([n,n+binSecs])
                    break

        toneDarting = pd.DataFrame({toneLabel: nDarts},index=['Darts (count)']).T
        darting = pd.concat([darting, toneDarting])
        nDarts = 0
        i += 1
    return(darting, dartingTimes)

def plot_outputs(anim, ID, trialTypeFull, outpath, trialType, ntones, FTs, DTs, epochLabel, printSettings, printLabels):
    colormap = [  [245/256, 121/256, 58/256],
                [169/256,  90/256,  161/256],
                [133/256,  192/256,  249/256],
                [15/256, 32/256, 128/256]  ]
    
                # [ [26/256,  14/256,  52/256],
                # [69/256,  32/256,  76/256],
                # [110/256,  62/256, 103/256], 
                # [133/256,  94/256, 120/256],
                # [141/256, 121/256, 130/256],
                # [146/256, 148/256, 137/256],
                # [151/256, 174/256, 145/256],
                # [167/256, 206/256, 157/256],
                # [213/256, 242/256, 188/256],
                # [254/256, 254/256, 216/256]]
    handle_list=[]
    ## plot stuff
    vels = pd.DataFrame(anim['Velocity'])
    # print('Trying to plot, 1')
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
    handle_list.append(line1)
    # print('Trying to plot, 2')
    # Loops through tones, plots each one in cyan
    i = 0
    hasTone = False
    while i < ntones:            
        tone = find_tone_vels(anim,i, epochLabel)
        line2, = plt.plot(tone,color='c',linewidth=0.5,label=epochLabel.strip())  
        i += 1
        hasTone = True
    if ntones > 0:
        handle_list.append(line2)
    # print('Trying to plot, 3')
    # Loops through shocks, plots each one in magenta
    line3=[]
    hasShock=False
    c_num=0
    for i in range(0,len(printLabels)):
        # print(printSettings[i])
        # print(bool(printSettings[i][3]))
        if(not printSettings[i] or len(printSettings[i]) < 4 or printSettings[i][3] == 'False'):
            continue
        c_num = c_num % 4
        hasShock = True
        for j in range(0,ntones):
            response = find_delim_vels(anim,j,epochLabel,printSettings[i])
            line_tmp, = plt.plot(response,color=colormap[c_num],linewidth=0.5,label=printLabels[i])
            
        if 'line_tmp' in locals():
            handle_list.append(line_tmp)    
        c_num += 1
    # print('Trying to plot, 4')
    # i = 0
    # hasShock = False
    # while i < ntones:
    #     j=0
    #     for j in range(0,len(printSettings)):
    #         if(not printSettings[3]):
    #             continue
    #         c_num =j % 10
    #         sresponse = find_delim_vels(anim,i,printLabels[j],printSettings)
    #         line3, = plt.plot(sresponse,color=colormap[c_num],linewidth=0.5,label=printLabels[j])
    #         i += 1
    #         hasShock = True

    # Loops through freezing bins, plots each below the x-axis
    for timebin in FTs:
        plt.plot([timebin[0],timebin[1]],[-0.3,-0.3],color='#ff4f38',linewidth=3)

    # Loops through darting bins, plots each below the x-axis
    # print(DTs)
    for timebin in DTs:
        plt.plot([timebin[0],timebin[1]],[-0.7,-0.7],color='#167512',linewidth=3)
        
    plt.ylim(-1,35)
    # print('Trying to plot, 5')
    sns.despine(left=True, bottom=True, right=True)
    plt.title(ID + " " + trialTypeFull)

    plt.legend(handles=handle_list, loc='best', fontsize='x-small',markerscale=0.6)
    
    plt.ylabel('Velocity (cm/s)')
    plt.xlabel('Trial time (s)')
    # print('Trying to plot, 6')
    ## define where to save the fig
    fname = outpath + '/' + trialType + '-' + epochLabel +'-plot-{}'
    fname = fname.format(ID)

    plt.savefig(fname, dpi=300)
    plt.savefig(fname+'.eps', format='eps',dpi=300)
    # print('Trying to plot, 7')
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
        num = str(i+1)
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

def get_top_vels(datadict, nmax, binlabel, ntones):
    '''
    Returns dataframe of nmax (int) maximum velocities for a timebin.
    The second section adds a column for an average of the maxima.
    '''
    nmaxes = pd.DataFrame()
    col_list=[]
    i = 0
    while i < ntones:
        epoch = datadict[i]
        vels = epoch['Velocity']
        vlist = vels.tolist()
        vlist.sort()
        if not vlist:
            i +=1
            continue
        elif(nmax==1):
            topvels = pd.DataFrame([vlist[-1]])
        else:
            topvels = pd.DataFrame([vlist[-nmax:-1]])
        nmaxes= nmaxes.append(topvels)
        col_list.append(binlabel + str(i+1) + ' Max Velocity')
        i += 1
    if nmax > 1:
        nmaxes.index = np.arange(1, nmaxes.shape[0] + 1)
        nmaxes.columns = np.arange(1, nmaxes.shape[1] + 1)
    
        nmaxes['Avg Max'] = nmaxes.mean(axis = 1)
    else:
        nmaxes.index = col_list
        nmaxes.columns = ['Max Velocity']
    return(nmaxes)
    

def get_anim(csv):
    exp_rgx = re.compile(r'.*-([a-zA-Z]+-?\d+)\.[a-zA-Z]+$')
    match = exp_rgx.match(csv)

    if match:
        anim = match.group(1)
        return anim
    else:
        raise ValueError('File name does not contain valid animal ID.')
    
def scaredy_find_csvs(csv_dir, prefix):
    csvlist = []

    for root, dirs, names in os.walk(csv_dir):
        for file in names:
            # print(file)
            if file.startswith(prefix):
                f = os.path.join(root, file)
                # print(f)
                csvlist.append(f)

    return(csvlist)

def concat_data(means, SEMs, meds, maxes, ntones):
    allData = pd.DataFrame()
    ix = []
    for n in range(ntones):
        if means.empty: continue
        allData = allData.append(means.iloc[:,n])
        ix.append('Tone {} Mean'.format(n+1))
    for n in range(ntones): 
        if SEMs.empty: continue
        allData = allData.append(SEMs.iloc[:,n])
        ix.append('Tone {} SEM'.format(n+1))
    for n in range(ntones):     
        if meds.empty: continue
        allData = allData.append(meds.iloc[:,n])       
        ix.append('Tone {} Median'.format(n+1))
    for n in range(ntones):     
        if maxes.empty: continue
        allData = allData.append(maxes.iloc[:,n])       
        ix.append('Tone {} Max'.format(n+1))

    allData.index = ix
    allData = allData.transpose()

    return(allData)

def concat_all_freezing(csvlist, tbin):
    freezing = pd.DataFrame()
    for csv in csvlist:
        anim = get_anim(csv)
        df = pd.read_csv(csv, index_col=0).T
        loc = (tbin * 3) + 2
        percentF = pd.DataFrame([df.iloc[loc]], index=[anim])
        freezing = pd.concat([freezing, percentF])

    return(freezing)

def concat_all_max(csvlist):
    maxes = pd.DataFrame()

    for csv in csvlist:
        anim = get_anim(csv)
        df = pd.read_csv(csv,index_col=0)
        if 'Avg Max' in df:
            meanMax = pd.DataFrame({anim: df['Avg Max']}).T
            maxes = pd.concat([maxes, meanMax])
        else:
            curMaxes = pd.DataFrame({anim: df['Max Velocity']}).T
            maxes = pd.concat([maxes, curMaxes])


    return(maxes)

def concat_all_darting(csvlist, loc):
    freezing = pd.DataFrame()
    for csv in csvlist:
        anim = get_anim(csv)
        df = pd.read_csv(csv, index_col=0).T
        # print(loc)
        # print(anim)
        # print(csv)
        # try:
        percentF = pd.DataFrame([df.iloc[loc]], index=[anim])
        # except:
        #     print('FAILED' + str(loc) + ' '+ str(anim))
        #     percentF = pd.DataFrame([df.iloc[loc]], index=[anim])
        freezing = pd.concat([freezing, percentF])

    return(freezing)
    
def compress_data(csvlist,tbin):

    allanims = pd.DataFrame()
    for csv in csvlist:
        anim = get_anim(csv)
        df = pd.read_csv(csv,index_col=0).transpose()
        # print(csv)
        # print(tbin)
        tonevels = pd.DataFrame(df.iloc[tbin]).transpose()
        tonevels.set_index([[anim]],inplace=True)

        allanims = pd.concat([allanims,tonevels])

    return(allanims)
def compile_BaselineSR(trialType, inpath, outpath):
        baselineCSVs = scaredy_find_csvs(inpath,trialType + '-baseline')
        
        baselineData = concat_all_darting(baselineCSVs,2)
        outfile = os.path.join(outpath, 'All-'+ trialType + '-baseline.csv' )
        baselineData.to_csv(outfile)        
        
def compile_SR(trialType, epochLabel, numEpoch, num_dEpoch,dEpoch_list, behavior, inpath, outpath):
    # print(inpath+trialType + '-' + behavior)
    dartingCSVs = scaredy_find_csvs(inpath,trialType + '-' + epochLabel + '-darting')
    darting_outfile = os.path.join(outpath, 'All-'+ trialType + '-' + epochLabel + '-darting.csv' )
    dartingData = concat_all_darting(dartingCSVs,0)
    dartingData.to_csv(darting_outfile)
    
    freezingCSVs = scaredy_find_csvs(inpath,trialType + '-' + epochLabel + '-freezing')
    freezing_outfile = os.path.join(outpath, 'All-'+ trialType + '-' + epochLabel + '-Percent_freezing.csv' )
    freezingData = concat_all_freezing(freezingCSVs,0)
    freezingData.to_csv(freezing_outfile)
    
    meanCSVs = scaredy_find_csvs(inpath,trialType + '-mean')
    medCSVs = scaredy_find_csvs(inpath,trialType + '-median')
    maxCSVs = scaredy_find_csvs(inpath,trialType + '-max')
    SEMCSVs = scaredy_find_csvs(inpath,trialType + '-SEM')
    maxes = compress_data(maxCSVs,0)
    means = compress_data(meanCSVs,0)
    meds = compress_data(medCSVs,0)
    SEMs = compress_data(SEMCSVs,0)
    allData = concat_data(means,SEMs,meds,maxes,numEpoch)
    outfile = os.path.join(outpath,'All-'+ trialType + '-VelocitySummary.csv')
    allData.to_csv(outfile)
    
    meanECSVs = scaredy_find_csvs(inpath,trialType + '-' + epochLabel + '-mean')
    medECSVs = scaredy_find_csvs(inpath,trialType + '-' + epochLabel + '-median')
    maxECSVs = scaredy_find_csvs(inpath,trialType + '-' + epochLabel + '-max')
    SEMECSVs = scaredy_find_csvs(inpath,trialType + '-' + epochLabel + '-SEM')
    Emaxes = compress_data(maxECSVs,0)
    Emeans = compress_data(meanECSVs,0)
    Emeds = compress_data(medECSVs,0)
    ESEMs = compress_data(SEMECSVs,0)
    allData = concat_data(Emeans,ESEMs,Emeds,Emaxes,numEpoch)
    outfile = os.path.join(outpath,'All-'+ trialType + '-' + epochLabel + '-VelocitySummary.csv')
    allData.to_csv(outfile)
    
    Emaxes_single = concat_all_max(maxECSVs)
    outfile = os.path.join(outpath, 'All-'+ trialType + '-' +  epochLabel + '-MaxVel.csv' )
    Emaxes_single.to_csv(outfile)

    # allMax = concat_all_max(maxCSVs)
    # outfile = os.path.join(outpath, 'All-'+ trialType + '-MaxVel.csv' )
    # allMax.to_csv(outfile)

    num_dEpoch = len(dEpoch_list)
    # print(dEpoch_list)
    if num_dEpoch==0 or dEpoch_list == ['']:
        return
    # print(num_dEpoch)
    for i in range(0,num_dEpoch):
        # print(trialType + '-' + epochLabel + '_' + dEpoch_list[i] + '-max')
        maxdECSVs = scaredy_find_csvs(inpath, trialType + '-' + epochLabel + '_' + dEpoch_list[i] + '-max')
        # print(maxdECSVs)
        allMax = concat_all_max(maxdECSVs)
        outfile = os.path.join(outpath, 'All-'+ trialType + '-' +  epochLabel + '_' + dEpoch_list[i] + '-MaxVel.csv' )
        allMax.to_csv(outfile)
        
        meandECSVs = scaredy_find_csvs(inpath, trialType + '-' + epochLabel + '_' + dEpoch_list[i] + '-mean')
        meddECSVs = scaredy_find_csvs(inpath, trialType + '-' + epochLabel + '_' + dEpoch_list[i] + '-median')
        SEMdECSVs = scaredy_find_csvs(inpath, trialType + '-' + epochLabel + '_' + dEpoch_list[i] + '-SEM')
        maxes = compress_data(maxdECSVs,0)
        means = compress_data(meandECSVs,0)
        meds = compress_data(meddECSVs,0)
        SEMs = compress_data(SEMdECSVs,0)
        allData = concat_data(means,SEMs,meds,maxes,numEpoch)
        outfile = os.path.join(outpath,'All-'+ trialType + '-' +  epochLabel + '_' + dEpoch_list[i] + '-VelocitySummary.csv')
        allData.to_csv(outfile)
        
        dEdartingCSVs = scaredy_find_csvs(inpath,trialType + '-' + epochLabel + '_' + dEpoch_list[i] + '-darting')
        dEdarting_outfile = os.path.join(outpath, 'All-'+ trialType + '-' +  epochLabel + '_' + dEpoch_list[i] + '-darting.csv' )
        dEdartingData = concat_all_darting(dEdartingCSVs,0)
        dEdartingData.to_csv(dEdarting_outfile)
        
        dEfreezingCSVs = scaredy_find_csvs(inpath,trialType + '-' + epochLabel + '_' + dEpoch_list[i] + '-freezing')
        dEfreezing_outfile = os.path.join(outpath, 'All-'+ trialType + '-'+  epochLabel + '_' + dEpoch_list[i] + '-Percent_freezing.csv' )
        dEfreezingData = concat_all_freezing(dEfreezingCSVs,0)
        dEfreezingData.to_csv(dEfreezing_outfile)
        

