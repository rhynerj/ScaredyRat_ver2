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
    """
    Return animal ID, trial context, and dataframe of trial data with long header removed.
    Use correct column labels and time as row index.
    """
    filepath = os.path.join(inpath, filename)
    try:
        df = pd.read_excel(filepath, sheet_name=sheet, index_col=0, na_values='-')
    except (FileNotFoundError, ValueError) as err:
        print(err)
        return -1, -1, -1

    animal = df.loc['Animal ID', '40'].strip()

    context = df.loc['Trial Control settings', '40'].strip()

    print(f"\n{filename} {sheet} is {animal} in {context}")

    df.columns = df.loc['Trial time']
    df.columns.names = ['']
    df = df.loc[0:, :]
    df.index.names = ['Trial time']
    df.fillna(0, inplace=True)  # note: in previous version, this was done by replacing '-' chars (which were not read in as na_values), data type of some cols changed

    return animal, context, df


def get_label_df(df, i, label):
    """
    For the current epoch_label at the current number (ex: 'Tone' and 1),
    returns rows where column for that epoch (tone) contains a 1.
    """
    num = str(i+1)

    label = label.strip()  # strip label to standardize

    # print(epochLabel)
    try:
        # default: assume epoch_label should have space
        label = f'{label} {num}'
        # print(label)
    except KeyError:
        # if epoch_label shouldn't contain space
        label += num
        # print(label)
    tone = df[df[label] == 1]

    return tone


def find_tone_vels(df, i,  epoch_label):
    """
    For the current epoch_label at the current number (ex: 'Tone' and 1),
    filter rows where column for that epoch (tone) contains a 1, and return
    the velocity column from those rows.
    """
    return get_label_df(df, i, epoch_label)['Velocity']


def find_delim_vels(df, i, epoch, delim_times):
    """
    Return the 'Velocity' column from the given df for the designated time range,
    where the time range is the interval specified by delim_times for the given epoch.

    Note: delim_times is list: eg, ['0', '-30', '0', 'True'] (this is for pretone),
    where items in list are: start_index, start_offset, stop_offset
    """

    # convert everything but the plot flag (last item) in delim times to an int
    start_idx, start_offset, stop_offset = map(int, delim_times[:-1])

    # get subsetted df for tone and time frame
    tf_df = get_recording_time_df(df, i, epoch, start_idx, start_offset, stop_offset)['Velocity']

    return tf_df


def get_recording_time_df(df, i, label, start_idx, start_offset, stop_offset):
    """
    For the given df, return a df where the column designated by the given i and label are 1
    for the time frame derived from the given start_idx, start_offset, and stop_offset times.
    """
    label_df = get_label_df(df, i, label)  # df for current label

    # get floored start and end times
    base_time = label_df.iloc[start_idx]['Recording time']
    start_time = math.floor(base_time + start_offset)
    end_time = math.floor(base_time + stop_offset)

    # print(start_time, end_time)

    # select df of start to end times,
    # where actual times are those calculated above or the nearest lesser value
    s_idx = df.index.get_indexer([start_time], method='bfill')[0]
    e_idx = df.index.get_indexer([end_time], method='bfill')[0]
    # print(s_idx, e_idx)

    return df.iloc[s_idx:e_idx]


def find_delim_segment(df, ndelim, delim):
    """
    Create dictionary for each delimited segment. Values are
    dataframes for times at which each delimiter == 1.
    Takes in the dataframe (df), the number of delimiters (ndelim)
    and the base delimiter string (delim).  It is assumed
    that each delimiter is either labeled as delim + ' ' + num
    or delim + num.

    Return a dictionary of dataframe (datarray) is made up of the
    delimited periods.
    """

    datarray = {}
    # print(delim)
    for i in range(ndelim):  # number of delimiters
        datarray[i] = get_label_df(df, i, delim)

    return datarray


def find_timeperiod(df, nperiods, starttime, endtime):
    """
    CANDIDATE FOR DELETION: no found usages
    Create a dictionary of the given time periods
    of the recording in order to calculate
    behaviors.
    """
    datarray = {}
    for i in range(nperiods):
        stime = starttime[i]
        stime_idx = df.index.get_indexer([stime], method='bfill')[0]

        etime = endtime[i]
        etime_idx = df.index.get_indexer([etime], method='bfill')[0]

        dat = df.iloc[stime_idx:etime_idx]
        datarray[i] = dat

    return datarray


def find_delim_based_time(df, ndelim, delim, startidx, startoffset, stopoffset):
    """
    Create a dictionary with all pretones. Keys are indices, values
    are dataframes for 30s before tone == 1 (i.e., a df for each pretone inverval).
    """
    datarray = {}
    for i in range(ndelim):
        # get pretone recording time df and add to dict
        datarray[i] = get_recording_time_df(df, i, delim, startidx, startoffset, stopoffset)

    return datarray


def get_vels_in_sec(vels, n, bin_secs, fill_method):
    """
    Return list of velocities in current time bin (where n is the approximate start time
    and n + bin_secs is the approximate end time), based on the dataframe vels.
    """
    start_of_second = vels.index.get_indexer([n], method=fill_method)[0]
    end_of_second = vels.index.get_indexer([round(n + (bin_secs - 0.1), 2)], method=fill_method)[0]
    vels_in_sec = [float(vels.iloc[frame]) for frame in range(start_of_second, end_of_second)]

    return vels_in_sec


def get_counts_and_times(vels, freezing_theshold, darting_threshold, bin_secs, fill_method, time_range):
    """
    For the given time_range, find the times below freezing theshold and the times above darting threshold,
    and calculate the total freezing seconds, non-freezing seconds, percent of time spent freezing,
    and darting seconds.
    note: darting_threshold should be None if baseline
    """
    freezing_times = []
    darting_times = []

    for n in time_range:
        vels_in_sec = get_vels_in_sec(vels, n, bin_secs, fill_method)
        if np.mean(vels_in_sec) < freezing_theshold:
            freezing_times.append([n, n + bin_secs])
        elif darting_threshold is not None:  # test to make sure that this should be elif and not if (i.e. that darting and freezing are truly mutually exclusive as they should be)
            if any(v > darting_threshold for v in vels_in_sec):
                darting_times.append([n, n + bin_secs])

    total_time = len(time_range)

    # freezing
    freezing_secs = len(freezing_times)
    nonfreezing_secs = total_time - freezing_secs
    percent_freezing = 100.0 * round(freezing_secs/total_time, 3)

    freezing_counts = [freezing_secs, nonfreezing_secs, percent_freezing]

    # darting
    darting_counts = len(darting_times)

    return freezing_counts, freezing_times, darting_counts, darting_times


def get_freezing_counts_times(vels, freezing_theshold, bin_secs, fill_method, time_range):
    """
    SLATED FOR REPLACEMENT BY get_counts_and_times
    Return the number of seconds spent freezing, the number of non-freezing seconds,
    the percent of time spent freezing, and the freezing times (as a list of lists, where the
    internal list consists of the start time and the end time - NOTE: CHECK THIS ASSUMPTION).
    """
    freezing_times = []

    for n in time_range:
        vels_in_sec = get_vels_in_sec(vels, n, bin_secs, fill_method)
        if np.mean(vels_in_sec) < freezing_theshold:
            freezing_times.append([n, n + bin_secs])

    total_time = len(time_range)
    freezing_secs = len(freezing_times)
    nonfreezing_secs = total_time - freezing_secs
    percent_freezing = 100.0 * round(freezing_secs/total_time, 3)

    freezing_counts = [freezing_secs, nonfreezing_secs, percent_freezing]

    return freezing_counts, freezing_times


def get_baseline_freezing(datadict, freezing_threshold, bin_secs):
    """
    Return a dataframe with the number of seconds spent freezing, the number of non-freezing seconds,
    and the percent of time spent freezing and a list of the freezing times at the baseline.
    """
    vels = datadict['Velocity']

    start_sec = int(round(vels.index[0], 0))
    end_sec = int(round(vels.index[-1], 0))
    time_range = range(start_sec, end_sec - 1, bin_secs)

    # get freezing counts and time (first two items returned by get_counts_and_times) at baseline
    # set darting_threshold to None because not applicable
    freezing_counts, freezing_times = \
        get_counts_and_times(vels, freezing_threshold, None, bin_secs, 'bfill', time_range)[:2]
    freezing_secs, nonfreezing_secs, percent_freezing = freezing_counts

    tone_label = 'Baseline Freezing'
    names = ['Baseline Freezing (Time Bins)', 'Baseline Nonfreezing (Time Bins)', 'Baseline Percent Freezing']
    freezing = pd.DataFrame([[freezing_secs, nonfreezing_secs, percent_freezing]], index=[tone_label], columns=names)

    return freezing, freezing_times


def get_baseline_freezing2(datadict, freezing_threshold, bin_secs):
    # freezing = pd.DataFrame()
    freezing_secs = 0
    nonfreezing_secs = 0
    freezing_times = []

    tone_label = 'Baseline Freezing'

    vels = datadict['Velocity']

    start_sec = int(round(vels.index[0], 0))
    end_sec = int(round(vels.index[-1], 0))
    for n in range(start_sec, end_sec-1, bin_secs):
        start_of_second = vels.index.get_indexer([n], method='bfill')[0]
        end_of_second = vels.index.get_indexer([round(n + (bin_secs - 0.1), 2)], method='bfill')[0]
        vels_in_sec = []
        for frame in range(start_of_second, end_of_second):
            velocity = float(vels.iloc[frame])
            vels_in_sec.append(velocity)
        if np.mean(vels_in_sec) < freezing_threshold:
            freezing_secs += 1
            freezing_times.append([n, n + bin_secs])
        else:
            nonfreezing_secs += 1
    percent_freezing = 100.0 * round(freezing_secs/(freezing_secs + nonfreezing_secs), 3)
    names = ['Baseline Freezing (Time Bins)', 'Baseline Nonfreezing (Time Bins)', 'Baseline Percent Freezing']
    freezing = pd.DataFrame([[freezing_secs, nonfreezing_secs, percent_freezing]], index=[tone_label], columns=names)

    return freezing, freezing_times


def get_freezing(datadict, ntones, freezing_threshold, scope_name, bin_secs):
    """
    Return a dataframe with the number of seconds spent freezing, the number of non-freezing seconds,
    and the percent of time spent freezing and a list of the freezing times for the given scope.
    """
    all_freezing = pd.DataFrame(columns=['Freezing  (Time Bins)', 'Nonfreezing  (Time Bins)', 'Percent Freezing'])
    all_freezing_times = []

    for i in range(ntones):
        vels = datadict[i]['Velocity']

        start_sec = round(vels.index[0], 2)
        end_sec = round(vels.index[-1], 2)

        print(f'\nNumber of indices - {vels.index.size}\nStart - {start_sec}\nEnd - {end_sec}')

        time_range = np.arange(start_sec, end_sec, bin_secs)

        freezing_counts, freezing_times = get_freezing_counts_times(vels, freezing_threshold, bin_secs, 'nearest', time_range)
        freezing_secs, nonfreezing_secs, percent_freezing = freezing_counts

        print(f'\n fs: {freezing_secs}\t nfs: {nonfreezing_secs}')

        # add current values to freezing df as a new row with the tone_label as the index
        tone_label = f'{scope_name} {i + 1}'
        # df.loc[_not_yet_existing_index_label_] = new_row
        all_freezing.loc[tone_label] = [freezing_secs, nonfreezing_secs, percent_freezing]

        # add freezing times to overall list
        all_freezing_times += freezing_times

    return all_freezing, all_freezing_times


def get_freezing2(datadict, ntones, freezingThreshold, scopeName, binSecs):
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


def get_darting(datadict, ntones, dart_threshold, scope_name, bin_secs):
    """
    TEMP: placeholder until all usages updated
    TODO: update usages for work with get_counts_and_times instead
    """
    get_freezing_darting()


def get_darting2(datadict, ntones, dartThreshold, scopeName, binSecs):
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


def get_freezing_darting(datadict, ntones, freezing_threshold, darting_threshold, scope_name, bin_secs):
    """
    Return data frames summarizing freezing and darting information, and lists of freezing and darting times.
    """
    # init dfs and lists
    all_freezing = pd.DataFrame(columns=['Freezing  (Time Bins)', 'Nonfreezing  (Time Bins)', 'Percent Freezing'])
    all_freezing_times = []
    all_darting = pd.DataFrame(columns=['Darts (count)'])
    all_darting_times = []

    # iterate over tones
    for i in range(ntones):
        vels = datadict[i]['Velocity']

        start_sec = round(vels.index[0], 2)
        end_sec = round(vels.index[-1], 2)

        # find freezing and darting data starting at the start_sec, ending at end_sec, and increasing in the interval
        # specified by bin_secs
        time_range = np.arange(start_sec, end_sec, bin_secs)

        freezing_counts, freezing_times, darting_counts, darting_times = \
            get_counts_and_times(vels, freezing_threshold, darting_threshold, bin_secs, 'nearest', time_range)
        freezing_secs, nonfreezing_secs, percent_freezing = freezing_counts  # break freezing counts down into sub-parts

        # print(f'\n fs: {freezing_secs}\t nfs: {nonfreezing_secs}')

        # add current values to freezing and darting dfs as a new row with the tone_label as the index
        tone_label = f'{scope_name} {i + 1}'
        # df.loc[_not_yet_existing_index_label_] = new_row
        all_freezing.loc[tone_label] = [freezing_secs, nonfreezing_secs, percent_freezing]
        all_darting.loc[tone_label] = [darting_counts]

        # add freezing and darting times to overall lists
        all_freezing_times += freezing_times
        all_darting_times += darting_times

    return all_freezing, all_freezing_times, all_darting, all_darting_times


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
        

