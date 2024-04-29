import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import math


# print = sg.Print # Set print to go to a window rather than the terminal

# individual (run_SR)
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
    df.fillna(0,
              inplace=True)  # note: in previous version, this was done by replacing '-' chars (which were not read in as na_values), data type of some cols changed

    return animal, context, df


# find_delim_segment, find_tone_vels, get_recording_time_df
def get_label_df(df, i, label):
    """
    For the current epoch_label at the current number (ex: 'Tone' and 1),
    returns rows where column for that epoch (tone) contains a 1.
    """
    num = str(i + 1)

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


# get_vels, plot_outputs
def find_tone_vels(df, i, epoch_label):
    """
    For the current epoch_label at the current number (ex: 'Tone' and 1),
    filter rows where column for that epoch (tone) contains a 1, and return
    the velocity column from those rows.
    """
    return get_label_df(df, i, epoch_label)['Velocity']


# plot_outputs
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


# find_delim_based_time, find_delim_vels
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


# individual
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


# no usages
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


# individual
def find_delim_based_time(df, ndelim, delim, startidx, startoffset, stopoffset):
    """
    Create a dictionary with all pretones. Keys are indices, values
    are dataframes for 30s before tone == 1 (i.e., a df for each pretone inverval).
    """
    delim_dict = {}
    for i in range(ndelim):
        # get pretone recording time df and add to dict
        delim_dict[i] = get_recording_time_df(df, i, delim, startidx, startoffset, stopoffset)

    return delim_dict


# get_counts_and_times, get_freezing_counts_times
def get_vels_in_sec(vels, n, bin_secs, fill_method):
    """
    Return list of velocities in current time bin (where n is the approximate start time
    and n + bin_secs is the approximate end time), based on the dataframe vels.
    """
    start_of_second = vels.index.get_indexer([n], method=fill_method)[0]
    end_of_second = vels.index.get_indexer([round(n + (bin_secs - 0.1), 2)], method=fill_method)[0]
    vels_in_sec = [float(vels.iloc[frame]) for frame in range(start_of_second, end_of_second)]

    return vels_in_sec


# get_baseline_freezing, get_freezing_darting
def get_counts_and_times(vels, freezing_threshold, darting_threshold, bin_secs, fill_method, time_range):
    """
    For the given time_range, find the times below freezing threshold and the times above darting threshold,
    and calculate the total freezing seconds, non-freezing seconds, percent of time spent freezing,
    and darting seconds.
    note: darting_threshold should be None if baseline
    """
    freezing_times = []
    darting_times = []

    for n in time_range:
        vels_in_sec = get_vels_in_sec(vels, n, bin_secs, fill_method)
        if np.mean(vels_in_sec) < freezing_threshold:
            freezing_times.append([n, n + bin_secs])
        elif darting_threshold is not None:  # test to make sure that this should be elif and not if (i.e. that darting and freezing are truly mutually exclusive as they should be)
            if any(v > darting_threshold for v in vels_in_sec):
                darting_times.append([n, n + bin_secs])

    total_time = len(time_range)

    # freezing
    freezing_secs = len(freezing_times)
    nonfreezing_secs = total_time - freezing_secs
    percent_freezing = 100.0 * round(freezing_secs / total_time, 3)

    freezing_counts = [freezing_secs, nonfreezing_secs, percent_freezing]

    # darting
    darting_counts = len(darting_times)

    return freezing_counts, freezing_times, darting_counts, darting_times


# no usages
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
    percent_freezing = 100.0 * round(freezing_secs / total_time, 3)

    freezing_counts = [freezing_secs, nonfreezing_secs, percent_freezing]

    return freezing_counts, freezing_times


# individual
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


# no usages (testing only)
def get_baseline_freezing2(datadict, freezing_threshold, bin_secs):
    # freezing = pd.DataFrame()
    freezing_secs = 0
    nonfreezing_secs = 0
    freezing_times = []

    tone_label = 'Baseline Freezing'

    vels = datadict['Velocity']

    start_sec = int(round(vels.index[0], 0))
    end_sec = int(round(vels.index[-1], 0))
    for n in range(start_sec, end_sec - 1, bin_secs):
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
    percent_freezing = 100.0 * round(freezing_secs / (freezing_secs + nonfreezing_secs), 3)
    names = ['Baseline Freezing (Time Bins)', 'Baseline Nonfreezing (Time Bins)', 'Baseline Percent Freezing']
    freezing = pd.DataFrame([[freezing_secs, nonfreezing_secs, percent_freezing]], index=[tone_label], columns=names)

    return freezing, freezing_times


# no usages
def get_freezing(datadict, ntones, freezing_threshold, scope_name, bin_secs):
    """
    TEMP: placeholder until all usages updated
    TODO: update usages for work with get_freezing_darting instead
    Return a dataframe with the number of seconds spent freezing, the number of non-freezing seconds,
    and the percent of time spent freezing and a list of the freezing times for the given scope.
    """
    print('get freezing', datadict, ntones, freezing_threshold, scope_name, bin_secs)

    #
    # all_freezing = pd.DataFrame(columns=['Freezing  (Time Bins)', 'Nonfreezing  (Time Bins)', 'Percent Freezing'])
    # all_freezing_times = []
    #
    # for i in range(ntones):
    #     vels = datadict[i]['Velocity']
    #
    #     start_sec = round(vels.index[0], 2)
    #     end_sec = round(vels.index[-1], 2)
    #
    #     print(f'\nNumber of indices - {vels.index.size}\nStart - {start_sec}\nEnd - {end_sec}')
    #
    #     time_range = np.arange(start_sec, end_sec, bin_secs)
    #
    #     freezing_counts, freezing_times = get_freezing_counts_times(vels, freezing_threshold, bin_secs, 'nearest',
    #                                                                 time_range)
    #     freezing_secs, nonfreezing_secs, percent_freezing = freezing_counts
    #
    #     print(f'\n fs: {freezing_secs}\t nfs: {nonfreezing_secs}')
    #
    #     # add current values to freezing df as a new row with the tone_label as the index
    #     tone_label = f'{scope_name} {i + 1}'
    #     # df.loc[_not_yet_existing_index_label_] = new_row
    #     all_freezing.loc[tone_label] = [freezing_secs, nonfreezing_secs, percent_freezing]
    #
    #     # add freezing times to overall list
    #     all_freezing_times += freezing_times
    #
    # return all_freezing, all_freezing_times


# no usages
def get_freezing2(datadict, ntones, freezingThreshold, scopeName, binSecs):
    freezing = pd.DataFrame()
    freezingSecs = 0
    nonfreezingSecs = 0
    freezingTimes = []

    i = 0
    while i < ntones:
        toneLabel = scopeName + ' {}'.format(str(i + 1))
        # print(toneLabel)
        epoch = datadict[i]
        vels = epoch['Velocity']

        startSec = round(vels.index[0], 2)
        endSec = round(vels.index[-1], 2)
        print('\nNumber of indices - ', vels.index.size, '\nStart - ', startSec, '\nEnd - ', endSec)
        counter = 0
        for n in np.arange(startSec, endSec, binSecs):
            startOfSecond = vels.index.get_loc(n, method='nearest')
            endOfSecond = vels.index.get_loc(round(n + (binSecs - 0.1), 2), method='nearest')
            velsInSec = []
            # counter += endOfSecond-startOfSecond
            for frame in range(startOfSecond, endOfSecond):
                velocity = float(vels.iloc[frame])
                velsInSec.append(velocity)
                counter += 1
            if np.mean(velsInSec) < freezingThreshold:
                freezingSecs += 1
                freezingTimes.append([n, n + binSecs])
            else:
                nonfreezingSecs += 1
        # print('\nCounter - ', counter)
        print('\n fs: ', freezingSecs, '\t nfs: ', nonfreezingSecs)
        percentFreezing = 100.0 * round(freezingSecs / (freezingSecs + nonfreezingSecs), 3)
        toneFreezing = pd.DataFrame({toneLabel: [freezingSecs, nonfreezingSecs, percentFreezing]},
                                    index=['Freezing  (Time Bins)', 'Nonfreezing  (Time Bins)', 'Percent Freezing']).T
        freezing = pd.concat([freezing, toneFreezing])
        freezingSecs = 0
        nonfreezingSecs = 0
        i += 1
    return (freezing, freezingTimes)


# no usages
def get_darting(datadict, ntones, dart_threshold, scope_name, bin_secs):
    """
    TEMP: placeholder until all usages updated
    TODO: update usages for work with get_freezing_darting instead
    """
    print('get darting', datadict, ntones, dart_threshold, scope_name, bin_secs)


# no usages
def get_darting2(datadict, ntones, dartThreshold, scopeName, binSecs):
    darting = pd.DataFrame()
    dartingTimes = []
    nDarts = 0

    i = 0
    while i < ntones:
        toneLabel = scopeName + ' {}'.format(str(i + 1))

        epoch = datadict[i]
        vels = epoch['Velocity']

        startSec = (round(vels.index[0], 2))
        endSec = (round(vels.index[-1], 2))

        for n in np.arange(startSec, endSec, binSecs):
            startOfSecond = vels.index.get_loc(n, method='nearest')
            endOfSecond = vels.index.get_loc(round(n + (binSecs - 0.1), 2), method='nearest')
            velsInSec = []
            for frame in range(startOfSecond, endOfSecond):
                velocity = float(vels.iloc[frame])
                velsInSec.append(velocity)
            for v in velsInSec:
                if v > dartThreshold:
                    nDarts += 1
                    dartingTimes.append([n, n + binSecs])
                    break

        toneDarting = pd.DataFrame({toneLabel: nDarts}, index=['Darts (count)']).T
        darting = pd.concat([darting, toneDarting])
        nDarts = 0
        i += 1
    return (darting, dartingTimes)


# individual
def get_freezing_darting(datadict, ntones, freezing_threshold, darting_threshold, scope_name, bin_secs):
    """
    Return data frames summarizing freezing and darting information, and lists of freezing and darting times.
    """
    # init dfs and lists
    all_freezing = pd.DataFrame(columns=['Freezing  (Time Bins)', 'Nonfreezing  (Time Bins)', 'Percent Freezing'])
    all_darting = pd.DataFrame(columns=['Darts (count)'])
    all_freezing_times = []
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
        # print(tone_label, len(freezing_times), len(darting_times))
        all_darting_times += darting_times

    return all_freezing, all_freezing_times, all_darting, all_darting_times


# individual
def plot_outputs(anim, id, trial_type_full, outpath, prefix, ntones, fts, dts, epoch_label, print_settings, print_labels):
    colormap = [[245 / 256, 121 / 256, 58 / 256],
                [169 / 256, 90 / 256, 161 / 256],
                [133 / 256, 192 / 256, 249 / 256],
                [15 / 256, 32 / 256, 128 / 256]]

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
    handle_list = []
    # plot stuff
    vels = pd.DataFrame(anim['Velocity'])
    # print('Trying to plot, 1')
    plt.style.use('seaborn-white')
    plt.figure(figsize=(16, 8), facecolor='white', edgecolor='white')
    plt.axhline(linewidth=2, color='black')
    plt.axvline(linewidth=2, color='black')
    parameters = {'axes.labelsize': 22,
                  'axes.titlesize': 35,
                  'xtick.labelsize': 18,
                  'ytick.labelsize': 18,
                  'legend.fontsize': 18}
    plt.rcParams.update(parameters)
    # Plots main velocity in black
    line1, = plt.plot(vels, color='k', linewidth=0.1, label='ITI')
    handle_list.append(line1)
    # print('Trying to plot, 2')
    # Loops through tones, plots each one in cyan
    i = 0
    # has_tone = False
    # while i < ntones:
    for i in range(ntones):
        tone = find_tone_vels(anim, i, epoch_label)
        line2, = plt.plot(tone, color='c', linewidth=0.5, label=epoch_label.strip())
        # i += 1
        # has_tone = True
    if ntones > 0:
        handle_list.append(line2)
    # print('Trying to plot, 3')
    # Loops through shocks, plots each one in magenta
    line3 = []
    # hasShock = False
    c_num = 0
    for i, label in enumerate(print_labels):
        # print(printSettings[i])
        # print(bool(printSettings[i][3]))
        if not print_settings[i] or len(print_settings[i]) < 4 or print_settings[i][3] == 'False':
            continue
        c_num = c_num % 4
        # hasShock = True
        for j in range(ntones):
            response = find_delim_vels(anim, j, epoch_label, print_settings[i])
            line_tmp, = plt.plot(response, color=colormap[c_num], linewidth=0.5, label=print_labels[i])

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
    for timebin in fts:
        plt.plot([timebin[0], timebin[1]], [-0.3, -0.3], color='#ff4f38', linewidth=3)

    # Loops through darting bins, plots each below the x-axis
    # print(DTs)
    for timebin in dts:
        plt.plot([timebin[0], timebin[1]], [-0.7, -0.7], color='#167512', linewidth=3)

    plt.ylim(-1, 35)
    # print('Trying to plot, 5')
    sns.despine(left=True, bottom=True, right=True)
    plt.title(id + " " + trial_type_full)

    plt.legend(handles=handle_list, loc='best', fontsize='x-small', markerscale=0.6)

    plt.ylabel('Velocity (cm/s)')
    plt.xlabel('Trial time (s)')
    # print('Trying to plot, 6')
    # define where to save the fig
    # fname = outpath + '/' + trial_type + '-' + epoch_label + '-plot-{}'
    # fname = fname.format(id)
    fname = os.path.join(outpath, f'{prefix}-plot-{id}')  # prefix is either {trial_type_abbr} or {trial_type_abbr}-{epoch_label}

    plt.savefig(fname, dpi=300)
    plt.savefig(fname + '.eps', format='eps', dpi=300)
    # print('Trying to plot, 7')
    # plt.show()
    plt.close()
    # return ()


# individual
def get_means(datadict, timebin, ntones):
    """
    Returns a dataframe of mean velocity of timebin at each tone.
    """
    meanlist = []
    # i = 0
    for i in range(ntones):
        epoch = datadict[i]
        vels = epoch['Velocity']
        mean = round(np.mean(vels), 3)
        meanlist.append(mean)
        # i += 1
    means = pd.DataFrame(meanlist, columns=[timebin + ' Mean Velocity'])
    means.index = np.arange(1, len(meanlist) + 1)
    return means


# individual
def get_meds(datadict, timebin, ntones):
    """
    Returns a dataframe of median velocity of timebin at each tone.
    """
    medlist = []
    # i = 0
    for i in range(ntones):
        epoch = datadict[i]
        vels = epoch['Velocity']
        med = round(np.median(vels), 3)
        medlist.append(med)
        # i += 1
    meds = pd.DataFrame(medlist, columns=[timebin + ' Median Velocity'])
    meds.index = np.arange(1, len(meds) + 1)
    return meds


# individual
def get_sems(datadict, timebin, ntones):
    """
    Returns a dataframe of median velocity of timebin at each tone.
    """
    semlist = []
    # i = 0
    for i in range(ntones):
        epoch = datadict[i]
        vels = epoch['Velocity']
        sem = round(np.std(vels), 3)
        semlist.append(sem)
        # i += 1
    sems = pd.DataFrame(semlist, columns=[timebin + 'SEM'])
    sems.index = np.arange(1, len(sems) + 1)
    return sems


# no usages
def get_vels(df, ntones):
    """
    Creates data of all velocities from original dataframe for plotting.
    """
    tonevels = {}
    for i in range(ntones):  # number of tones
        # vels = []
        # num = str(i + 1)
        # try:
        #     label = 'Tone ' + num
        #     tone = pd.DataFrame(df[df[label] == 1])
        # except:
        #     label = 'Tone' + num
        #     tone = pd.DataFrame(df[df[label] == 1])
        # vels.append(tone['Velocity'])

        # NOTE: might need to change this so that vels is a list
        vels = find_tone_vels(df, i, 'Tone')
        tonevels[i] = vels

    return tonevels


# individual
def get_top_vels(datadict, nmax, binlabel, ntones):
    """
    Returns dataframe of nmax (int) maximum velocities for a timebin.
    The second section adds a column for an average of the maxima.
    """
    nmax_list = []
    idx = []
    for i in range(ntones):
        epoch = datadict[i]
        vels = epoch['Velocity']
        vlist = vels.to_list()
        if not vlist:
            continue
        elif nmax == 1:
            topvels = [max(vlist)]
        else:
            topvels = sorted(vlist)[-nmax:]
        nmax_list.append(topvels)
        idx.append(f'{binlabel} {i + 1} Max Velocity')

    nmaxes = pd.DataFrame(nmax_list)
    if nmax > 1:
        nmaxes.index = np.arange(1, nmaxes.shape[0] + 1)
        nmaxes.columns = np.arange(1, nmaxes.shape[1] + 1)

        nmaxes['Avg Max'] = nmaxes.mean(axis=1)
    else:
        nmaxes.index = idx
        nmaxes.columns = ['Max Velocity']

    return nmaxes


def get_top_vels2(datadict, nmax, binlabel, ntones):
    """
    Returns dataframe of nmax (int) maximum velocities for a timebin.
    The second section adds a column for an average of the maxima.
    """
    nmaxes = pd.DataFrame()
    col_list = []
    # i = 0
    for i in range(ntones):
        epoch = datadict[i]
        vels = epoch['Velocity']
        vlist = vels.tolist()
        vlist.sort()
        if not vlist:
            # i += 1
            continue
        elif nmax == 1:
            topvels = pd.DataFrame([vlist[-1]])
        else:
            topvels = pd.DataFrame([vlist[-nmax:-1]])
        # nmaxes = nmaxes.append(topvels)
        nmaxes = pd.concat([nmaxes, topvels])
        col_list.append(binlabel + str(i + 1) + ' Max Velocity')
        # i += 1
    if nmax > 1:
        nmaxes.index = np.arange(1, nmaxes.shape[0] + 1)
        nmaxes.columns = np.arange(1, nmaxes.shape[1] + 1)

        nmaxes['Avg Max'] = nmaxes.mean(axis=1)
    else:
        nmaxes.index = col_list
        nmaxes.columns = ['Max Velocity']

    return nmaxes


# compress_data, concat_all_max
def get_anim(csv):
    exp_rgx = re.compile(r'.*-([a-zA-Z]+-?\d+)\.[a-zA-Z]+$')
    match = exp_rgx.match(csv)

    if match:
        anim = match.group(1)
        return anim
    else:
        raise ValueError('File name does not contain valid animal ID.')


# compiled stuff: all_darting_out, all_freezing_out, all_velocity_out, compile_baseline_sr
def scaredy_find_csvs(csv_dir, prefix):
    """
    Return list of csvs in given dir with given prefix
    """
    csvlist = []

    for root, dirs, names in os.walk(csv_dir):
        for file in names:
            # print(file)
            if file.startswith(prefix):
                f = os.path.join(root, file)
                # print(f)
                csvlist.append(f)

    return csvlist


# all_velocity_out
def concat_vel_data(means, sems, meds, maxes, ntones):
    """
    Return a dataframe that contains the given means, SEMs, meds, and maxes
    """
    all_data = pd.DataFrame()

    for key, summary_df in {"Mean": means, "SEM": sems, "Median": meds, "Max": maxes}.items():
        if not summary_df.empty:
            if all_data.empty:
                # initialize the all_data index to be the first non-empty input
                # all non-empty input frames should have the same index (if they don't, something has gone wrong)
                all_data.index = summary_df.index
            summary_df = summary_df.add_prefix('Tone ')
            summary_df = summary_df.add_suffix(f' {key}')
            # inner join onto existing data, joining on index
            # indexes should always be an exact match (see above note)
            all_data = all_data.join(summary_df)

    # ix = []
    # for n in range(ntones):
    #     if means.empty: continue
    #     all_data = all_data.append(means.iloc[:, n])
    #     ix.append('Tone {} Mean'.format(n + 1))
    # for n in range(ntones):
    #     if sems.empty: continue
    #     all_data = all_data.append(sems.iloc[:, n])
    #     ix.append('Tone {} SEM'.format(n + 1))
    # for n in range(ntones):
    #     if meds.empty: continue
    #     all_data = all_data.append(meds.iloc[:, n])
    #     ix.append('Tone {} Median'.format(n + 1))
    # for n in range(ntones):
    #     if maxes.empty: continue
    #     all_data = all_data.append(maxes.iloc[:, n])
    #     ix.append('Tone {} Max'.format(n + 1))
    #
    # all_data.index = ix
    # all_data = all_data.transpose()

    return all_data


# compiled stuff: all_velocity_out, concat_all_darting, concat_all_freezing
def compress_data(csvlist, row):
    """
    Get the given time bin from each of the CSVs in the given list, and combine the data into a single df with the
    animal id as the index.
    """
    all_anims = pd.DataFrame()
    for csv in csvlist:
        anim = get_anim(csv)
        df = pd.read_csv(csv, index_col=0).transpose()
        # print(csv)
        # print(tbin)
        # tonevels = df.iloc[:, [tbin]]
        # tonevels = pd.DataFrame(df.iloc[tbin]).transpose()
        # tonevels = df.iloc[[row]]
        # tonevels.set_index([[anim]], inplace=True)
        curr_anim_val = pd.DataFrame([df.iloc[row]], index=[anim])
        all_anims = pd.concat([all_anims, curr_anim_val])

    return all_anims


# all_freezing_out
def concat_all_freezing(csvlist, tbin):
    """
    Return the combined freezing data for the given time bin for all animals in the CSV list
    """
    # freezing = pd.DataFrame()
    # for csv in csvlist:
    #     anim = get_anim(csv)
    #     df = pd.read_csv(csv, index_col=0).T
    #     loc = (tbin * 3) + 2
    #     percentF = pd.DataFrame([df.iloc[loc]], index=[anim])
    #     freezing = pd.concat([freezing, percentF])
    #
    # return (freezing)
    row = (tbin * 3) + 2  # for freezing, row is slightly offset from time bin
    freezing = compress_data(csvlist, row)
    return freezing


# all_darting_out
def concat_all_darting(csvlist, loc):
    """
    Return the combined darting data for the given time bin for all animals in the CSV list
    """
    # freezing = pd.DataFrame()
    # for csv in csvlist:
    #     anim = get_anim(csv)
    #     df = pd.read_csv(csv, index_col=0).T
    #     # print(loc)
    #     # print(anim)
    #     # print(csv)
    #     # try:
    #     percentF = pd.DataFrame([df.iloc[loc]], index=[anim])
    #     # except:
    #     #     print('FAILED' + str(loc) + ' '+ str(anim))
    #     #     percentF = pd.DataFrame([df.iloc[loc]], index=[anim])
    #     freezing = pd.concat([freezing, percentF])
    #
    # return (freezing)
    darting = compress_data(csvlist, loc)
    return darting


# all_velocity_out
def concat_all_max(csvlist):
    """
    Combine all the average max or max velocity values from the list of CSVs
    """
    maxes = pd.DataFrame()

    for csv in csvlist:
        anim = get_anim(csv)
        df = pd.read_csv(csv, index_col=0)
        if 'Avg Max' in df:
            curr_max = pd.DataFrame({anim: df['Avg Max']}).T
        else:
            curr_max = pd.DataFrame({anim: df['Max Velocity']}).T

        maxes = pd.concat([maxes, curr_max])

    return maxes


# individual? in run_SR but is compiled data?
def compile_baseline_sr(trial_type, inpath, outpath):
    """
    Combine the data from the csvs for the baseline measurements for each animal into a single csv file.
    """
    baseline_csvs = scaredy_find_csvs(inpath, trial_type + '-baseline')

    baseline_data = concat_all_darting(baseline_csvs, 2)
    outfile = os.path.join(outpath, 'All-' + trial_type + '-baseline.csv')
    baseline_data.to_csv(outfile)


# individual (still needs to be added)
# add additional outputted csvs here?
# TO DO: update/test this where the actual output is created (move to other file?)
# rough outline of function to write additional freezing/darting csvs
# take in df with animal data, animal id?, freezing threshold, darting threshold, epoch label, bin size (secs), output path
def freezing_darting_times(anim, id, freezing_threshold, darting_threshold, trial_type, epoch_label, bin_secs, outpath):
    """
    TODO: cols: start and stop, numbered index; one for freezing, one for darting
    note: have to call freezing/darting separately because looking at ALL behavior, not just tone response behavior
    number of tones in set to 1, and animal data frame is supplied as a data dict with a single item (potentially change
    get_freezing_darting behavior to improve this)
    """
    # get freezing and darting time data
    freezing_darting = get_freezing_darting({0: anim}, 1, freezing_threshold, darting_threshold, epoch_label, bin_secs)

    # extract freezing and darting times
    freezing_times = freezing_darting[1]
    darting_times = freezing_darting[3]

    # create data frames with the start and stop times
    freezing_time_df, darting_time_df = (pd.DataFrame(times, columns=['start', 'stop'])
                                         for times in (freezing_times, darting_times))

    # write data to csv file output using outpath
    freezing_time_df.to_csv(outpath, f'{trial_type}-{epoch_label}-freezing-times-{id}')
    darting_time_df.to_csv(outpath, f'{trial_type}-{epoch_label}-darting-times-{id}')


# compiled
def all_darting_out(prefix, inpath, outpath):
    """
    Combine the data from all darting csvs and write to file
    """
    darting_csvs = scaredy_find_csvs(inpath, f'{prefix}-darting')
    darting_outfile = os.path.join(outpath, f'All-{prefix}-darting.csv')
    darting_data = concat_all_darting(darting_csvs, 0)
    darting_data.to_csv(darting_outfile)


# compiled
def all_freezing_out(prefix, inpath, outpath):
    """
    Combine the data from all freezing csvs and write to file
    """
    freezing_csvs = scaredy_find_csvs(inpath, f'{prefix}-freezing')
    freezing_outfile = os.path.join(outpath, f'All-{prefix}-Percent_freezing.csv')
    freezing_data = concat_all_freezing(freezing_csvs, 0)
    freezing_data.to_csv(freezing_outfile)


# compiled
def all_velocity_out(prefix, inpath, outpath, num_epoch, full_analysis, epoch_level=False):
    """
    Combine data from all csvs related to general velocity measurements and write to output file.
    If epoch_level, also write more detailed max data csv.
    """
    max_csvs = scaredy_find_csvs(inpath, f'{prefix}-max')

    # full velocity summary only for full analysis (else just max)
    if full_analysis:
        mean_csvs = scaredy_find_csvs(inpath, f'{prefix}-mean')
        med_csvs = scaredy_find_csvs(inpath, f'{prefix}-median')
        sem_csvs = scaredy_find_csvs(inpath, f'{prefix}-SEM')

        # combine data into a single df for each csv list
        means, meds, maxes, sems = [compress_data(csvs, 0) for csvs in [mean_csvs, med_csvs, max_csvs, sem_csvs]]

        # merge data frames into one with all data
        all_data = concat_vel_data(means, sems, meds, maxes, num_epoch)

        outfile = os.path.join(outpath, f'All-{prefix}-VelocitySummary.csv')
        all_data.to_csv(outfile)

    if epoch_level:
        # combine a more detailed version of the epoch max velocity and write to its own file
        e_maxes_single = concat_all_max(max_csvs)
        outfile = os.path.join(outpath, f'All-{prefix}-MaxVel.csv')
        e_maxes_single.to_csv(outfile)


# compiled
def all_subepoch_out(d_epoch_list, prefix, inpath, outpath, num_epoch, full_analysis):
    """
    Combine and output velocity, freezing, and darting data for each sub(derived)-epoch in the given list.
    """
    # check whether the sub-epoch input list is valid
    if not d_epoch_list or not d_epoch_list[0]:
        return
    for d_epoch in d_epoch_list:
        # add current sub-epoch name to prefix
        d_epoch_prefix = f'{prefix}_{d_epoch}'

        # velocity summary data
        all_velocity_out(d_epoch_prefix, inpath, outpath, num_epoch, full_analysis, True)

        # darting summary data
        all_darting_out(d_epoch_prefix, inpath, outpath)

        # freezing summary data
        all_freezing_out(d_epoch_prefix, inpath, outpath)
