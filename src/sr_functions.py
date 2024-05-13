"""
Functions to perform analysis.
"""
# imports
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import math


# functions
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


# get_counts_and_times
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


# individual
def get_baseline_freezing_darting(datadict, freezing_threshold, darting_threshold, bin_secs):
    """
    Return a dataframe with the number of seconds spent freezing, the number of non-freezing seconds,
    and the percent of time spent freezing and a list of the freezing times at the baseline.
    """
    vels = datadict['Velocity']

    start_sec = int(round(vels.index[0], 0))
    end_sec = int(round(vels.index[-1], 0))
    time_range = range(start_sec, end_sec - 1, bin_secs)

    # get freezing and darting counts and time at baseline
    freezing_counts, freezing_times, darting_counts, darting_times = \
        get_counts_and_times(vels, freezing_threshold, darting_threshold, bin_secs, 'bfill', time_range)
    freezing_secs, nonfreezing_secs, percent_freezing = freezing_counts

    # freezing df
    tone_label = 'Baseline Freezing'
    names = ['Baseline Freezing (Time Bins)', 'Baseline Nonfreezing (Time Bins)', 'Baseline Percent Freezing']
    freezing = pd.DataFrame([[freezing_secs, nonfreezing_secs, percent_freezing]], index=[tone_label], columns=names)

    # darting df
    tone_label = 'Baseline Darting'
    darting = pd.DataFrame([[darting_counts]], index=[tone_label], columns=['Darts (count)'])

    return freezing, freezing_times, darting, darting_times


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


# plot_outputs
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


# individual
def plot_outputs(anim, anim_id, trial_type_full, outpath, prefix, ntones, fts, dts, epoch_label, print_settings,
                 print_labels):
    """
    Create plots summarizing animal behavior over time
    """
    colormap = [[245 / 256, 121 / 256, 58 / 256],
                [169 / 256, 90 / 256, 161 / 256],
                [133 / 256, 192 / 256, 249 / 256],
                [15 / 256, 32 / 256, 128 / 256]]

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
    for i in range(ntones):
        tone = find_tone_vels(anim, i, epoch_label)
        line2, = plt.plot(tone, color='c', linewidth=0.5, label=epoch_label.strip())

    if ntones > 0:
        handle_list.append(line2)
    # print('Trying to plot, 3')

    # Loops through shocks, plots each one in magenta
    line3 = []
    c_num = 0
    for i, label in enumerate(print_labels):
        # print(printSettings[i])
        # print(bool(printSettings[i][3]))
        if not print_settings[i] or len(print_settings[i]) < 4 or print_settings[i][3] == 'False':
            continue
        c_num = c_num % 4
        for j in range(ntones):
            response = find_delim_vels(anim, j, epoch_label, print_settings[i])
            line_tmp, = plt.plot(response, color=colormap[c_num], linewidth=0.5, label=print_labels[i])

        if 'line_tmp' in locals():
            handle_list.append(line_tmp)
        c_num += 1

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
    plt.title(anim_id + " " + trial_type_full)

    plt.legend(handles=handle_list, loc='best', fontsize='x-small', markerscale=0.6)

    plt.ylabel('Velocity (cm/s)')
    plt.xlabel('Trial time (s)')
    # print('Trying to plot, 6')
    # define where to save the fig
    fname = os.path.join(outpath,
                         f'{prefix}-plot-{anim_id}')  # prefix is either {trial_type_abbr} or {trial_type_abbr}-{epoch_label}

    plt.savefig(fname, dpi=300)
    plt.savefig(fname + '.eps', format='eps', dpi=300)
    # print('Trying to plot, 7')
    # plt.show()
    plt.close()
