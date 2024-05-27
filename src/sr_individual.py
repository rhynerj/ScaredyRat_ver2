"""
Functions to run analysis for individual animals and output appropriate analysis files.
"""
# imports
import pandas as pd
import os

import src.sr_functions as srf


# functions
def get_file_list(inpath):
    """Return list of all files in dir at given inpath."""
    # search the inpath for files, make list
    filelist = []
    for entry in os.scandir(inpath):
        if entry.is_file() and not entry.name.startswith('~'):
            filelist.append(entry.path)

    return filelist


def get_trial_info(ctx, sheet_settings, epoch_settings):
    """
    Return the list of Epoch, the trial type label (context), the full trial type name, and the trial type abbreviation
    for the given trial context (ctx), based on the given sheet and epoch settings.
    """
    # note: make sure that returning list of Epoch actually works here

    try:
        # get TrialType whose detection label matches specified context
        trial_type = sheet_settings.get_trial_type_by_label(ctx)
        # retrieve epoch list for context
        epochs = epoch_settings[ctx]
        # the context is eg something like "Fear Conditioning"
        trial_type_key = ctx
        # get full name and abbreviation for trial
        trial_type_full = trial_type.trial_type_full  # default: 'Fear Conditioning'
        trial_type_abbr = trial_type.trial_type_abbr  # default: 'FC'

        return epochs, trial_type_key, trial_type_full, trial_type_abbr
    # handle case where no TrialType matches context
    except KeyError:
        print(f'Trial Control Setting for {ctx} not found!')
        print('Trial Control Settings known:')
        print(sheet_settings.trial_type_list)
        return False


def baseline_data_out(anim_id, anim, outpath, bin_secs, baseline_duration, freeze_thresh, dart_thresh, trial_type_abbr,
                      label=None):
    """
    Get baseline freezing data and output csv file.
    """
    # get baseline data
    # label defaults to 'Recording Time'
    if label is None:
        label = 'Recording time'
    # filter the dataframe to include only recording times less than or equal to the baseline duration
    # i.e. before baseline period ends
    # baseline_duration is from TrialSettingss -> default: 120
    baseline = anim[anim[label] <= baseline_duration]
    # call baseline function for df and times
    baseline_freezing, _, baseline_darting, _ = srf.get_baseline_freezing_darting(baseline,
                                                                                  freezing_threshold=freeze_thresh,
                                                                                  darting_threshold=dart_thresh,
                                                                                  bin_secs=bin_secs)
    # write to output path w/ animal id
    baseline_freezing_outfile = os.path.join(outpath, f'{trial_type_abbr}-baseline-freezing-{anim_id}.csv')
    baseline_freezing.to_csv(baseline_freezing_outfile)

    baseline_darting_outfile = os.path.join(outpath, f'{trial_type_abbr}-baseline-darting-{anim_id}.csv')
    baseline_darting.to_csv(baseline_darting_outfile)


# TODO: add analysis fns here
# for each epoch (check if full velocity or not to decide which analyses to run),
# write current to csv and add results to something that stores the overall results
# do the same for derived epochs
# how to avoid repetitive code? decorator issue: needing the outpath arg
# outpath: same except for end, except for that one specific maxvels file that is response
# can add directly to data frame rather than making list and then converting to dataframe
# (i.e. will only have one loop for epoch and one in it for derived and write combined df to csv at end)


def standard_analysis(delim_df_dict, label, ntones, freezing_threshold, darting_threshold, bin_secs):
    """
    Run standard analysis functions (max vels, freezing, darting).
    Return results.
    """
    # might need to change this (see note1 in output_analysis_files())
    # max_vels
    max_vels = srf.get_top_vels(delim_df_dict, 1, label, ntones)
    # freezing and darting -> make sure that epoch does not contain a space
    freezing_df, freezing_times, darting_df, darting_times = srf.get_freezing_darting(delim_df_dict,
                                                                                      ntones,
                                                                                      freezing_threshold,
                                                                                      darting_threshold,
                                                                                      label,
                                                                                      bin_secs)

    return [freezing_times, darting_times, max_vels, freezing_df, darting_df]


def add_tone_timebin_labels(times_df, counts_df):
    """
    Add tone epoch/subepoch labels from counts_df to corresponding rows in times_df. Return updated times_df.
    """
    # check if time df is empty; if yes, return it as is
    if times_df.empty:
        return times_df
    # init empty labels list
    labels = []
    # for each counts_df index value, add it to labels list n times, where n is the val in the first column for that row
    for idx in counts_df.index:
        labels += ([idx] * int(counts_df.loc[idx][0]))
    # add labels to times_df
    times_df['label'] = labels
    # return updated times_df
    return times_df


def collapse_time_bins(time_bin_df):
    # check if time df is empty; if yes, return it as is
    if time_bin_df.empty:
        return time_bin_df
    # get col1 and col2 from df as sets
    starts = set(time_bin_df['behavior start'])
    ends = set(time_bin_df['behavior end'])
    # get all items ony in one set, as sorted list
    times = sorted(list(starts.symmetric_difference(ends)))
    # all even idx items are col1, odd idx items are col2
    start_col = times[::2]
    end_col = times[1::2]
    # make new df w/ only start and end cols
    new_df = pd.DataFrame({'behavior start': start_col, 'behavior end': end_col})
    # join new df w/ start and labels cols from og df (left, but should produce same behavior as inner join would)
    new_df = pd.merge(new_df, time_bin_df[['behavior start', 'label']], on='behavior start', how='left')

    return new_df

    # init curr_start to first item from col1
    # for all items in col2, if not in col1:
    # set as curr_end
    # update curr_start to be val from col1 1 row down
    # delete in between rows


# tone labels note:
# Freezing (Time Bins) and Darts (count) cols
def convert_times_lists_to_dfs(standard_analysis_results):
    """
    Convert the freezing and darting times lists for df for given standard results list.
    Add label based on freezing and darting dfs.
    """

    # convert lists to dfs
    standard_analysis_results[0:2] = (pd.DataFrame(times, columns=['behavior start', 'behavior end'])
                                      for times in standard_analysis_results[0:2])

    print('pre-collapse', add_tone_timebin_labels(standard_analysis_results[0],
                                                  standard_analysis_results[3]))
    # add freezing labels and collapse
    standard_analysis_results[0] = collapse_time_bins(add_tone_timebin_labels(standard_analysis_results[0],
                                                                              standard_analysis_results[3]))
    # add darting labels and collapse
    standard_analysis_results[1] = collapse_time_bins(add_tone_timebin_labels(standard_analysis_results[1],
                                                                              standard_analysis_results[4]))

    return standard_analysis_results


def extended_analysis(delim_df_dict, label, epoch_count):
    """Run additional analysis functions (mean, median, and sem vels)."""
    mean_vels = srf.get_means(delim_df_dict, label, epoch_count)
    med_vels = srf.get_meds(delim_df_dict, label, epoch_count)
    sem_vels = srf.get_sems(delim_df_dict, label, epoch_count)

    return [mean_vels, med_vels, sem_vels]


def add_analyses_outputs(anim_id, base_outpath, suffixes, comb_dfs_list, analysis_dfs):
    """
    Write given analysis df to outpath and add it to the given list of list of dataframes
    that stores the combined data for all epochs/sub-epochs (later to be merged into single df)
    for each analysis provided.
    """
    # loop over all included analyses, outputting csvs and adding to appropriate list of df
    for suffix, comb_df, analysis_df in zip(suffixes, comb_dfs_list, analysis_dfs):
        outpath = f'{base_outpath}-{suffix}-{anim_id}.csv'
        # print(outpath, '\n')
        analysis_df.to_csv(outpath)
        comb_df.append(analysis_df)

    return comb_dfs_list


def run_analysis(anim, anim_id, outpath, prefix, epoch_label, epoch_count,
                 freezing_threshold, darting_threshold, bin_secs,
                 comb_dfs_list, suffixes, full_analysis,
                 trial_type_full=None, sub_epoch_timings=None, sub_epoch_labels=None, sub_epoch=None):
    """
    Run analysis (outputs analysis csvs) and return updated list of analysis dataframes (for combined dataframe output)
    for given epoch or sub epoch. For epoch, also produce output plots.
    """
    # check whether there is a sub-epoch; if not, epoch level analysis will be performed
    epoch_level = sub_epoch is None

    # get the dict of dfs for each tone in the epoch
    if epoch_level:
        delim_df_dict = srf.find_delim_segment(anim, epoch_count, epoch_label)
    else:
        delim_df_dict = srf.find_delim_based_time(anim, epoch_count, epoch_label, *sub_epoch_timings)

    label = epoch_label if epoch_level else f'{epoch_label}-{sub_epoch}'
    # run standard analysis
    analysis_results = standard_analysis(delim_df_dict, label, epoch_count, freezing_threshold, darting_threshold,
                                         bin_secs)
    # check whether full velocity output desired and add if yes
    if full_analysis:
        analysis_results += extended_analysis(delim_df_dict, label, epoch_count)

    if epoch_level:
        # output plots (using freezing and darting times, which are first two items in list)
        srf.plot_outputs(anim, anim_id, trial_type_full, outpath, prefix,
                         epoch_count, analysis_results[0], analysis_results[1],
                         epoch_label, sub_epoch_timings, sub_epoch_labels)
        # print("plot")

    # convert freezing/darting times to dfs (mutates original list)
    analysis_results = convert_times_lists_to_dfs(analysis_results)

    base_outpath = os.path.join(outpath, prefix)
    comb_dfs_list = add_analyses_outputs(anim_id, base_outpath, suffixes, comb_dfs_list,
                                         analysis_results)  # this will repeat for each sub epoch, except that the suffixes will be different (will include the sub_epoch, and will be shock_response for shock max vel)

    # return the updated combined data frames
    return comb_dfs_list


def analysis_files_for_epoch(anim, anim_id, outpath, prefix, trial_type_full, epoch,
                             freezing_threshold, darting_threshold, bin_secs,
                             comb_dfs_list, suffixes, full_analysis):
    """
    For given epoch, output epoch and subepoch analysis files (including plot),
    and return list of analysis dataframes.
    """

    # complete list of sub epoch labels and times for epoch (needed only for plot)
    all_sub_epoch_labels, all_sub_epoch_timings = epoch.get_sub_epoch_lists()

    # run analysis, update combined dfs, and output files
    comb_dfs_list = run_analysis(anim, anim_id, outpath, prefix, epoch.label, epoch.epoch_count,
                                 freezing_threshold, darting_threshold, bin_secs,
                                 comb_dfs_list, suffixes, full_analysis,
                                 trial_type_full=trial_type_full,
                                 sub_epoch_timings=all_sub_epoch_timings, sub_epoch_labels=all_sub_epoch_labels)
    # loop over sub-epochs
    for sub_epoch, sub_epoch_timings in epoch.get_sub_epochs_with_int_timings().items():
        # skip empties
        if sub_epoch == '':
            continue

        # add sub epoch name to suffixes
        sub_epoch_suffixes = [f'{sub_epoch}-{suffix}' for suffix in suffixes]
        # for the shock sub epoch, name suffix should be 'shock response'; replace
        if sub_epoch == 'Shock':
            sub_epoch_suffixes[2] = 'Shock-response'

        # run analysis, update combined dfs, and output files
        comb_dfs_list = run_analysis(anim, anim_id, outpath, prefix, epoch.label, epoch.epoch_count,
                                     freezing_threshold, darting_threshold, bin_secs,
                                     comb_dfs_list, sub_epoch_suffixes, full_analysis,
                                     sub_epoch_timings=sub_epoch_timings, sub_epoch=sub_epoch)

    # return updated combined dfs
    return comb_dfs_list


def comb_outputs(base_outpath, anim_id, suffixes, combined_dfs):
    """
    Write given combined analysis df to outpath with associated suffix.
    """
    # loop over all included analyses, outputting csvs
    for suffix, combined_df in zip(suffixes, combined_dfs):
        outpath = f'{base_outpath}-all-{suffix}-{anim_id}.csv'
        # print(outpath)
        # print(combined_df)
        combined_df.to_csv(outpath)


def all_epoch_analysis(anim, anim_id, outpath, trial_type_full, trial_type_abbr, epochs,
                       freeze_thresh, dart_thresh, bin_secs, full_analysis=False):
    """
    For given animal, generate epoch, subepoch, and combined output files.
    Epoch name is included iff there are multiple epochs.
    """

    # possibilities for future abstraction: take standard and additional suffixes as args, base n_analyses on length
    # list of suffixes -> note that the one the max-vel in the sub-epoch (specifically) must be "shock-response" -> move one function up? (to all_epochs_analysis)?
    suffixes = ['freezing-times', 'darting-times', 'max-vels', 'freezing', 'darting']

    # number of analysis outputs per analysis
    # (standard version is 5: freezing_times, darting_times, max_vels, freezing_df, darting_df)
    n_analyses = 5
    # add number of additional analyses that will be run in full version
    if full_analysis:
        n_analyses += 3
        suffixes += ['mean-vels', 'med-vels', 'SEM-vels']
    # initialize empty list of lists for combined dfs
    comb_dfs_list = [[] for _ in range(n_analyses)]

    # check whether multiple epochs -> need to do this to see whether to include epoch name
    if len(epochs) > 1:
        for epoch in epochs:
            # add epoch name to base-outpath for epoch -> need to strip epoch? check whether already stripped (should be)
            prefix = f'{trial_type_abbr}-{epoch}'
            # print(base_outpath)
            # run the analysis (above) -> might not need to reassign to list (since og is mutatated) but keep for clarity?
            comb_dfs_list = analysis_files_for_epoch(anim, anim_id, outpath, prefix, trial_type_full,
                                                     epoch, freeze_thresh, dart_thresh, bin_secs, comb_dfs_list,
                                                     suffixes, full_analysis)
    else:
        # print(outpath)
        # run analysis for single epoch (exclude epoch name from file names)
        comb_dfs_list = analysis_files_for_epoch(anim, anim_id, outpath, trial_type_abbr, trial_type_full, epochs[0],
                                                 freeze_thresh, dart_thresh, bin_secs, comb_dfs_list,
                                                 suffixes, full_analysis)

    # combined dfs
    # for each item in comb_dfs_list, do something list pd.concat(comb_df, axis=1) -> comb_df should be list of dfs
    # for full analysis, last 3 dfs combined along matching row idx (all others combined along matching columns)
    if full_analysis:
        combined_dfs = [pd.concat(comb_dfs, axis=0) for comb_dfs in comb_dfs_list[:-3]]
        combined_dfs += [pd.concat(comb_dfs, axis=1) for comb_dfs in comb_dfs_list[5:]]
    else:
        combined_dfs = [pd.concat(comb_dfs, axis=0) for comb_dfs in comb_dfs_list]

    base_outpath = os.path.join(outpath, trial_type_abbr)
    # output each combined df
    comb_outputs(base_outpath, anim_id, suffixes, combined_dfs)
