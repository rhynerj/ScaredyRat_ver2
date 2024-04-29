"""
file to run analysis for individual animals and output appropriate analysis files
can be run independently using main function (requires input and output folder parameters)
"""
import pandas as pd
import math
import os
import PySimpleGUI as sg

import src.sr_functions as srf
import src.sr_compiled as srco
import src.sr_settings as srs


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
    # TODO: make sure that returning list of Epoch actually works here

    # function: get_trial_info
    # this can probably be replaced by a simple if statement:
    # if ctx in detectionSettingsLabel: get the associated values
    # else: print the unknown messages (see below) return false
    # iterate over the labels in the detection settings (default is length 1)
    # for label in detection_settings_label:
    # if the context of the sheet matches the current iteration, get the trialTypeFull and the trialType for that context

    try:
        trial_type = sheet_settings.get_trial_type_by_label(ctx)
        # set the epoch label to be the key of the nested dict that is referred to by the current detection settings label
        # i.e., with the defaults, this returns ['Tone'] (have struct {'Fear Conditioning': {'Tone': {stuff}}})
        # has to be an easier way to do this
        epochs = epoch_settings[ctx]
        # the context is eg something like "Fear Conditioning"
        trial_type_key = ctx
        # these two are from the raw_sheet_settings variable
        trial_type_full = trial_type.trial_type_full  # default: 'Fear Conditioning'
        trial_type_abbr = trial_type.trial_type_abbr  # default: 'FC'
        # don't skip bc have valid values
        return epochs, trial_type_key, trial_type_full, trial_type_abbr
        # stop iterating, because only one match for ctx makes sense -> can probably filter instead
    # this is checking whether we are at the last item in detection settings, and it doesn't match ctx
    # should be replaced with a condition that executes if filtering/search doesn't return anything
    except KeyError:
        print(f'Trial Control Setting for {ctx} not found!')
        print('Trial Control Settings known:')
        print(sheet_settings.trial_type_list)
        return False


####

# TODO function: baseline_data_out
def baseline_data_out(anim_id, anim, outpath, bin_secs, baseline_duration, freeze_thresh, trial_type_abbr, label=None):
    """
    Get baseline freezing data and output csv file.
    """
    # get baseline data
    # filter the dataframe to include only recording times less than or equal to the baseline duration
    # baselineDuration is from raw_trialSettings -> default: 120
    if label is None:
        label = 'Recording time'
    baseline = anim[anim[label] <= baseline_duration]
    # call baseline function for df and times
    baseline_freezing, bfts = srf.get_baseline_freezing(baseline, freezing_threshold=freeze_thresh,
                                                        bin_secs=bin_secs)
    # write to output path w/ animal id
    # baseline_outfile = f'{outpath}/{trial_type_abbr}-baseline-freezing-{anim_id}.csv'
    baseline_outfile = os.path.join(outpath, f'{trial_type_abbr}-baseline-freezing-{anim_id}.csv')
    baseline_freezing.to_csv(baseline_outfile)
    ####


# # does the decorator actually help? -> delete?
# def write_analysis_to_csv(outfile):
#     """
#     Decorator to run analysis, write result to given csv file, and return result
#     """
#     def write_decorator(func):
#         @functools.wraps(func)
#         def write_wrapper(*args, **kwargs):
#             result = func(*args, **kwargs)
#             result.to_csv(outfile)
#             return result
#         return write_wrapper
#     return write_decorator

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
    """Add tone epoch/subepoch labels from counts_df to corresponding rows in times_df. Return updated times_df."""
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


# tone labels note:
# Freezing (Time Bins) and Darts (count) cols ->
# each level (epoch vs sub_epoch) has its own col, but that's probably only for final df, so should be able
# to just pull the appropriate value from col; still TODO: figure out how to combine things
def convert_times_lists_to_dfs(standard_analysis_results):
    """Convert the freezing and darting times lists for df for given standard results list.
    Add label based on freezing and darting dfs."""

    # convert lists to dfs
    standard_analysis_results[0:2] = (pd.DataFrame(times, columns=['bin start', 'bin end'])
                                      for times in standard_analysis_results[0:2])

    # add freezing labels
    standard_analysis_results[0] = add_tone_timebin_labels(standard_analysis_results[0],
                                                           standard_analysis_results[3])
    # add darting labels
    standard_analysis_results[1] = add_tone_timebin_labels(standard_analysis_results[1],
                                                           standard_analysis_results[4])

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

    # analysis_df.to_csv(outpath)
    # updated_df = pd.concat([all_df, analysis_df], axis=1)

    # return updated_df

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
    # Check whether there is a sub-epoch; if not, epoch level analysis will be performed
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


# done: update freezing darting times dfs to include tone number & epoch/subepoch label;
# retest analysis_files_for_epoch -> may rewrite combining of dfs
# reset add_analyses_outputs(), run_analysis(), analysis_files_for_epoch(), comb_dfs()
def analysis_files_for_epoch(anim, anim_id, outpath, prefix, trial_type_full, epoch,
                             freezing_threshold, darting_threshold, bin_secs,
                             comb_dfs_list, suffixes, full_analysis):
    """
    For given epoch, output epoch and subepoch analysis files (including plot), and return list of analysis dataframes.
    """
    # CONSTRUCTION NOTICE: moving sections of this to run_analysis -> rewrite accordingly
    # get the dict of dfs for each tone in the epoch
    # epc = srf.find_delim_segment(anim, ntones, epoch)
    # run standard analysis
    # analysis_results = standard_analysis(epc, epoch, ntones, freezing_threshold, darting_threshold, bin_secs)
    # done: move this list of suffixes to all_epoch_analysis
    # list of suffixes -> note that the one the max-vel in the sub-epoch (specifically) must be "shock-response" -> move one function up? (to all_epochs_analysis)?
    # suffixes = ['freezing-times', 'darting-times', 'max-vels', 'freezing', 'darting']
    # check whether full velocity output desired and add if yes
    # if full_analysis:
    #    analysis_results += extended_analysis()

    # output plots (using freezing and darting times, which are first two items in list)
    # srf.plot_outputs(anim, anim_id, trial_type_full, outpath, trial_type_abbr,
    #                  ntones, analysis_results[0], analysis_results[1], epoch, sub_epoch_timings, sub_epoch_labels)

    # convert freezing/darting times to dfs (mutates original list)
    # convert_times_lists_to_dfs(analysis_results)

    # comb_dfs_list = add_analyses_outputs(outpath, suffixes, comb_dfs_list, analysis_results)  # this will repeat for each sub epoch, except that the suffixes will be different (will include the sub_epoch, and will be shock_response for shock max vel)

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

    # done: GENERAL: 1. fn documentation, 2. test fns, 3. update run_SR (prior to this, write outline of all steps needed)

    # zip together list of names (for outpath), all_dfs, and analysis outputs (from helper fn) and call add_analysis_output() -> note1: move this somewhere else? need to make sure adding to overall df, but want to avoid duplication w/ sub-epochs

    # freezing and darting time data frames
    # freezing_time_df, darting_time_df = (pd.DataFrame(times, columns=['start', 'stop'])
    #                                     for times in (freezing_times, darting_times))

    # loop over sub-epochs
    # base-outpath for current epoch


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
    # done: move suffixes from analysis_files_for_epoch to here st they can also be used in the combined
    # add additional analyses w/ if

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

    # anim, anim_id, outpath, trial_type_full, trial_type_abbr, epoch,
    #                          freezing_threshold, darting_threshold, bin_secs,
    #                          comb_dfs_list, suffixes, full_analysis=False
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
    # ex of concat: pd.concat([all_df, analysis_df], axis=1);
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


# TODO: move this to SR_GUI.py
def run_SR(in_out_settings=srs.InOutSettings(), sheet_settings=srs.SheetSettings(), trial_settings=srs.TrialSettings(),
           epoch_settings=srs.EpochSettings()):
    # extract components of the raw_sheet_settings variable
    # sheetlist, detectionSettingsLabel, trialTypeFull_list, trialType_list, = parse_sheet_settings(raw_sheet_settings)
    # extract components of the raw_trial_settings variable
    # binSize, baselineDuration, freezeThresh, dartThresh = parse_trial_settings(raw_trial_settings)

    # input and output paths
    inpath = in_out_settings.inpath
    outpath = in_out_settings.ind_outpath
    full_analysis = in_out_settings.full_vel

    # function: get_in_file_list
    # search the inpath for files, make list
    # filelist = []
    # for entry in os.scandir(inpath):
    #     if entry.is_file() and not entry.name.startswith('~'):
    #         filelist.append(entry.path)

    # get list of imput files
    filelist = get_file_list(inpath)
    ####

    # set up layout for progress bar (only necessary when running as GUI)
    prog_layout = [[sg.ProgressBar(max_value=4 * len(filelist), orientation='horizontal', size=(20, 10), style='clam',
                                   key='SR_PROG')],
                   [sg.Cancel()]]
    # initialize sheets processed count to 0
    f_ct = 0
    # add progress window
    prog_win = sg.Window('ScaredyRat', prog_layout)
    prog_bar = prog_win['SR_PROG']
    ####

    # TODO function: ? -> might just keep here
    # iterate over list of files (from input folder)
    for file in filelist:
        # iterate over the sheets in the given sheetlist (note that these are manually provided, not automatically extracted
        for sheet in sheet_settings.sheet_list:
            # increment count by one (can be replaced by enum)
            f_ct += 1

            # TODO GUI function -> write progress_update function in sr_gui_setup.py:
            # get stuff from the progress bar -> not 100% sure what this is doing
            progevent, progvals = prog_win.read(timeout=1)
            # handle closing events
            if progevent == 'Cancel' or progevent == sg.WIN_CLOSED:
                break
            # update progress bar according to current count
            prog_bar.update(f_ct)
            ####

            # get data from current sheet
            anim_id, ctx, anim = srf.animal_read(inpath, file, sheet)
            print('\nEvaluating ' + sheet + ' in the file ' + file)
            # print(ctx)
            # print(ID)
            # skipFlag = False  # TBR probably

            # check the context, skipping if the file isn't an excel sheet or isn't properly labeled
            # don't need to be checking whether ID is float if casting to float -> can simplify things by casting to float and checking if nan and casting to int and checking if -1
            # if (ID == "-1" or ID == "nan" or (isinstance(ID, float) and math.isnan(float(ID))) or (
            #         isinstance(ID, int) and ID == -1)):
            if int(anim_id) == -1 or math.isnan(float(anim_id)):
                print('Animal Detection Failure: failed to load sheet or animal ID not found',
                      anim_id, '\n', ctx, sep='\n')
                # print(anim_id)
                # print('\n')
                # print(ctx)
                continue
            else:
                # get epochs, trial_type_key, trial_type_full, trial_type_abbr or false if error
                trial_info = get_trial_info(ctx, sheet_settings, epoch_settings)
                if trial_info:
                    # extract vars from results
                    epochs, trial_type_key, trial_type_full, trial_type_abbr = trial_info
                else:
                    continue
            #     # TODO function: get_trial_info
            #     # this can probably be replaced by a simple if statement:
            #     # if ctx in detectionSettingsLabel: get the associated values
            #     # else: print the unknown messages (see below) return false
            #     # iterate over the labels in the detection settings (default is length 1)
            #     for i in range(0, len(detectionSettingsLabel)):
            #         # if the context of the sheet matches the current iteration, get the trialTypeFull and the trialType for that context
            #         if (ctx == detectionSettingsLabel[i]):
            #             # set the epoch label to be the key of the nested dict that is referred to by the current detection settings label
            #             # i.e., with the defaults, this returns ['Tone'] (have struct {'Fear Conditioning': {'Tone': {stuff}}})
            #             # has to be an easier way to do this
            #             epochLabel = list(raw_epochSettings[detectionSettingsLabel[i]])
            #             # the context is eg something like "Fear Conditioning"
            #             trialType_key = ctx
            #             # these two are from the raw_sheet_settings variable
            #             trialTypeFull = trialTypeFull_list[i]  # default: 'Fear Conditioning'
            #             trialType = trialType_list[i]  # default: 'FC'
            #             # don't skip bc have valid values
            #             skipFlag = False
            #             # stop iterating, because only one match for ctx makes sense -> can probably filter instead
            #             break
            #         # this is checking whether we are at the last item in detection settings, and it doesn't match ctx
            #         # should be replaced with a condition that executes if filtering/search doesn't return anything
            #         elif (i >= len(detectionSettingsLabel) - 1 and ctx != detectionSettingsLabel[i]):
            #             print('Trial Control Settings (' + ctx + ' vs. ' + detectionSettingsLabel[
            #                 i] + ') not found! Ignoring ' + sheet + ' in ' + file)
            #             print(type(ctx))
            #             print(type(detectionSettingsLabel[0]))
            #             print('Trial Control Settings known:')
            #             print(detectionSettingsLabel)
            #             skipFlag = True
            #
            #     ####
            # # instead of skipFlag, this should check whether the get_trial_info function (above) returned False
            # if (skipFlag):
            #     continue

            # # TODO function: baseline_data_out
            # # get baseline data
            # baseline = {}  # TBR
            # # filter the dataframe to include only recording times less than or equal to the baseline duration
            # # baselineDuration is from raw_trialSettings -> default: 120
            # label = 'Recording time'
            # baseline = anim[anim[label] <= baselineDuration]
            # # call baselinne function for df and times
            # baselineFreezing, bFTs = srf.get_baseline_freezing(baseline, freezing_threshold=freezeThresh,
            #                                                    bin_secs=binSize)
            # # write to output path w/ animal id
            # BaselineOutfile = outpath + '/' + trial_type_abbr + '-baseline-freezing-{}.csv'
            # BaselineOutfile = BaselineOutfile.format(anim_id)
            # bFreezing = pd.concat([baselineFreezing],
            #                       axis=1)  # this doesn't actually do anything??? -> why would you concat df list of length 1??? maybe was originally longer list
            # bFreezing.to_csv(BaselineOutfile)
            # ####

            # output baseline data csv
            baseline_data_out(anim_id, anim, outpath, trial_settings.bin_secs,
                              trial_settings.baseline_duration, trial_settings.freeze_thresh,
                              trial_type_abbr)

            # output analysis files for given animal, for each epoch and subepoch
            all_epoch_analysis(anim, anim_id, outpath, trial_type_full, trial_type_abbr, epochs,
                               trial_settings.freeze_thresh, trial_settings.dart_thresh,
                               trial_settings.bin_secs, full_analysis)

            # TODO function: ?
            # function will need to be called twice: once for epochs, once for derived epochs

            # print(nEpochEvents)
            # if(nEpochEvents==0):
            #     continue
            # epoch_maxVel = []  # [0]*len(epochLabel)
            # epoch_meanVel = []  # [0]*len(epochLabel)
            # epoch_medVel = []  # [0]*len(epochLabel)
            # epoch_semVel = []  # [0]*len(epochLabel)
            # epochFreezing = []  # [0]*len(epochLabel)
            # epochFTs = []  # [0]*len(epochLabel)
            # epochDarting = []  # [0]*len(epochLabel)
            # epochDTs = []  # [0]*len(epochLabel)
            #
            # dEpoch_maxVel = []  # [[0]*len(derivedEpoch_list)]*len(epochLabel)
            # dEpoch_meanVel = []  # [[0]*len(derivedEpoch_list)]*len(epochLabel)
            # dEpoch_medVel = []  # [[0]*len(derivedEpoch_list)]*len(epochLabel)
            # dEpoch_semVel = []  # [[0]*len(derivedEpoch_list)]*len(epochLabel)
            # dEpochFreezing = []  # [[0]*len(derivedEpoch_list)]*len(epochLabel)
            # dEpochFTs = []  # [[0]*len(derivedEpoch_list)]*len(epochLabel)
            # dEpochDarting = []  # [[0]*len(derivedEpoch_list)]*len(epochLabel)
            # dEpochDTs = []  # [[0]*len(derivedEpoch_list)]*len(epochLabel)

            # find each epoch and derived epoch
            # for i in range(0, len(epoch_label)):  # is by default list of length 1: ['Tone']; don't need i (can do for each loop)
            # for epoch in epochs:
            #     # print(epochLabel[i])
            #
            #     # epoch_maxVel = [] #[0]*len(epochLabel)
            #     # epoch_meanVel = [] #[0]*len(epochLabel)
            #     # epoch_medVel = [] #[0]*len(epochLabel)
            #     # epoch_semVel = [] #[0]*len(epochLabel)
            #     # epochFreezing = [] #[0]*len(epochLabel)
            #     # epochFTs = [] #[0]*len(epochLabel)
            #     # epochDarting = [] #[0]*len(epochLabel)
            #     # epochDTs = [] #[0]*len(epochLabel)
            #
            #     # dEpoch_maxVel = [] #[[0]*len(derivedEpoch_list)]*len(epochLabel)
            #     # dEpoch_meanVel = [] #[[0]*len(derivedEpoch_list)]*len(epochLabel)
            #     # dEpoch_medVel = [] #[[0]*len(derivedEpoch_list)]*len(epochLabel)
            #     # dEpoch_semVel = [] #[[0]*len(derivedEpoch_list)]*len(epochLabel)
            #     # dEpochFreezing = [] #[[0]*len(derivedEpoch_list)]*len(epochLabel)
            #     # dEpochFTs = [] #[[0]*len(derivedEpoch_list)]*len(epochLabel)
            #     # dEpochDarting = [] #[[0]*len(derivedEpoch_list)]*len(epochLabel)
            #     # dEpochDTs = [] #[[0]*len(derivedEpoch_list)]*len(epochLabel)
            #     # derivedEpoch_list_list = list(raw_epochSettings[trialType_key][epochLabel[i]]['SubEpochs'])
            #
            #     # TODO function: get_sub_epoch_lists (in settings object)
            #     # (use get_sub_epoch_lists() from Epoch; have Epoch because this should be a for each loop)
            #     # # list of SubEpochs for current epoch -> write as fn in settings object; re-use to print settings?
            #     # derivedEpoch_list = []  # list of subepochs -> really just list(d['SubEpochs'].keys())
            #     # derivedEpochTiming_list = []  # list of lists with times for each dEpoch
            #     # for k, v in raw_epochSettings[trial_type_key][str(epoch_label[i]).strip()]['SubEpochs'].items():
            #     #     derivedEpoch_list.append(k)
            #     #     derivedEpochTiming_list.append(list(map(str.strip, v.split(','))))
            #     #     # derivedEpochTiming_list.append(list(map(str.strip, v)))
            #     #
            #     # ####
            #     sub_epoch_labels, sub_epoch_timings = epoch.get_sub_epoch_lists()
            #
            #     # TODO function: get_epoch_count (in settings object)
            #     # this is just getting epoch count from settings -> if object, won't need to cast to int
            #     # nEpochEvents = int(raw_epochSettings[trial_type_key][epochs[i]]['EpochCount'])
            #     ####
            #     epoch_count = epoch.epoch_count
            #
            #     if epoch_count == 0:  # skip
            #         continue
            #
            #     # TODO: delete, probably: most likely redundant bc get_label_df adds space and has try/catch
            #     # isSpace = raw_epochSettings[trial_type_key][epochs[i]]['UseSpace']
            #     # if isSpace:
            #     #     epochs[i] = epochs[i].strip() + ' '
            #     # else:
            #     #     epochs[i] = epochs[i].strip()
            #
            #     ####
            #
            #
            #     # print(epochLabel[i])
            #
            #     # TODO: function
            #     epc = srf.find_delim_segment(anim, epoch_count, epochs[i])
            #     # print('EPC')
            #     # print(epc)
            #     # Epoch max velocity
            #     epoch_maxVel.append(srf.get_top_vels(epc, 1, epochs[i], epoch_count))  #
            #     # print('EPC')
            #     # print(epc)
            #     # print(nEpochEvents)
            #     epoch_maxVel_file = outpath + '/' + trial_type_abbr + '-' + epochs[i].strip() + '-max-vels-{}.csv'
            #     epoch_maxVel_file = epoch_maxVel_file.format(anim_id)
            #     epoch_maxVel[i].to_csv(epoch_maxVel_file)
            #
            #     # Epoch Mean velocity
            #     epoch_meanVel.append(srf.get_means(epc, epochs[i], epoch_count))  #
            #     epoch_meanVel_file = outpath + '/' + trial_type_abbr + '-' + epochs[
            #         i].strip() + '-mean-vels-{}.csv'
            #     epoch_meanVel_file = epoch_meanVel_file.format(anim_id)
            #     epoch_meanVel[i].to_csv(epoch_meanVel_file)
            #
            #     # Epoch Median velocity
            #     epoch_medVel.append(srf.get_meds(epc, epochs[i], epoch_count))  #
            #     epoch_medVel_file = outpath + '/' + trial_type_abbr + '-' + epochs[i].strip() + '-med-vels-{}.csv'
            #     epoch_medVel_file = epoch_medVel_file.format(anim_id)
            #     epoch_medVel[i].to_csv(epoch_medVel_file)
            #
            #     # Epoch Velocity SEM
            #     epoch_semVel.append(srf.get_sems(epc, epochs[i], epoch_count))  #
            #     epoch_semVel_file = outpath + '/' + trial_type_abbr + '-' + epochs[i].strip() + '-SEM-vels-{}.csv'
            #     epoch_semVel_file = epoch_semVel_file.format(anim_id)
            #     epoch_semVel[i].to_csv(epoch_semVel_file)
            #     # print(type(epoch_semVel[i]))
            #
            #     # TODO: integrate new code below better
            #     # new:
            #     # get freezing and darting data using new function
            #     epochFreezing_TMP, epochFTs_TMP, epochDarting_TMP, epochDTs_TMP = srf.get_freezing_darting(epc,
            #                                                                                                epoch_count,
            #                                                                                                freezeThresh,
            #                                                                                                dartThresh,
            #                                                                                                epochs[
            #                                                                                                    i].strip(),
            #                                                                                                binSize)
            #     # end new
            #
            #     # Epoch freezing
            #     epoch_freezing_file = outpath + '/' + trial_type_abbr + '-' + epochs[
            #         i].strip() + '-freezing-{}.csv'
            #     epoch_freezing_file = epoch_freezing_file.format(anim_id)
            #     # epochFreezing_TMP, epochFTs_TMP = srf.get_freezing(epc,nEpochEvents,freezeThresh, epochLabel[i].strip(), binSize)
            #     epochFreezing.append(epochFreezing_TMP)
            #     epochFTs.append(epochFTs_TMP)
            #     epochFreezing_TMP.to_csv(epoch_freezing_file)
            #
            #     # Epoch Darting
            #     epoch_darting_file = outpath + '/' + trial_type_abbr + '-' + epochs[i].strip() + '-darting-{}.csv'
            #     epoch_darting_file = epoch_darting_file.format(anim_id)
            #     # epochDarting_TMP, epochDTs_TMP = srf.get_darting(epc,nEpochEvents,dartThresh, epochLabel[i].strip(), binSize)
            #     epochDarting.append(epochDarting_TMP)
            #     epochDTs.append(epochDTs_TMP)
            #     epochDarting_TMP.to_csv(epoch_darting_file)
            #
            #     # Epoch Plots
            #     # plotSettings = parse_plotSettings(derivedEpochTiming_list, derivedEpoch_list)
            #     srf.plot_outputs(anim, anim_id, trial_type_full, outpath, trial_type_abbr, epoch_count, epochFTs[i],
            #                      epochDTs[i],
            #                      epochs[i], sub_epoch_timings,
            #                      sub_epoch_labels)  # Needs to be updated to insert
            #
            #     # print(epochDTs[i])
            #
            #     dEpoch_maxVel.append([])
            #     dEpoch_meanVel.append([])
            #     dEpoch_medVel.append([])
            #     dEpoch_semVel.append([])
            #
            #     dEpochFreezing.append([])
            #     dEpochFTs.append([])
            #
            #     dEpochDarting.append([])
            #     dEpochDTs.append([])
            #
            #     for m in range(0, len(sub_epoch_labels)):
            #         if (sub_epoch_labels[m]) == '':
            #             continue;
            #
            #         # Get derived-epoch data frame
            #         dEpoch_df = srf.find_delim_based_time(anim, epoch_count, epochs[i],
            #                                               int(sub_epoch_timings[m][0]),
            #                                               int(sub_epoch_timings[m][1]),
            #                                               int(sub_epoch_timings[m][2]))
            #         # Derived-epoch max velocity
            #         # TODO: change epochLabel[i] to derivedEpoch_list[m] (or equivalent) to match others? Clarify intended column naming scheme
            #         # only include epoch label if there is more than one epoch
            #         dEpoch_maxVel[i].append(srf.get_top_vels(dEpoch_df, 1, epochs[i], epoch_count))  #
            #         dEpoch_maxVel_file = outpath + '/' + trial_type_abbr + '-' + epochs[i].strip() + '_' + \
            #                              sub_epoch_labels[m] + '-max-vels-{}.csv'
            #         dEpoch_maxVel_file = dEpoch_maxVel_file.format(anim_id)
            #         dEpoch_maxVel[i][m].to_csv(dEpoch_maxVel_file)
            #
            #         # print('derived epoch')
            #
            #         # Derived-epoch Mean velocity
            #         dEpoch_meanVel[i].append(srf.get_means(dEpoch_df, sub_epoch_labels[m], epoch_count))
            #         dEpoch_meanVel_file = outpath + '/' + trial_type_abbr + '-' + epochs[i].strip() + '_' + \
            #                               sub_epoch_labels[m] + '-mean-vels-{}.csv'
            #         dEpoch_meanVel_file = dEpoch_meanVel_file.format(anim_id)
            #         dEpoch_meanVel[i][m].to_csv(dEpoch_meanVel_file)
            #
            #         # Derived-epoch Median velocity
            #         dEpoch_medVel[i].append(srf.get_meds(dEpoch_df, sub_epoch_labels[m], epoch_count))  #
            #         dEpoch_medVel_file = outpath + '/' + trial_type_abbr + '-' + epochs[i].strip() + '_' + \
            #                              sub_epoch_labels[m] + '-med-vels-{}.csv'
            #         dEpoch_medVel_file = dEpoch_medVel_file.format(anim_id)
            #         dEpoch_medVel[i][m].to_csv(dEpoch_medVel_file)
            #
            #         # Derived-epoch Velocity SEM
            #         dEpoch_semVel[i].append(srf.get_sems(dEpoch_df, sub_epoch_labels[m], epoch_count))  #
            #         dEpoch_semVel_file = outpath + '/' + trial_type_abbr + '-' + epochs[i].strip() + '_' + \
            #                              sub_epoch_labels[m] + '-SEM-vels-{}.csv'
            #         dEpoch_semVel_file = dEpoch_semVel_file.format(anim_id)
            #         dEpoch_semVel[i][m].to_csv(dEpoch_semVel_file)
            #
            #         # TODO: integrate new code below better
            #         # new:
            #         # get freezing and darting data using new function
            #         dEpochFreezing_TMP, dEpochFTs_TMP, dEpochDarting_TMP, dEpochDTs_TMP = srf.get_freezing_darting(
            #             dEpoch_df, epoch_count, freezeThresh, dartThresh,
            #             epochs[i].strip() + '-' + sub_epoch_labels[m], binSize)
            #         # end new
            #
            #         # Derived-epoch freezing
            #         dEpoch_freezing_file = outpath + '/' + trial_type_abbr + '-' + epochs[i].strip() + '_' + \
            #                                sub_epoch_labels[m] + '-freezing-{}.csv'
            #         dEpoch_freezing_file = dEpoch_freezing_file.format(anim_id)
            #         # dEpochFreezing_TMP, dEpochFTs_TMP = srf.get_freezing(dEpoch_df,nEpochEvents,freezeThresh, epochLabel[i].strip() + '-' + derivedEpoch_list[m], binSize)
            #         dEpochFreezing[i].append(dEpochFreezing_TMP)
            #         dEpochFTs[i].append(dEpochFTs_TMP)
            #         dEpochFreezing_TMP.to_csv(dEpoch_freezing_file)
            #
            #         # Derived-epoch Darting
            #         dEpoch_darting_file = outpath + '/' + trial_type_abbr + '-' + epochs[i].strip() + '_' + \
            #                               sub_epoch_labels[m] + '-darting-{}.csv'
            #         dEpoch_darting_file = dEpoch_darting_file.format(anim_id)
            #         # dEpochDarting_TMP, dEpochDTs_TMP = srf.get_darting(dEpoch_df,nEpochEvents,dartThresh, epochLabel[i].strip() + '-' + derivedEpoch_list[m], binSize)
            #         dEpochDarting[i].append(dEpochDarting_TMP)
            #         dEpochDTs[i].append(dEpochDTs_TMP)
            #         dEpochDarting_TMP.to_csv(dEpoch_darting_file)
            #         # Derived-epoch Plots
            #         # srf.plot_outputs(anim, ID, trialTypeFull, outpath, trialType, nEpochEvents, dEpochFTs[i][m], dEpochDTs[i][m])  #Needs to be updated to insert
            #
            # # TODO function: trial_level_vels
            # #  : collapse all epoch and derived epoch dfs into a single df for each vel measure -> rows are tones (same across all, usually);
            # #  columns are max velocities for each epoch, then each sub-epoch, the problem is that there is nothing distinguishing the columns from another
            # # intialize df
            # allMaxes = pd.DataFrame()
            # # print("Length of epoch_maxVel: "+str(len(epoch_maxVel)))
            # # epoch_maxVel is list of things returned by get_top_vels for each epoch
            # for i in range(0, len(epoch_maxVel)):
            #     # print(len(epoch_maxVel[i]))
            #     # print(len(allMaxes))
            #     # print(allMaxes.shape)
            #     # print(epoch_maxVel[i].shape)
            #
            #     # if either the current vel df or the one with everything is shorter (row-wise), reindex according to longer one
            #     if (allMaxes.shape[0] < epoch_maxVel[i].shape[0]):
            #         allMaxes.reindex(epoch_maxVel[i].index)
            #     elif (allMaxes.shape[0] > epoch_maxVel[i].shape[0]):
            #         epoch_maxVel[i].reindex(allMaxes.index)
            #
            #     # add current epoch-level df to all maxes
            #     allMaxes = pd.concat([allMaxes, epoch_maxVel[i]], axis=1)
            #
            #     # add the max vels for each epoch
            #     for j in range(0, len(dEpoch_maxVel[i])):
            #         allMaxes = pd.concat([allMaxes, dEpoch_maxVel[i][j]], axis=1)
            #
            # maxOutFile = outpath + '/' + trial_type_abbr + '-max-vels-{}.csv'
            # maxOutFile = maxOutFile.format(anim_id)
            # allMaxes.to_csv(maxOutFile)
            # # Concatinate the full mean file per-animal
            # # print(dEpoch_meanVel[0][0])
            # # print('m=1')
            # # print(dEpoch_meanVel[0][1])
            # # print('m=2')
            # # print(dEpoch_meanVel[0][2])
            #
            # # similar to above, but avoids re-indexing (re-indexing is probably unnecessary?)
            # allMeans = pd.DataFrame()
            # for i in range(0, len(epoch_meanVel)):
            #     allMeans = pd.concat([allMeans, epoch_meanVel[i]], axis=1)
            #     for j in range(0, len(dEpoch_meanVel[i])):
            #         allMeans = pd.concat([allMeans, dEpoch_meanVel[i][j]], axis=1)
            #
            # meanOutFile = outpath + '/' + trial_type_abbr + '-mean-vels-{}.csv'
            # meanOutFile = meanOutFile.format(anim_id)
            # allMeans.to_csv(meanOutFile)
            #
            # # Concatinate the full median file per-animal
            # # allMedians = pd.concat([epoch_medVel, dEpoch_medVel],axis=1)
            # allMedians = pd.DataFrame()
            # for i in range(0, len(epoch_medVel)):
            #     allMedians = pd.concat([allMedians, epoch_medVel[i]], axis=1)
            #     for j in range(0, len(dEpoch_medVel[i])):
            #         allMedians = pd.concat([allMedians, dEpoch_medVel[i][j]], axis=1)
            # medOutFile = outpath + '/' + trial_type_abbr + '-median-vels-{}.csv'
            # medOutFile = medOutFile.format(anim_id)
            # allMedians.to_csv(medOutFile)
            #
            # # Concatinate the full SEM file per-animal
            # # allsem = pd.concat([epoch_semVel, dEpoch_semVel],axis=1)
            # allsem = pd.DataFrame()
            # for i in range(0, len(epoch_semVel)):
            #     allsem = pd.concat([allsem, epoch_semVel[i]], axis=1)
            #     for j in range(0, len(dEpoch_semVel[i])):
            #         allsem = pd.concat([allsem, dEpoch_semVel[i][j]], axis=1)
            # semOutFile = outpath + '/' + trial_type_abbr + '-SEM-vels-{}.csv'
            # semOutFile = semOutFile.format(anim_id)
            # allsem.to_csv(semOutFile)
            #
            # # Concatinate the full freezing file per-animal
            # # allFreeze = pd.concat([epochFreezing, dEpochFreezing],axis=1)
            # allFreeze = pd.DataFrame()
            # for i in range(0, len(epochFreezing)):
            #     allFreeze = pd.concat([allFreeze, epochFreezing[i]], axis=1)
            #     for j in range(0, len(dEpochFreezing[i])):
            #         allFreeze = pd.concat([allFreeze, dEpochFreezing[i][j]], axis=1)
            # freezeOutFile = outpath + '/' + trial_type_abbr + '-Freezing-{}.csv'
            # freezeOutFile = freezeOutFile.format(anim_id)
            # allFreeze.to_csv(freezeOutFile)
            #
            # # Concatinate the full darting file per-animal
            # # allDart = pd.concat([epochDarting, dEpochDarting],axis=1)
            # allDart = pd.DataFrame()
            # for i in range(0, len(epochDarting)):
            #     allDart = pd.concat([allDart, epochDarting[i]], axis=1)
            #     for j in range(0, len(dEpochDarting[i])):
            #         allDart = pd.concat([allDart, dEpochDarting[i][j]], axis=1)
            # dartOutFile = outpath + '/' + trial_type_abbr + '-Darting-{}.csv'
            # dartOutFile = dartOutFile.format(anim_id)
            # allDart.to_csv(dartOutFile)
            #
            # ####
            # # (the above will end up being one function called for each vel measurement)
            # ####

    # compiled_output function
    srco.compiled_output(in_out_settings, sheet_settings, epoch_settings)

    # iterate over trial_type_list (from SheetSettings object)

    # compile baseline (getting csvs from ind outpath, ouputting to comb outpath)

    # compile for each epoch and each subepoch -> need to modify compileSR to toggle full_analysis T/F
    # -> also, remove 'num-d_epoch' and 'behavior' arguments (not used)
    # ran twice in original, once with and once without valid d_epoch_list, but logically that would just cause some files to be written twice (not noticeable in output bc overwrites)
    # for k in range(0,
    #                len(trialType_list)):  # Should produce darting and freezing files for each trial type x epoch x sub-epoch
    #
    #     # this runs does compilation specifically for baseline measurements (i.e. not per epoch)
    #     srf.compile_baseline_sr(trialType_list[k], outpath, outpath2)
    #
    #     # detectionSettingsLabel is ['Fear Conditioning'] by default
    #     # raw_epochSettings[detectionSettingsLabel[k]] is a dict of dicts {'Tone': etc} by default
    #     for epoch_iter in raw_epochSettings[detectionSettingsLabel[k]]:
    #         epoch_ct = int(raw_epochSettings[detectionSettingsLabel[k]][epoch_iter]['EpochCount'])
    #         if (epoch_ct == 0):
    #             continue
    #         # srf.compile_SR(trialType_list[k], epoch_iter, epoch_ct, 1, [''],'Darting',outpath,outpath2)
    #
    #         # TODO: figure out why this is being called twice? -> different args, but why? also, get rid of unnecessary args
    #
    #         # run the compile function without derived epochs
    #         srf.compile_SR(trialType_list[k], epoch_iter, epoch_ct, [''], outpath, outpath2)
    #         # srf.compile_SR(trialType_list[k], epoch_iter, epoch_ct, 1, list(raw_epochSettings[detectionSettingsLabel[k]][epoch_iter]['SubEpochs'].keys()),'Darting',outpath,outpath2)
    #
    #         # run the compile function with derived epochs
    #         srf.compile_SR(trialType_list[k], epoch_iter, epoch_ct,
    #                        list(raw_epochSettings[detectionSettingsLabel[k]][epoch_iter]['SubEpochs'].keys()), outpath,
    #                        outpath2)

    # for dEpoch_iter in raw_epochSettings[detectionSettingsLabel[k]][epoch_iter]['SubEpochs']:
    #     srf.compile_SR(trialType_list[k], epoch_iter, epoch_ct, 1, [dEpoch_iter],'Darting',outpath,outpath2)
    #     srf.compile_SR(trialType_list[k], epoch_iter, epoch_ct, 1, [dEpoch_iter],'Freezing',outpath,outpath2)
    prog_win.close()

# TODO: 1. reset fns (see above) (done), 2. test them w/ valid input st they produce useful output (done),
#  3. send test output for review/feedback; next: move on to integration step

# TODO: GENERAL
# 1. reformat GUI stuff -> main function with setup stuff; all more complicated stuff factored out into helper fns
# !!!(make sure to integrate new settings objects)
# 1.1. test
# 2. reorganize functions into files; add any missing documentation
# 2.1. test
# 3. remove unused functions
# 3.1. test
# 4. remove commented out code
# 4.1. test


# TODO: IP: break GUI stuff into fns (simplify logic where possible), add in Settings objects (nearly done)
# next: test, then reorganize fns and test again
