"""
file to run analysis for individual animals and output appropriate analysis files
can be run independently using main function (requires input and output folder parameters)
"""
import functools
import pandas as pd
import math
import os

import src.sr_functions as srf
from src.sr_settings import SheetSettings, EpochSettings, TrialSettings, InOutSettings


def get_file_list(inpath):
    """Return list of all files in dir at given inpath."""
    # search the inpath for files, make list
    filelist = []
    for entry in os.scandir(inpath):
        if entry.is_file() and not entry.name.startswith('~'):
            filelist.append(entry.path)

    return filelist


def get_trial_info(ctx, sheet_settings, epoch_settings):
    # TODO function: get_trial_info
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
        epochs = epoch_settings.get_epoch_by_label(ctx)
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
def baseline_data_out(anim_id, anim, outpath, bin_secs, baseline_duration, freeze_thresh, trial_type_abbr):
    # get baseline data
    # filter the dataframe to include only recording times less than or equal to the baseline duration
    # baselineDuration is from raw_trialSettings -> default: 120
    label = 'Recording time'
    baseline = anim[anim[label] <= baseline_duration]
    # call baseline function for df and times
    baseline_freezing, bfts = srf.get_baseline_freezing(baseline, freezing_threshold=freeze_thresh,
                                                        bin_secs=bin_secs)
    # write to output path w/ animal id
    baseline_outfile = f'{outpath}/{trial_type_abbr}-baseline-freezing-{anim_id}.csv'
    baseline_freezing.to_csv(baseline_outfile)
    ####


# does the decorator actually help?
def write_analysis_to_csv(outfile):
    """
    Decorator to run analysis, write result to given csv file, and return result
    """
    def write_decorator(func):
        @functools.wraps(func)
        def write_wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            result.to_csv(outfile)
            return result
        return write_wrapper
    return write_decorator

# TODO: add analysis fns here


def run_SR(in_out_settings=InOutSettings(), sheet_settings=SheetSettings(), trial_settings=TrialSettings(),
           epoch_settings=EpochSettings()):
    # extract components of the raw_sheet_settings variable
    # sheetlist, detectionSettingsLabel, trialTypeFull_list, trialType_list, = parse_sheet_settings(raw_sheet_settings)
    # extract components of the raw_trial_settings variable
    # binSize, baselineDuration, freezeThresh, dartThresh = parse_trial_settings(raw_trial_settings)

    # input and output paths
    inpath = in_out_settings.inpath
    outpath = in_out_settings.ind_outpath

    # TODO function: get_in_file_list
    # search the inpath for files, make list
    # filelist = []
    # for entry in os.scandir(inpath):
    #     if entry.is_file() and not entry.name.startswith('~'):
    #         filelist.append(entry.path)
    filelist = get_file_list(inpath)
    ####

    # TODO GUI function -> write file_progress function in sr_gui_setup.py:
    # set up layout for progress bar (only necessary when running as GUI)
    prog_layout = [[sg.ProgressBar(max_value=4 * len(filelist), orientation='horizontal', size=(20, 10), style='clam',
                                   key='SR_PROG')],
                   [sg.Cancel()]]
    # initialize sheet count to 0
    f_ct = 0
    # add progress window
    prog_win = sg.Window('ScaredyRat', prog_layout)
    prog_bar = prog_win['SR_PROG']
    ####

    # TODO function: ?
    # iterate over files (all files in input folder)
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

            # set input/output info
            anim_id, ctx, anim = srf.animal_read(inpath, file, sheet)
            print('\nEvaluating ' + sheet + ' in the file ' + file)
            # print(ctx)
            # print(ID)
            # skipFlag = False  # TBR probably

            # check the context, skipping if the file isn't an excel sheet or properly labeled
            # don't need to be checking whether ID is float if casting to float -> can simplify things by casting to float and checking if nan and casting to int and checking if -1
            # if (ID == "-1" or ID == "nan" or (isinstance(ID, float) and math.isnan(float(ID))) or (
            #         isinstance(ID, int) and ID == -1)):
            if int(anim_id) == -1 or math.isnan(float(anim_id)):
                print('Animal Detection Failure: failed to load sheet or animal ID not found')
                print(anim_id)
                print('\n')
                print(ctx)
                continue
            else:
                trial_info = get_trial_info(ctx, sheet_settings, epoch_settings)
                if trial_info:
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

            baseline_data_out(anim_id, anim, outpath, trial_settings.bin_secs,
                              trial_settings.baseline_duration, trial_settings.freeze_thresh,
                              trial_type_abbr)

            # TODO function: ?
            # function will need to be called twice: once for epochs, once for derived epochs

            # print(nEpochEvents)
            # if(nEpochEvents==0):
            #     continue
            epoch_maxVel = []  # [0]*len(epochLabel)
            epoch_meanVel = []  # [0]*len(epochLabel)
            epoch_medVel = []  # [0]*len(epochLabel)
            epoch_semVel = []  # [0]*len(epochLabel)
            epochFreezing = []  # [0]*len(epochLabel)
            epochFTs = []  # [0]*len(epochLabel)
            epochDarting = []  # [0]*len(epochLabel)
            epochDTs = []  # [0]*len(epochLabel)

            dEpoch_maxVel = []  # [[0]*len(derivedEpoch_list)]*len(epochLabel)
            dEpoch_meanVel = []  # [[0]*len(derivedEpoch_list)]*len(epochLabel)
            dEpoch_medVel = []  # [[0]*len(derivedEpoch_list)]*len(epochLabel)
            dEpoch_semVel = []  # [[0]*len(derivedEpoch_list)]*len(epochLabel)
            dEpochFreezing = []  # [[0]*len(derivedEpoch_list)]*len(epochLabel)
            dEpochFTs = []  # [[0]*len(derivedEpoch_list)]*len(epochLabel)
            dEpochDarting = []  # [[0]*len(derivedEpoch_list)]*len(epochLabel)
            dEpochDTs = []  # [[0]*len(derivedEpoch_list)]*len(epochLabel)

            # find each epoch and derived epoch
            # for i in range(0, len(epoch_label)):  # is by default list of length 1: ['Tone']; don't need i (can do for each loop)
            for epoch in epochs:
                # print(epochLabel[i])

                # epoch_maxVel = [] #[0]*len(epochLabel)
                # epoch_meanVel = [] #[0]*len(epochLabel)
                # epoch_medVel = [] #[0]*len(epochLabel)
                # epoch_semVel = [] #[0]*len(epochLabel)
                # epochFreezing = [] #[0]*len(epochLabel)
                # epochFTs = [] #[0]*len(epochLabel)
                # epochDarting = [] #[0]*len(epochLabel)
                # epochDTs = [] #[0]*len(epochLabel)

                # dEpoch_maxVel = [] #[[0]*len(derivedEpoch_list)]*len(epochLabel)
                # dEpoch_meanVel = [] #[[0]*len(derivedEpoch_list)]*len(epochLabel)
                # dEpoch_medVel = [] #[[0]*len(derivedEpoch_list)]*len(epochLabel)
                # dEpoch_semVel = [] #[[0]*len(derivedEpoch_list)]*len(epochLabel)
                # dEpochFreezing = [] #[[0]*len(derivedEpoch_list)]*len(epochLabel)
                # dEpochFTs = [] #[[0]*len(derivedEpoch_list)]*len(epochLabel)
                # dEpochDarting = [] #[[0]*len(derivedEpoch_list)]*len(epochLabel)
                # dEpochDTs = [] #[[0]*len(derivedEpoch_list)]*len(epochLabel)
                # derivedEpoch_list_list = list(raw_epochSettings[trialType_key][epochLabel[i]]['SubEpochs'])

                # TODO function: get_sub_epoch_lists (in settings object)
                # (use get_sub_epoch_lists() from Epoch; have Epoch because this should be a for each loop)
                # # list of SubEpochs for current epoch -> write as fn in settings object; re-use to print settings?
                # derivedEpoch_list = []  # list of subepochs -> really just list(d['SubEpochs'].keys())
                # derivedEpochTiming_list = []  # list of lists with times for each dEpoch
                # for k, v in raw_epochSettings[trial_type_key][str(epoch_label[i]).strip()]['SubEpochs'].items():
                #     derivedEpoch_list.append(k)
                #     derivedEpochTiming_list.append(list(map(str.strip, v.split(','))))
                #     # derivedEpochTiming_list.append(list(map(str.strip, v)))
                #
                # ####
                sub_epoch_labels, sub_epoch_timings = epoch.get_sub_epoch_lists()

                # TODO function: get_epoch_count (in settings object)
                # this is just getting epoch count from settings -> if object, won't need to cast to int
                # nEpochEvents = int(raw_epochSettings[trial_type_key][epochs[i]]['EpochCount'])
                ####
                epoch_count = epoch.epoch_count

                if epoch_count == 0:  # skip
                    continue

                # TODO: delete, probably: most likely redundant bc get_label_df adds space and has try/catch
                # isSpace = raw_epochSettings[trial_type_key][epochs[i]]['UseSpace']
                # if isSpace:
                #     epochs[i] = epochs[i].strip() + ' '
                # else:
                #     epochs[i] = epochs[i].strip()

                ####

                # BOOKMARK

                # print(epochLabel[i])

                # TODO: function
                epc = srf.find_delim_segment(anim, epoch_count, epochs[i])
                # print('EPC')
                # print(epc)
                # Epoch max velocity
                epoch_maxVel.append(srf.get_top_vels(epc, 1, epochs[i], epoch_count))  #
                # print('EPC')
                # print(epc)
                # print(nEpochEvents)
                epoch_maxVel_file = outpath + '/' + trial_type_abbr + '-' + epochs[i].strip() + '-max-vels-{}.csv'
                epoch_maxVel_file = epoch_maxVel_file.format(anim_id)
                epoch_maxVel[i].to_csv(epoch_maxVel_file)

                # Epoch Mean velocity
                epoch_meanVel.append(srf.get_means(epc, epochs[i], epoch_count))  #
                epoch_meanVel_file = outpath + '/' + trial_type_abbr + '-' + epochs[
                    i].strip() + '-mean-vels-{}.csv'
                epoch_meanVel_file = epoch_meanVel_file.format(anim_id)
                epoch_meanVel[i].to_csv(epoch_meanVel_file)

                # Epoch Median velocity
                epoch_medVel.append(srf.get_meds(epc, epochs[i], epoch_count))  #
                epoch_medVel_file = outpath + '/' + trial_type_abbr + '-' + epochs[i].strip() + '-med-vels-{}.csv'
                epoch_medVel_file = epoch_medVel_file.format(anim_id)
                epoch_medVel[i].to_csv(epoch_medVel_file)

                # Epoch Velocity SEM
                epoch_semVel.append(srf.get_sems(epc, epochs[i], epoch_count))  #
                epoch_semVel_file = outpath + '/' + trial_type_abbr + '-' + epochs[i].strip() + '-SEM-vels-{}.csv'
                epoch_semVel_file = epoch_semVel_file.format(anim_id)
                epoch_semVel[i].to_csv(epoch_semVel_file)
                # print(type(epoch_semVel[i]))

                # TODO: integrate new code below better
                # new:
                # get freezing and darting data using new function
                epochFreezing_TMP, epochFTs_TMP, epochDarting_TMP, epochDTs_TMP = srf.get_freezing_darting(epc,
                                                                                                           epoch_count,
                                                                                                           freezeThresh,
                                                                                                           dartThresh,
                                                                                                           epochs[
                                                                                                               i].strip(),
                                                                                                           binSize)
                # end new

                # Epoch freezing
                epoch_freezing_file = outpath + '/' + trial_type_abbr + '-' + epochs[
                    i].strip() + '-freezing-{}.csv'
                epoch_freezing_file = epoch_freezing_file.format(anim_id)
                # epochFreezing_TMP, epochFTs_TMP = srf.get_freezing(epc,nEpochEvents,freezeThresh, epochLabel[i].strip(), binSize)
                epochFreezing.append(epochFreezing_TMP)
                epochFTs.append(epochFTs_TMP)
                epochFreezing_TMP.to_csv(epoch_freezing_file)

                # Epoch Darting
                epoch_darting_file = outpath + '/' + trial_type_abbr + '-' + epochs[i].strip() + '-darting-{}.csv'
                epoch_darting_file = epoch_darting_file.format(anim_id)
                # epochDarting_TMP, epochDTs_TMP = srf.get_darting(epc,nEpochEvents,dartThresh, epochLabel[i].strip(), binSize)
                epochDarting.append(epochDarting_TMP)
                epochDTs.append(epochDTs_TMP)
                epochDarting_TMP.to_csv(epoch_darting_file)

                # Epoch Plots
                # plotSettings = parse_plotSettings(derivedEpochTiming_list, derivedEpoch_list)
                srf.plot_outputs(anim, anim_id, trial_type_full, outpath, trial_type_abbr, epoch_count, epochFTs[i],
                                 epochDTs[i],
                                 epochs[i], sub_epoch_timings,
                                 sub_epoch_labels)  # Needs to be updated to insert

                # print(epochDTs[i])

                dEpoch_maxVel.append([])
                dEpoch_meanVel.append([])
                dEpoch_medVel.append([])
                dEpoch_semVel.append([])

                dEpochFreezing.append([])
                dEpochFTs.append([])

                dEpochDarting.append([])
                dEpochDTs.append([])

                for m in range(0, len(sub_epoch_labels)):
                    if (sub_epoch_labels[m]) == '':
                        continue;

                    # Get derived-epoch data frame
                    dEpoch_df = srf.find_delim_based_time(anim, epoch_count, epochs[i],
                                                          int(sub_epoch_timings[m][0]),
                                                          int(sub_epoch_timings[m][1]),
                                                          int(sub_epoch_timings[m][2]))
                    # Derived-epoch max velocity
                    # TODO: change epochLabel[i] to derivedEpoch_list[m] (or equivalent) to match others? Clarify intended column naming scheme
                    # only include epoch label if there is more than one epoch
                    dEpoch_maxVel[i].append(srf.get_top_vels(dEpoch_df, 1, epochs[i], epoch_count))  #
                    dEpoch_maxVel_file = outpath + '/' + trial_type_abbr + '-' + epochs[i].strip() + '_' + \
                                         sub_epoch_labels[m] + '-max-vels-{}.csv'
                    dEpoch_maxVel_file = dEpoch_maxVel_file.format(anim_id)
                    dEpoch_maxVel[i][m].to_csv(dEpoch_maxVel_file)

                    # print('derived epoch')

                    # Derived-epoch Mean velocity
                    dEpoch_meanVel[i].append(srf.get_means(dEpoch_df, sub_epoch_labels[m], epoch_count))
                    dEpoch_meanVel_file = outpath + '/' + trial_type_abbr + '-' + epochs[i].strip() + '_' + \
                                          sub_epoch_labels[m] + '-mean-vels-{}.csv'
                    dEpoch_meanVel_file = dEpoch_meanVel_file.format(anim_id)
                    dEpoch_meanVel[i][m].to_csv(dEpoch_meanVel_file)

                    # Derived-epoch Median velocity
                    dEpoch_medVel[i].append(srf.get_meds(dEpoch_df, sub_epoch_labels[m], epoch_count))  #
                    dEpoch_medVel_file = outpath + '/' + trial_type_abbr + '-' + epochs[i].strip() + '_' + \
                                         sub_epoch_labels[m] + '-med-vels-{}.csv'
                    dEpoch_medVel_file = dEpoch_medVel_file.format(anim_id)
                    dEpoch_medVel[i][m].to_csv(dEpoch_medVel_file)

                    # Derived-epoch Velocity SEM
                    dEpoch_semVel[i].append(srf.get_sems(dEpoch_df, sub_epoch_labels[m], epoch_count))  #
                    dEpoch_semVel_file = outpath + '/' + trial_type_abbr + '-' + epochs[i].strip() + '_' + \
                                         sub_epoch_labels[m] + '-SEM-vels-{}.csv'
                    dEpoch_semVel_file = dEpoch_semVel_file.format(anim_id)
                    dEpoch_semVel[i][m].to_csv(dEpoch_semVel_file)

                    # TODO: integrate new code below better
                    # new:
                    # get freezing and darting data using new function
                    dEpochFreezing_TMP, dEpochFTs_TMP, dEpochDarting_TMP, dEpochDTs_TMP = srf.get_freezing_darting(
                        dEpoch_df, epoch_count, freezeThresh, dartThresh,
                        epochs[i].strip() + '-' + sub_epoch_labels[m], binSize)
                    # end new

                    # Derived-epoch freezing
                    dEpoch_freezing_file = outpath + '/' + trial_type_abbr + '-' + epochs[i].strip() + '_' + \
                                           sub_epoch_labels[m] + '-freezing-{}.csv'
                    dEpoch_freezing_file = dEpoch_freezing_file.format(anim_id)
                    # dEpochFreezing_TMP, dEpochFTs_TMP = srf.get_freezing(dEpoch_df,nEpochEvents,freezeThresh, epochLabel[i].strip() + '-' + derivedEpoch_list[m], binSize)
                    dEpochFreezing[i].append(dEpochFreezing_TMP)
                    dEpochFTs[i].append(dEpochFTs_TMP)
                    dEpochFreezing_TMP.to_csv(dEpoch_freezing_file)

                    # Derived-epoch Darting
                    dEpoch_darting_file = outpath + '/' + trial_type_abbr + '-' + epochs[i].strip() + '_' + \
                                          sub_epoch_labels[m] + '-darting-{}.csv'
                    dEpoch_darting_file = dEpoch_darting_file.format(anim_id)
                    # dEpochDarting_TMP, dEpochDTs_TMP = srf.get_darting(dEpoch_df,nEpochEvents,dartThresh, epochLabel[i].strip() + '-' + derivedEpoch_list[m], binSize)
                    dEpochDarting[i].append(dEpochDarting_TMP)
                    dEpochDTs[i].append(dEpochDTs_TMP)
                    dEpochDarting_TMP.to_csv(dEpoch_darting_file)
                    # Derived-epoch Plots
                    # srf.plot_outputs(anim, ID, trialTypeFull, outpath, trialType, nEpochEvents, dEpochFTs[i][m], dEpochDTs[i][m])  #Needs to be updated to insert

            # TODO function: trial_level_vels
            #  : collapse all epoch and derived epoch dfs into a single df for each vel measure -> rows are tones (same across all, usually);
            #  columns are max velocities for each epoch, then each sub-epoch, the problem is that there is nothing distinguishing the columns from another
            # intialize df
            allMaxes = pd.DataFrame()
            # print("Length of epoch_maxVel: "+str(len(epoch_maxVel)))
            # epoch_maxVel is list of things returned by get_top_vels for each epoch
            for i in range(0, len(epoch_maxVel)):
                # print(len(epoch_maxVel[i]))
                # print(len(allMaxes))
                # print(allMaxes.shape)
                # print(epoch_maxVel[i].shape)

                # if either the current vel df or the one with everything is shorter (row-wise), reindex according to longer one
                if (allMaxes.shape[0] < epoch_maxVel[i].shape[0]):
                    allMaxes.reindex(epoch_maxVel[i].index)
                elif (allMaxes.shape[0] > epoch_maxVel[i].shape[0]):
                    epoch_maxVel[i].reindex(allMaxes.index)

                # add current epoch-level df to all maxes
                allMaxes = pd.concat([allMaxes, epoch_maxVel[i]], axis=1)

                # add the max vels for each epoch
                for j in range(0, len(dEpoch_maxVel[i])):
                    allMaxes = pd.concat([allMaxes, dEpoch_maxVel[i][j]], axis=1)

            maxOutFile = outpath + '/' + trial_type_abbr + '-max-vels-{}.csv'
            maxOutFile = maxOutFile.format(anim_id)
            allMaxes.to_csv(maxOutFile)
            # Concatinate the full mean file per-animal
            # print(dEpoch_meanVel[0][0])
            # print('m=1')
            # print(dEpoch_meanVel[0][1])
            # print('m=2')
            # print(dEpoch_meanVel[0][2])

            # similar to above, but avoids re-indexing (re-indexing is probably unnecessary?)
            allMeans = pd.DataFrame()
            for i in range(0, len(epoch_meanVel)):
                allMeans = pd.concat([allMeans, epoch_meanVel[i]], axis=1)
                for j in range(0, len(dEpoch_meanVel[i])):
                    allMeans = pd.concat([allMeans, dEpoch_meanVel[i][j]], axis=1)

            meanOutFile = outpath + '/' + trial_type_abbr + '-mean-vels-{}.csv'
            meanOutFile = meanOutFile.format(anim_id)
            allMeans.to_csv(meanOutFile)

            # Concatinate the full median file per-animal
            # allMedians = pd.concat([epoch_medVel, dEpoch_medVel],axis=1)
            allMedians = pd.DataFrame()
            for i in range(0, len(epoch_medVel)):
                allMedians = pd.concat([allMedians, epoch_medVel[i]], axis=1)
                for j in range(0, len(dEpoch_medVel[i])):
                    allMedians = pd.concat([allMedians, dEpoch_medVel[i][j]], axis=1)
            medOutFile = outpath + '/' + trial_type_abbr + '-median-vels-{}.csv'
            medOutFile = medOutFile.format(anim_id)
            allMedians.to_csv(medOutFile)

            # Concatinate the full SEM file per-animal
            # allsem = pd.concat([epoch_semVel, dEpoch_semVel],axis=1)
            allsem = pd.DataFrame()
            for i in range(0, len(epoch_semVel)):
                allsem = pd.concat([allsem, epoch_semVel[i]], axis=1)
                for j in range(0, len(dEpoch_semVel[i])):
                    allsem = pd.concat([allsem, dEpoch_semVel[i][j]], axis=1)
            semOutFile = outpath + '/' + trial_type_abbr + '-SEM-vels-{}.csv'
            semOutFile = semOutFile.format(anim_id)
            allsem.to_csv(semOutFile)

            # Concatinate the full freezing file per-animal
            # allFreeze = pd.concat([epochFreezing, dEpochFreezing],axis=1)
            allFreeze = pd.DataFrame()
            for i in range(0, len(epochFreezing)):
                allFreeze = pd.concat([allFreeze, epochFreezing[i]], axis=1)
                for j in range(0, len(dEpochFreezing[i])):
                    allFreeze = pd.concat([allFreeze, dEpochFreezing[i][j]], axis=1)
            freezeOutFile = outpath + '/' + trial_type_abbr + '-Freezing-{}.csv'
            freezeOutFile = freezeOutFile.format(anim_id)
            allFreeze.to_csv(freezeOutFile)

            # Concatinate the full darting file per-animal
            # allDart = pd.concat([epochDarting, dEpochDarting],axis=1)
            allDart = pd.DataFrame()
            for i in range(0, len(epochDarting)):
                allDart = pd.concat([allDart, epochDarting[i]], axis=1)
                for j in range(0, len(dEpochDarting[i])):
                    allDart = pd.concat([allDart, dEpochDarting[i][j]], axis=1)
            dartOutFile = outpath + '/' + trial_type_abbr + '-Darting-{}.csv'
            dartOutFile = dartOutFile.format(anim_id)
            allDart.to_csv(dartOutFile)

            ####
            # (the above will end up being one function called for each vel measurement)
            ####

    for k in range(0,
                   len(trialType_list)):  # Should produce darting and freezing files for each trial type x epoch x sub-epoch

        # this runs does compilation specifically for baseline measurements (i.e. not per epoch)
        srf.compile_baseline_sr(trialType_list[k], outpath, outpath2)

        # detectionSettingsLabel is ['Fear Conditioning'] by default
        # raw_epochSettings[detectionSettingsLabel[k]] is a dict of dicts {'Tone': etc} by default
        for epoch_iter in raw_epochSettings[detectionSettingsLabel[k]]:
            epoch_ct = int(raw_epochSettings[detectionSettingsLabel[k]][epoch_iter]['EpochCount'])
            if (epoch_ct == 0):
                continue
            # srf.compile_SR(trialType_list[k], epoch_iter, epoch_ct, 1, [''],'Darting',outpath,outpath2)

            # TODO: figure out why this is being called twice? also, get rid of unnecessary args

            # run the compile function without derived epochs
            srf.compile_SR(trialType_list[k], epoch_iter, epoch_ct, 1, [''], 'Freezing', outpath, outpath2)
            # srf.compile_SR(trialType_list[k], epoch_iter, epoch_ct, 1, list(raw_epochSettings[detectionSettingsLabel[k]][epoch_iter]['SubEpochs'].keys()),'Darting',outpath,outpath2)

            # run the compile function with derived epochs
            srf.compile_SR(trialType_list[k], epoch_iter, epoch_ct, 1,
                           list(raw_epochSettings[detectionSettingsLabel[k]][epoch_iter]['SubEpochs'].keys()),
                           'Freezing', outpath, outpath2)

            # for dEpoch_iter in raw_epochSettings[detectionSettingsLabel[k]][epoch_iter]['SubEpochs']:
            #     srf.compile_SR(trialType_list[k], epoch_iter, epoch_ct, 1, [dEpoch_iter],'Darting',outpath,outpath2)
            #     srf.compile_SR(trialType_list[k], epoch_iter, epoch_ct, 1, [dEpoch_iter],'Freezing',outpath,outpath2)
    # TODO: move this outside of function
    prog_win.close()
