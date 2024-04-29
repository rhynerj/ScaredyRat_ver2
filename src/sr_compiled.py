# file to run compiled analysis for all animals and output appropriate analysis files
# can be run independently using main function (requires input and output folder parameters) -> will probably not be implemented
import src.sr_settings as srs
import src.sr_functions as srf
from src.sr_functions import all_darting_out, all_freezing_out, all_velocity_out, all_subepoch_out


def compile_SR(trial_type, epoch_label, num_epoch, d_epoch_list, inpath, outpath, full_analysis):
    """
    TODO -> rename function more descriptive name (i.e. is writing summary csvs)
    """

    epoch_prefix = f'{trial_type}-{epoch_label}'

    # print(inpath+trialType + '-' + behavior)
    # combine the data from all darting csvs and write to file
    all_darting_out(epoch_prefix, inpath, outpath)
    # darting_csvs = scaredy_find_csvs(inpath, trial_type + '-' + epoch_label + '-darting')
    # darting_outfile = os.path.join(outpath, 'All-' + trial_type + '-' + epoch_label + '-darting.csv')
    # darting_data = concat_all_darting(darting_csvs, 0)
    # darting_data.to_csv(darting_outfile)

    # combine the data from all freezing csvs and write to file
    all_freezing_out(epoch_prefix, inpath, outpath)
    # freezing_csvs = scaredy_find_csvs(inpath, trial_type + '-' + epoch_label + '-freezing')
    # freezing_outfile = os.path.join(outpath, 'All-' + trial_type + '-' + epoch_label + '-Percent_freezing.csv')
    # freezing_data = concat_all_freezing(freezing_csvs, 0)
    # freezing_data.to_csv(freezing_outfile)

    # combine data from all csvs related to general velocity measurements and write to output file
    all_velocity_out(trial_type, inpath, outpath, num_epoch, full_analysis)
    # mean_csvs = scaredy_find_csvs(inpath, trial_type + '-mean')
    # med_csvs = scaredy_find_csvs(inpath, trial_type + '-median')
    # max_csvs = scaredy_find_csvs(inpath, trial_type + '-max')
    # sem_csvs = scaredy_find_csvs(inpath, trial_type + '-SEM')
    # maxes = compress_data(max_csvs, 0)
    # means = compress_data(mean_csvs, 0)
    # meds = compress_data(med_csvs, 0)
    # sems = compress_data(sem_csvs, 0)
    # all_data = concat_data(means, sems, meds, maxes, num_epoch)
    # outfile = os.path.join(outpath, 'All-' + trial_type + '-VelocitySummary.csv')
    # all_data.to_csv(outfile)

    # combine data from all csvs related to epoch-specific velocity measurements and write to output file
    all_velocity_out(epoch_prefix, inpath, outpath, num_epoch, full_analysis, True)
    # e_mean_csvs = scaredy_find_csvs(inpath, trial_type + '-' + epoch_label + '-mean')
    # e_med_csvs = scaredy_find_csvs(inpath, trial_type + '-' + epoch_label + '-median')
    # e_max_csvs = scaredy_find_csvs(inpath, trial_type + '-' + epoch_label + '-max')
    # e_sem_csvs = scaredy_find_csvs(inpath, trial_type + '-' + epoch_label + '-SEM')
    # e_maxes = compress_data(e_max_csvs, 0)
    # e_means = compress_data(e_mean_csvs, 0)
    # e_meds = compress_data(e_med_csvs, 0)
    # e_sems = compress_data(e_sem_csvs, 0)
    # all_data = concat_data(e_means, e_sems, e_meds, e_maxes, num_epoch)
    # outfile = os.path.join(outpath, 'All-' + trial_type + '-' + epoch_label + '-VelocitySummary.csv')
    # all_data.to_csv(outfile)

    # combine a more detailed version of the epoch max velocity and write to its own file
    # e_maxes_single = concat_all_max(e_max_csvs)
    # outfile = os.path.join(outpath, 'All-' + trial_type + '-' + epoch_label + '-MaxVel.csv')
    # e_maxes_single.to_csv(outfile)

    # allMax = concat_all_max(max_csvs)
    # outfile = os.path.join(outpath, 'All-'+ trialType + '-MaxVel.csv' )
    # allMax.to_csv(outfile)

    # summaries for each sub_epoch
    all_subepoch_out(d_epoch_list, epoch_prefix, inpath, outpath, num_epoch, full_analysis)
    # num_d_epoch = len(d_epoch_list)
    # # print(dEpoch_list)
    # if num_d_epoch == 0 or d_epoch_list == ['']:
    #     return
    # # print(num_dEpoch)
    # for i in range(0, num_d_epoch):
    #     # print(trialType + '-' + epochLabel + '_' + dEpoch_list[i] + '-max')
    #     maxdECSVs = scaredy_find_csvs(inpath, trial_type + '-' + epoch_label + '_' + d_epoch_list[i] + '-max')
    #     # print(maxdECSVs)
    #     allMax = concat_all_max(maxdECSVs)
    #     outfile = os.path.join(outpath, 'All-' + trial_type + '-' + epoch_label + '_' + d_epoch_list[i] + '-MaxVel.csv')
    #     allMax.to_csv(outfile)
    #
    #     meandECSVs = scaredy_find_csvs(inpath, trial_type + '-' + epoch_label + '_' + d_epoch_list[i] + '-mean')
    #     meddECSVs = scaredy_find_csvs(inpath, trial_type + '-' + epoch_label + '_' + d_epoch_list[i] + '-median')
    #     SEMdECSVs = scaredy_find_csvs(inpath, trial_type + '-' + epoch_label + '_' + d_epoch_list[i] + '-SEM')
    #     maxes = compress_data(maxdECSVs, 0)
    #     means = compress_data(meandECSVs, 0)
    #     meds = compress_data(meddECSVs, 0)
    #     sems = compress_data(SEMdECSVs, 0)
    #     all_data = concat_data(means, sems, meds, maxes, num_epoch)
    #     outfile = os.path.join(outpath,
    #                            'All-' + trial_type + '-' + epoch_label + '_' + d_epoch_list[i] + '-VelocitySummary.csv')
    #     all_data.to_csv(outfile)
    #
    #     dEdartingCSVs = scaredy_find_csvs(inpath, trial_type + '-' + epoch_label + '_' + d_epoch_list[i] + '-darting')
    #     dEdarting_outfile = os.path.join(outpath,
    #                                      'All-' + trial_type + '-' + epoch_label + '_' + d_epoch_list[i] + '-darting.csv')
    #     dEdartingData = concat_all_darting(dEdartingCSVs, 0)
    #     dEdartingData.to_csv(dEdarting_outfile)
    #
    #     dEfreezingCSVs = scaredy_find_csvs(inpath, trial_type + '-' + epoch_label + '_' + d_epoch_list[i] + '-freezing')
    #     dEfreezing_outfile = os.path.join(outpath, 'All-' + trial_type + '-' + epoch_label + '_' + d_epoch_list[
    #         i] + '-Percent_freezing.csv')
    #     dEfreezingData = concat_all_freezing(dEfreezingCSVs, 0)
    #     dEfreezingData.to_csv(dEfreezing_outfile)


# done: moved this to the compile file
def compiled_output(in_out_settings=srs.InOutSettings(), sheet_settings=srs.SheetSettings(),
                    epoch_settings=srs.EpochSettings()):
    """
    Generate and write compiled csvs to compile outpath (based on individual csvs in individual outpath)
    """
    # iterate over trial_type_list (from SheetSettings object)
    for trial_type in sheet_settings.trial_type_list:
        trial_type_abbr = trial_type.trial_type_abbr
        # compile baseline (getting csvs from ind outpath, ouputting to comb outpath); takes trial_type_abbr
        srf.compile_baseline_sr(trial_type_abbr, in_out_settings.ind_outpath, in_out_settings.com_outpath)

        # compile for each epoch and each subepoch -> need to modify compileSR to toggle full_analysis T/F
        # -> also, remove 'num-d_epoch' and 'behavior' arguments (not used)
        # ran twice in original, once with and once without valid d_epoch_list, but logically that would just cause some files to be written twice (not noticeable in output bc overwrites)
        # get epochs for current trial
        epochs = epoch_settings[trial_type.detection_settings_label]
        for epoch in epochs:
            if epoch.epoch_count == 0:
                continue
            sub_epoch_labels, _ = epoch.get_sub_epoch_lists()
            compile_SR(trial_type_abbr, epoch.label, epoch.epoch_count,
                       sub_epoch_labels, in_out_settings.ind_outpath,
                       in_out_settings.com_outpath, in_out_settings.full_vel)

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
    #
    #         # for dEpoch_iter in raw_epochSettings[detectionSettingsLabel[k]][epoch_iter]['SubEpochs']:
    #         #     srf.compile_SR(trialType_list[k], epoch_iter, epoch_ct, 1, [dEpoch_iter],'Darting',outpath,outpath2)
    #         #     srf.compile_SR(trialType_list[k], epoch_iter, epoch_ct, 1, [dEpoch_iter],'Freezing',outpath,outpath2)
