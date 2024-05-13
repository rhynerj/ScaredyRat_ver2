"""
Functions to run compiled analysis for all animals and output appropriate analysis files

"""

# imports
import os
import re
import pandas as pd

import src.sr_settings as srs


# functions
def get_anim(csv):
    """
    Extract animal ID from file name.
    """
    exp_rgx = re.compile(r'.*-([a-zA-Z]+-?\d+)\.[a-zA-Z]+$')
    match = exp_rgx.match(csv)

    if match:
        anim = match.group(1)
        return anim
    else:
        raise ValueError('File name does not contain valid animal ID.')


def scaredy_find_csvs(csv_dir, prefix):
    """
    Return list of csvs in given dir with given prefix.
    """
    csvlist = []

    for root, dirs, names in os.walk(csv_dir):
        for file in names:
            # print(file)
            if file.startswith(prefix) and 'times' not in file:
                f = os.path.join(root, file)
                # print(f)
                csvlist.append(f)

    return csvlist


def compress_data(csvlist, row):
    """
    Get the given time bin from each of the CSVs in the given list, and combine the data into a single df with the
    animal id as the index.
    """
    all_anims = pd.DataFrame()
    for csv in csvlist:
        anim = get_anim(csv)
        df = pd.read_csv(csv, index_col=0).transpose()
        print(df)
        curr_anim_val = pd.DataFrame([df.iloc[row]], index=[anim])
        all_anims = pd.concat([all_anims, curr_anim_val])

    return all_anims


def concat_all_csv_list(csvlist, loc):
    """
    Return the combined data for the given time bin for all animals in the CSV list
    """
    combined = compress_data(csvlist, loc)
    return combined


def compile_baseline_sr(trial_type, inpath, outpath):
    """
    Combine the data from the csvs for the baseline measurements for each animal into a single csv file.
    """
    # freezing
    baseline_freezing_csvs = scaredy_find_csvs(inpath, trial_type + '-baseline-freezing')

    baseline_freezing_data = concat_all_csv_list(baseline_freezing_csvs, 2)
    freezing_outfile = os.path.join(outpath, 'All-' + trial_type + '-baseline-freezing.csv')
    baseline_freezing_data.to_csv(freezing_outfile)

    # darting
    baseline_darting_csvs = scaredy_find_csvs(inpath, trial_type + '-baseline-darting')

    baseline_darting_data = concat_all_csv_list(baseline_darting_csvs, 0)
    darting_outfile = os.path.join(outpath, 'All-' + trial_type + '-baseline-darting.csv')
    baseline_darting_data.to_csv(darting_outfile)


def all_darting_out(prefix, inpath, outpath):
    """
    Combine the data from all darting csvs and write to file
    """
    darting_csvs = scaredy_find_csvs(inpath, f'{prefix}-darting')
    print(darting_csvs)
    darting_outfile = os.path.join(outpath, f'All-{prefix}-darting.csv')
    darting_data = concat_all_csv_list(darting_csvs, 0)
    darting_data.to_csv(darting_outfile)


def concat_all_freezing(csvlist, tbin):
    """
    Return the combined freezing data for the given time bin for all animals in the CSV list
    """
    row = (tbin * 3) + 2  # for freezing, row is slightly offset from time bin
    freezing = compress_data(csvlist, row)
    return freezing


def all_freezing_out(prefix, inpath, outpath):
    """
    Combine the data from all freezing csvs and write to file
    """
    freezing_csvs = scaredy_find_csvs(inpath, f'{prefix}-freezing')
    freezing_outfile = os.path.join(outpath, f'All-{prefix}-Percent_freezing.csv')
    freezing_data = concat_all_freezing(freezing_csvs, 0)
    freezing_data.to_csv(freezing_outfile)


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


def concat_vel_data(means, sems, meds, maxes):
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

    return all_data


def all_velocity_out(prefix, inpath, outpath, full_analysis, epoch_level=False):
    """
    Combine data from all csvs related to general velocity measurements and write to output file.
    If epoch_level, also write more detailed max data csv.
    """
    max_csvs = scaredy_find_csvs(inpath, f'{prefix}-max')
    max_csvs += scaredy_find_csvs(inpath, f'{prefix}-response')

    # full velocity summary only for full analysis (else just max)
    if full_analysis:
        mean_csvs = scaredy_find_csvs(inpath, f'{prefix}-mean')
        med_csvs = scaredy_find_csvs(inpath, f'{prefix}-median')
        sem_csvs = scaredy_find_csvs(inpath, f'{prefix}-SEM')

        # combine data into a single df for each csv list
        means, meds, maxes, sems = [compress_data(csvs, 0) for csvs in [mean_csvs, med_csvs, max_csvs, sem_csvs]]

        # merge data frames into one with all data
        all_data = concat_vel_data(means, sems, meds, maxes)

        outfile = os.path.join(outpath, f'All-{prefix}-VelocitySummary.csv')
        all_data.to_csv(outfile)

    if epoch_level:
        # combine a more detailed version of the epoch max velocity and write to its own file
        e_maxes_single = concat_all_max(max_csvs)
        outfile = os.path.join(outpath, f'All-{prefix}-MaxVel.csv')
        e_maxes_single.to_csv(outfile)


def all_subepoch_out(d_epoch_list, prefix, inpath, outpath, full_analysis):
    """
    Combine and output velocity, freezing, and darting data for each sub(derived)-epoch in the given list.
    """
    # check whether the sub-epoch input list is valid
    if not d_epoch_list or not d_epoch_list[0]:
        return
    for d_epoch in d_epoch_list:
        # add current sub-epoch name to prefix
        d_epoch_prefix = f'{prefix}-{d_epoch}'

        # velocity summary data
        all_velocity_out(d_epoch_prefix, inpath, outpath, full_analysis, True)

        # darting summary data
        all_darting_out(d_epoch_prefix, inpath, outpath)

        # freezing summary data
        all_freezing_out(d_epoch_prefix, inpath, outpath)


def compile_SR(epoch_prefix, d_epoch_list, inpath, outpath, full_analysis):
    """
    Compile data from individual animals into summary csvs.
    """

    # combine the data from all darting csvs and write to file
    all_darting_out(epoch_prefix, inpath, outpath)

    # combine the data from all freezing csvs and write to file
    all_freezing_out(epoch_prefix, inpath, outpath)

    # combine data from all csvs related to general velocity measurements and write to output file
    all_velocity_out(epoch_prefix, inpath, outpath, full_analysis)

    # combine data from all csvs related to epoch-specific velocity measurements and write to output file
    all_velocity_out(epoch_prefix, inpath, outpath, full_analysis, True)

    # summaries for each sub_epoch
    all_subepoch_out(d_epoch_list, epoch_prefix, inpath, outpath, full_analysis)


def compiled_output(in_out_settings=srs.InOutSettings(), sheet_settings=srs.SheetSettings(),
                    epoch_settings=srs.EpochSettings()):
    """
    Generate and write compiled csvs to compile outpath (based on individual csvs in individual outpath).
    """
    # iterate over trial_type_list (from SheetSettings object)
    for trial_type in sheet_settings.trial_type_list:
        trial_type_abbr = trial_type.trial_type_abbr
        # compile baseline (getting csvs from ind outpath, ouputting to comb outpath); takes trial_type_abbr
        compile_baseline_sr(trial_type_abbr, in_out_settings.ind_outpath, in_out_settings.com_outpath)

        # compile for each epoch and each subepoch -> need to modify compileSR to toggle full_analysis T/F
        # ran twice in original, once with and once without valid d_epoch_list, but logically that would just cause some files to be written twice (not noticeable in output bc overwrites)
        # get epochs for current trial
        epochs = epoch_settings[trial_type.detection_settings_label]
        print('epochs', epochs)
        if len(epochs) > 1:
            for epoch in epochs:
                if epoch.epoch_count == 0:
                    continue
                prefix = f'{trial_type_abbr}-{epoch}'
                sub_epoch_labels, _ = epoch.get_sub_epoch_lists()
                compile_SR(prefix, sub_epoch_labels, in_out_settings.ind_outpath, in_out_settings.com_outpath,
                           in_out_settings.full_vel)
        else:
            if epochs[0].epoch_count != 0:
                sub_epoch_labels, _ = epochs[0].get_sub_epoch_lists()
                compile_SR(trial_type_abbr, sub_epoch_labels, in_out_settings.ind_outpath, in_out_settings.com_outpath,
                           in_out_settings.full_vel)
