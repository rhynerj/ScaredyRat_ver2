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


def compress_data(csvlist, data_loc):
    """
    Get the given time bin from each of the CSVs in the given list, and combine the data into a single df with the
    animal id as the index.
    """
    all_anims = pd.DataFrame()
    for csv in csvlist:
        anim = get_anim(csv)
        df = pd.read_csv(csv, index_col=0).transpose()
        curr_anim_val = pd.DataFrame([df.iloc[data_loc]], index=[anim])
        all_anims = pd.concat([all_anims, curr_anim_val])

    return all_anims


def concat_all_csv_list(csvlist, data_loc):
    """
    Return the combined data for the given time bin for all animals in the CSV list
    """
    combined = compress_data(csvlist, data_loc)
    return combined


def compiled_out(combine_fn, data_loc, name, prefix, inpath, outpath, compile_dfs, compile_names,
                 file_spec=''):
    """
    Combine data from csvs that match given prefix/name combine, output combined data file, add to df and name lists.
    Return updated df and name lists.
    """
    # get csv list
    csv_list = scaredy_find_csvs(inpath, f'{prefix}-{name}')
    # combine csvs based on function
    combined_data = combine_fn(csv_list, data_loc)
    # output file
    outfile = os.path.join(outpath, f'All-{prefix}-{file_spec}{name}.csv')
    combined_data.to_csv(outfile)

    # update and return lists
    compile_dfs.append(combined_data)
    compile_names.append(f'{file_spec}{name}')

    return compile_dfs, compile_names


def compile_baseline_sr(prefix, inpath, outpath, compile_dfs, compile_names):
    """
    Combine the data from the csvs for the baseline measurements for each animal into a single csv file.
    Add both baseline dfs to compile lists and return updated.
    """
    # freezing
    compile_dfs, compile_names = \
        compiled_out(concat_all_csv_list, 2, 'baseline-freezing', prefix, inpath, outpath, compile_dfs, compile_names)
    # freezing_name = 'baseline-freezing'
    # baseline_freezing_csvs = scaredy_find_csvs(inpath, f'{prefix}-{freezing_name}')
    # # output
    # baseline_freezing_data = concat_all_csv_list(baseline_freezing_csvs, 2)
    # freezing_outfile = os.path.join(outpath, f'All-{prefix}-{freezing_name}.csv')
    # baseline_freezing_data.to_csv(freezing_outfile)
    # # add to super compile
    # compile_dfs.append(baseline_freezing_data)
    # compile_names.append(freezing_name)

    # darting
    compile_dfs, compile_names = \
        compiled_out(concat_all_csv_list, 0, 'baseline-darting', prefix, inpath, outpath, compile_dfs, compile_names)
    # darting_name = 'baseline-darting'
    # baseline_darting_csvs = scaredy_find_csvs(inpath, f'{prefix}-{darting_name}')
    #
    # baseline_darting_data = concat_all_csv_list(baseline_darting_csvs, 0)
    # darting_outfile = os.path.join(outpath, f'All-{prefix}-{darting_name}.csv')
    # baseline_darting_data.to_csv(darting_outfile)
    # # add to super compile
    # compile_dfs.append(baseline_darting_data)
    # compile_names.append(darting_name)

    return compile_dfs, compile_names


def all_darting_out(prefix, inpath, outpath, compile_dfs, compile_names):
    """
    Combine the data from all darting csvs and write to file
    """
    compile_dfs, compile_names = \
        compiled_out(concat_all_csv_list, 0, 'darting', prefix, inpath, outpath, compile_dfs, compile_names)
    # darting_csvs = scaredy_find_csvs(inpath, f'{prefix}-darting')
    # darting_outfile = os.path.join(outpath, f'All-{prefix}-darting.csv')
    # darting_data = concat_all_csv_list(darting_csvs, 0)
    # darting_data.to_csv(darting_outfile)

    return compile_dfs, compile_names


def concat_all_freezing(csvlist, tbin):
    """
    Return the combined freezing data for the given time bin for all animals in the CSV list
    """
    row = (tbin * 3) + 2  # for freezing, row is slightly offset from time bin
    freezing = compress_data(csvlist, row)
    return freezing


def all_freezing_out(prefix, inpath, outpath, compile_dfs, compile_names):
    """
    Combine the data from all freezing csvs and write to file
    """
    compile_dfs, compile_names = \
        compiled_out(concat_all_csv_list, 2, 'freezing', prefix, inpath, outpath, compile_dfs, compile_names,
                     file_spec='percent')
    # freezing_csvs = scaredy_find_csvs(inpath, f'{prefix}-freezing')
    # freezing_outfile = os.path.join(outpath, f'All-{prefix}-Percent_freezing.csv')
    # freezing_data = concat_all_freezing(freezing_csvs, 0)
    # freezing_data.to_csv(freezing_outfile)

    return compile_dfs, compile_names


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


def all_velocity_out(prefix, inpath, outpath, full_analysis, compile_dfs, compile_names, epoch_level=False):
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

        # add maxes to compile
        # update and return lists
        compile_dfs.append(e_maxes_single)
        compile_names.append('maxvels')

    return compile_dfs, compile_names


def all_subepoch_out(d_epoch_list, prefix, inpath, outpath, full_analysis, compile_dfs, compile_names):
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
        compile_dfs, compile_names = \
            all_velocity_out(d_epoch_prefix, inpath, outpath, full_analysis, compile_dfs, compile_names,
                             epoch_level=True)

        # darting summary data
        compile_dfs, compile_names = \
            all_darting_out(d_epoch_prefix, inpath, outpath, compile_dfs, compile_names)

        # freezing summary data
        compile_dfs, compile_names = \
            all_freezing_out(d_epoch_prefix, inpath, outpath, compile_dfs, compile_names)

    return compile_dfs, compile_names


def compile_SR(epoch_prefix, d_epoch_list, inpath, outpath, full_analysis, compile_dfs, compile_names):
    """
    Compile data from individual animals into summary csvs.
    """

    # combine the data from all darting csvs and write to file
    compile_dfs, compile_names = \
        all_darting_out(epoch_prefix, inpath, outpath, compile_dfs, compile_names)

    # combine the data from all freezing csvs and write to file
    compile_dfs, compile_names = \
        all_freezing_out(epoch_prefix, inpath, outpath, compile_dfs, compile_names)

    # combine data from all csvs related to general velocity measurements and write to output file
    compile_dfs, compile_names = \
        all_velocity_out(epoch_prefix, inpath, outpath, full_analysis, compile_dfs, compile_names)

    # combine data from all csvs related to epoch-specific velocity measurements and write to output file
    compile_dfs, compile_names = \
        all_velocity_out(epoch_prefix, inpath, outpath, full_analysis, compile_dfs, compile_names,
                         epoch_level=True)

    # summaries for each sub_epoch
    compile_dfs, compile_names = \
        all_subepoch_out(d_epoch_list, epoch_prefix, inpath, outpath, full_analysis, compile_dfs, compile_names)

    return compile_dfs, compile_names


def make_super_compile(compile_dfs, compile_names):
    """
    Create Multiindex for each compile_df using corresponding compile_name
    Combine compile_dfs to make super_compile_df
    Return super_compile_df
    """
    # add all multi-indices
    for compile_df, compile_name in zip(compile_dfs, compile_names):
        # multiindex, with name and original cols
        new_cols = pd.MultiIndex.from_product([[compile_name], compile_df.columns])
        # update cols
        compile_df.columns = new_cols
    # combine all to make super compile, on row index; Transpose
    super_compile_df = pd.concat(compile_dfs, axis=1).T

    return super_compile_df


def compiled_output(in_out_settings=srs.InOutSettings(), sheet_settings=srs.SheetSettings(),
                    epoch_settings=srs.EpochSettings()):
    """
    Generate and write compiled csvs to compile outpath (based on individual csvs in individual outpath).
    """
    # iterate over trial_type_list (from SheetSettings object)
    for trial_type in sheet_settings.trial_type_list:
        trial_type_abbr = trial_type.trial_type_abbr
        # init empty compile df and compile names lists
        compile_dfs = []
        compile_names = []

        # compile baseline (getting csvs from ind outpath, ouputting to comb outpath); takes trial_type_abbr
        compile_dfs, compile_names = \
            compile_baseline_sr(trial_type_abbr, in_out_settings.ind_outpath, in_out_settings.com_outpath,
                                compile_dfs, compile_names)

        # compile for each epoch and each subepoch -> need to modify compileSR to toggle full_analysis T/F
        # ran twice in original, once with and once without valid d_epoch_list, but logically that would just cause some files to be written twice (not noticeable in output bc overwrites)
        # get epochs for current trial
        epochs = epoch_settings[trial_type.detection_settings_label]
        if len(epochs) > 1:
            for epoch in epochs:
                if epoch.epoch_count == 0:
                    continue
                prefix = f'{trial_type_abbr}-{epoch}'
                sub_epoch_labels, _ = epoch.get_sub_epoch_lists()
                compile_dfs, compile_names = \
                    compile_SR(prefix, sub_epoch_labels, in_out_settings.ind_outpath, in_out_settings.com_outpath,
                               in_out_settings.full_vel, compile_dfs, compile_names)
        else:
            if epochs[0].epoch_count != 0:
                sub_epoch_labels, _ = epochs[0].get_sub_epoch_lists()
                compile_dfs, compile_names = \
                    compile_SR(trial_type_abbr, sub_epoch_labels, in_out_settings.ind_outpath,
                               in_out_settings.com_outpath, in_out_settings.full_vel, compile_dfs, compile_names)

        # make super compile file for trial type
        outfile = os.path.join(in_out_settings.com_outpath, f'{trial_type_abbr}-super-compile-file.csv')
        super_compile_df = make_super_compile(compile_dfs, compile_names)
        super_compile_df.to_csv(outfile)
