class InOutSettings:
    """
    Class for settings related to input and output files and paths
    """

    def __init__(self, inpath='./', ind_outpath='./', com_outpath='./',
                 com_only=False, full_vel=False):
        self.inpath = inpath
        self.ind_outpath = ind_outpath
        self.com_outpath = com_outpath
        self.com_only = com_only
        self.full_vel = full_vel

    @property
    def inpath(self):
        return self.__inpath

    @property
    def ind_outpath(self):
        return self.__ind_outpath

    @property
    def com_outpath(self):
        return self.__com_outpath

    @property
    def com_only(self):
        return self.__com_only

    @property
    def full_vel(self):
        return self.__full_vel

    @inpath.setter
    def inpath(self, inpath):
        self.__inpath = inpath

    @ind_outpath.setter
    def ind_outpath(self, ind_outpath):
        self.__ind_outpath = ind_outpath

    @com_outpath.setter
    def com_outpath(self, com_outpath):
        self.__com_outpath = com_outpath

    @com_only.setter
    def com_only(self, com_only):
        self.__com_only = com_only

    @full_vel.setter
    def full_vel(self, full_vel):
        self.__full_vel = full_vel


class SheetSettings:
    """
    Class for settings related to input sheets
    """

    def __init__(self, sheet_list=None, trial_type_list=None):
        self.sheet_list = sheet_list
        self.trial_type_list = trial_type_list

    @classmethod
    def settings_from_dict(cls, epoch_settings, settings_dict):
        # transform raw settings to lists
        settings = []
        # raw settings are dict values
        for raw_setting in settings_dict.values():
            # split the setting into a list and strip left whitespace
            setting = [subsetting.lstrip() for subsetting in raw_setting.split(',')]
            settings.append(setting)
        sheet_list, detection_labels, trial_full_list, trial_abbr_list = settings

        # add default epochs for any new detection_labels
        epoch_settings = epoch_settings.init_default_epochs(detection_labels)

        # create list of TrialType objects
        trial_type_list = []
        for detect_label, trial_full, trial_abbr in zip(detection_labels, trial_full_list, trial_abbr_list):
            trial_type_item = TrialType(detect_label, trial_full, trial_abbr)
            trial_type_list.append(trial_type_item)

        return cls(sheet_list=sheet_list, trial_type_list=trial_type_list), epoch_settings

    @property
    def sheet_list(self):
        return self.__sheet_list

    @property
    def trial_type_list(self):
        return self.__trial_type_list

    @sheet_list.setter
    def sheet_list(self, sheet_list):
        if sheet_list is None:
            sheet_list = ['Track-Arena 1-Subject 1',
                          'Track-Arena 2-Subject 1',
                          'Track-Arena 3-Subject 1',
                          'Track-Arena 4-Subject 1']
        self.__sheet_list = sheet_list

    @trial_type_list.setter
    def trial_type_list(self, trial_type_list):
        if trial_type_list is None:
            trial_type_list = [TrialType()]
        self.__trial_type_list = trial_type_list

    def __repr__(self):
        # trial_type_strings = [str(trial_type) for trial_type in self.trial_type_list]
        return f'sheet list: {self.sheet_list}, trial type list: {self.trial_type_list}'

    def __str__(self):
        trial_type_strings = [str(trial_type) for trial_type in self.trial_type_list]
        return f'sheet list: {self.sheet_list}, trial type list: {trial_type_strings}'

    def disintegrate_sheet_settings(self):
        detection_labels = [trial_type.detection_settings_labels for trial_type in self.trial_type_list]
        trial_type_fulls = [trial_type.trial_type_full for trial_type in self.trial_type_list]
        trial_type_abbrs = [trial_type.trial_type_abbr for trial_type in self.trial_type_list]

        return self.sheet_list, detection_labels, trial_type_fulls, trial_type_abbrs

    def get_trial_type_by_label(self, detection_label):
        """
        Get the TrialType object that matches the given label, if it exists
        """
        try:
            # provide default with next instead? (see next() documentation) (eg. next((arg for arg in sys.argv if not os.path.exists(arg)), None))
            return next(trial_types for trial_types in self.trial_type_list if
                        trial_types.detection_settings_label == detection_label)
        except StopIteration:
            print(f'No trial type matches given detection label {detection_label}')
            raise KeyError


class TrialType:
    """
    Class for settings related to trial type
    """

    def __init__(self,
                 detection_settings_label='Fear Conditioning',
                 trial_type_full='Fear Conditioning',
                 trial_type_abbr='FC'):
        self.detection_settings_label = detection_settings_label
        self.trial_type_full = trial_type_full
        self.trial_type_abbr = trial_type_abbr

    @property
    def detection_settings_label(self):
        return self.__detection_settings_label

    @property
    def trial_type_full(self):
        return self.__trial_type_full

    @property
    def trial_type_abbr(self):
        return self.__trial_type_abbr

    @detection_settings_label.setter
    def detection_settings_label(self, detection_settings_label):
        self.__detection_settings_label = detection_settings_label

    @trial_type_full.setter
    def trial_type_full(self, trial_type_full):
        self.__trial_type_full = trial_type_full

    @trial_type_abbr.setter
    def trial_type_abbr(self, trial_type_abbr):
        self.__trial_type_abbr = trial_type_abbr

    def __repr__(self):
        return f'{self.detection_settings_label}, {self.trial_type_full}, {self.__trial_type_abbr}'

    def __str__(self):
        return f'(detection label: {self.detection_settings_label}, trial type: {self.trial_type_full}, trial abbr: {self.__trial_type_abbr})'


class EpochSettings(dict):
    """
    Class to map detection settings names to settings (inherits from dict)
    """

    def __init__(self, use_default=True):
        if use_default:
            self['Fear Conditioning'] = [Epoch()]

    def get_epoch_by_label(self, trial_type, epoch_label):
        """
        Return epoch for given trial type that matches given label (None if no matches)
        """
        # provide default with next instead? (see next() documentation) (eg. next((arg for arg in sys.argv if not os.path.exists(arg)), None))
        return next((epoch for epoch in self[trial_type] if epoch.label == epoch_label), None)

    def init_default_epochs(self, detection_settings_labels):
        """
        Add default Epoch objects to settings for all keys in given key_list that are not already present.
        """
        for label in detection_settings_labels:
            if label not in self:
                self[label] = [Epoch()]
        return self


class Epoch:
    """
    Class for settings related to epochs
    """

    def __init__(self, label='Tone', use_space=True, epoch_count=7, sub_epochs=None):
        self.label = label
        self.use_space = use_space
        self.epoch_count = epoch_count
        self.sub_epochs = sub_epochs

    @property
    def label(self):
        return self.__label

    @property
    def use_space(self):
        return self.__use_space

    @property
    def epoch_count(self):
        return self.__epoch_count

    @property
    def sub_epochs(self):
        return self.__sub_epochs

    @label.setter
    def label(self, label):
        self.__label = label

    @use_space.setter
    def use_space(self, use_space):
        self.__use_space = use_space

    @epoch_count.setter
    def epoch_count(self, epoch_count):
        self.__epoch_count = epoch_count

    @sub_epochs.setter
    def sub_epochs(self, sub_epochs):
        if sub_epochs is None:
            sub_epochs = {'PreTone': '0,-30,0,True',
                          'Shock': '-1,0,5,True',
                          'PostShock': '-1,5,35,True'}
        self.__sub_epochs = sub_epochs

    def __repr__(self):
        return f'{self.label}, {self.use_space}, {self.epoch_count}, {self.sub_epochs}'

    def __str__(self):
        return f'(label: {self.label}, use space: {self.use_space}, ' \
               f'epoch count: {self.epoch_count}, sub epochs: {self.sub_epochs})'

    def get_sub_epoch_lists(self):
        """
        Return the sub epoch labels as a list and the sub epoch timings as a list of lists
        """
        sub_epoch_labels = list(self.sub_epochs.keys())
        sub_epoch_timings = [timing.split(',') for timing in self.sub_epochs.values()]

        return sub_epoch_labels, sub_epoch_timings

    def get_sub_epochs_with_int_timings(self):
        """
        Return the sub_epoch dict of this epoch, with the timings converted to ints.
        """
        return {k: [int(x) for x in v.split(',')[0:3]] for k, v in self.sub_epochs.items()}


class TrialSettings:
    """
    Class for settings related to trial specifications
    """

    def __init__(self, bin_secs=1, baseline_duration=120.0, freeze_thresh=0.1, dart_thresh=20.0):
        self.bin_secs = bin_secs
        self.baseline_duration = baseline_duration
        self.freeze_thresh = freeze_thresh
        self.dart_thresh = dart_thresh

    @classmethod
    def trial_from_dict(cls, trial_dict):
        bin_secs, baseline_duration, freeze_thresh, dart_thresh = (float(setting) for setting in trial_dict.values())
        bin_secs = int(bin_secs)
        return cls(bin_secs, baseline_duration, freeze_thresh, dart_thresh)

    @property
    def bin_secs(self):
        return self.__bin_secs

    @property
    def baseline_duration(self):
        return self.__baseline_duration

    @property
    def freeze_thresh(self):
        return self.__freeze_thresh

    @property
    def dart_thresh(self):
        return self.__dart_thresh

    @bin_secs.setter
    def bin_secs(self, bin_secs):
        self.__bin_secs = bin_secs

    @baseline_duration.setter
    def baseline_duration(self, baseline_duration):
        self.__baseline_duration = baseline_duration

    @freeze_thresh.setter
    def freeze_thresh(self, freeze_thresh):
        self.__freeze_thresh = freeze_thresh

    @dart_thresh.setter
    def dart_thresh(self, dart_thresh):
        self.__dart_thresh = dart_thresh

    def __str__(self):
        return f'(bin secs: {self.bin_secs}, baseline dur: {self.baseline_duration}, ' \
               f'freeze thresh: {self.freeze_thresh}, dart thresh: {self.dart_thresh})'

