% EEG BIDS Pre processing pipeline by Arthur Alonso 2024
% ------------------------------------------------
 
clear
clc
loadDir = 'Datasets\BCICIV_2a_gdf\';
saveDir = 'Datasets\BCICIV_2a_gdf\Derivatives\';

%
% LIST ALL THE SUBJECTS
%
ListSub=dir(fullfile(loadDir));  % List all the volunteers on the sourcedatafolder
TotSub=size(ListSub,1); % the first two are "." and ".." 

for NumSub = 3: TotSub % Iterate on the Subject list, the first two are "." and ".." so begin with index 3
    NameSub = ListSub(NumSub).name;
    Pathfile = [loadDir, NameSub]; % Path to the sub file'
    NameFilePure = NameSub(1:(find(NameSub=='.')-1));
    savePathMat = [saveDir,NameFilePure,'.mat']; %save path for the .mat file
    disp(savePathMat) % display the path of the file being created
    if exist(savePathMat, "file") % Do not replace the file
        %
        % Now Preprocess the data
        %
        EEG = pop_biosig(Pathfile); % load the data
        % Check the channels "EEG.chanlocs.labels"
        % Extract EOG channels
        EOG = pop_select(EEG, 'channel', {'EOG-left', 'EOG-central', 'EOG-right'});

        % remove the EOG channels
        EEG = pop_select(EEG, 'nochannel', {'EOG-left', 'EOG-central', 'EOG-right'});
        
        % re-reference, CAR,common average reference
        % To be verified if referencing on CZ is the better
        EEG = pop_reref( EEG, []);
        EEGreref = EEG;

        % filter
        EEG = pop_eegfiltnew(EEG, 'locutoff', 7, 'hicutoff', 35); % Bandpass filter: 1-40 Hz
        %EEG = pop_eegfiltnew(EEG,  7, []);
        %EEG = pop_eegfiltnew(EEG, [], 35);
        
        % re-sample to 128 Hz to reduce computational cost
        % EEG = pop_resample(EEG, 128);
        % remove base ???????????
        % EEG = pop_rmbase(EEG, [1 EEG.times(end)]);   
        % EEGbaseless = EEG;         

        % remove the EOG with AAR_fd (Automatic Artifact Removal)
        EEG = pop_autobsseog( EEG, [416.32], [416.32], 'sobi', {'eigratio', [1000000]}, 'eog_fd', {'range',[2  22]});
        EEGclean = EEG;

        % reref the data
        EEG = pop_reref( EEG, []);

        % check the data
        EEG = eeg_checkset(EEG);

    
        %pop_eegplot(EEGreref, 1, 1, 1);
        %pop_eegplot(EEG, 1, 1, 1);
        %pop_eegplot(EMG, 1, 1, 1); 
        %pop_eegplot(EOG, 1, 1, 1);  
 
        % Epoching
        % Define parameters
        trial_duration = 3; % seconds (from cue onset to end of trial: 3s to 6s after trial start)
        sampling_rate = EEG.srate; % e.g., 128 Hz
        trial_samples = trial_duration * sampling_rate;
        class_event_codes = [769, 770, 771, 772]; % Event codes for left hand, right hand, feet, tongue

        % 2. Find trial events
        trial_events = [];
        for i = 1:length(EEG.event)
            if ismember(EEG.event(i).edftype, class_event_codes)
                trial_events = [trial_events, i];
            end
        end

        % 3. Initialize arrays
        num_trials = length(trial_events);
        num_channels = EEG.nbchan;
        all_data = zeros(num_trials, num_channels, trial_samples);
        all_label = zeros(num_trials, 1);

        % 4. Extract trials
        for i = 1:num_trials
            event_idx = trial_events(i);
            event_pos = EEG.event(event_idx).latency;
            event_type = EEG.event(event_idx).edftype;
            
            % Convert event type to class label (0-3)
            class_label = event_type - 769;
            
            % Extract data segment
            start_sample = round(event_pos);
            end_sample = start_sample + trial_samples - 1;
            
            % Check if trial goes beyond data bounds
            if end_sample <= size(EEG.data, 2)
                % Extract and store trial data
                trial_data = EEG.data(:, start_sample:end_sample);
                all_data(i, :, :) = trial_data;
                all_label(i) = class_label;
            end
        end

        % 6. Reshape data to match Python expectations (samples × channels × length)
        all_data = permute(all_data, [1, 2, 3]);

        % 7. Save as .mat file
        save(savePathMat, 'all_data', 'all_label');

        % display class distribution
        disp(['Class distribution: ', ...
      'Left hand: ', num2str(sum(all_label == 0)), ', ', ...
      'Right hand: ', num2str(sum(all_label == 1)), ', ', ...
      'Feet: ', num2str(sum(all_label == 2)), ', ', ...
      'Tongue: ', num2str(sum(all_label == 3))]);
    end
end 