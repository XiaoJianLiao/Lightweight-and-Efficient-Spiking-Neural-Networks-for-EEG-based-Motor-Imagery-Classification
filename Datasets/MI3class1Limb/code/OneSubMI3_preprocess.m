% Modified pre processing by Arthur Alonso 2024
% ------------------------------------------------

clear
clc

%Load and save directories
loadDir = 'Datasets\MI3class1Limb\sourcedata\';
saveDir = 'Datasets\MI3class1Limb\derivatives\';

% Add EEGLAB / Neuroscan Extension / EEG-Bids extension / Biosig Extension / AAR extension to MATLAB path
% Change this path to where you have EEGLAB installed
eeglab_path = 'MatlabToolkits\eeglab'; % You need to set this to your EEGLAB installation path
addpath(eeglab_path);
eeglab nogui; % Initialize EEGLAB without GUI


%
% LIST ALL THE SUBJECTS
%
ListSub=dir(fullfile(loadDir));  % List all the volunteers on the sourcedatafolder
TotSub=size(ListSub,1); % the first two are "." and ".." 

NumSub = 3; % for NumSub = 3: TotSub % Iterate on the Subject list, the first two are "." and ".." so begin with index 3
NameSub = ListSub(NumSub).name;
PathSes = [loadDir, NameSub, '\']; % Path to the sessions folder
if ~exist(PathSes) % if dosent exist create a folder
    mkdir(PathSes)
end

% Initialize arrays to store all data and labels across sessions
all_data = []; 
all_label = [];

%
% LIST ALL THE SESSIONS
%
ListSes=dir(fullfile(PathSes));  % List all the sessions for a given volunteer
TotSes=size(ListSes,1); % the first two are "." and ".." 

NumSes = 3; % for NumSes = 3: TotSes % Iterate on the Sessions list, the first two are "." and ".." so begin with index 3
NameSes = ListSes(NumSes).name;
PathTech = [PathSes, NameSes, '\']; % Path to the techniques folder
if ~exist(PathTech) % if it dosent exist create a folder
    mkdir(PathTech)
end


%
% List all the techniques
%
ListTech=dir(fullfile(PathTech));  % List all the techniques for a given volunteer
TotTech=size(ListTech,1); % the first two are "." and ".." 

NumTech = 3; % for NumTech = 3: TotTech % Iterate on the tech list, the first two are "." and ".." so begin with index 3
NameTech = ListTech(NumSes).name; %EEG, EMG, EOG...
    % NameTech = 'EEG'
PathFile = [PathTech, NameTech, '\'];
savePathSet = [saveDir,NameSub,'\', NameSes, '\', NameTech, '\'];
if ~exist(savePathSet) % if dosent exist create a folder
    mkdir(savePathSet)
end


%
% List all the files
%
ListFiles = dir(PathFile); % List of the data files from the given session
TotFiles= size(ListFiles, 1); % Number of files

for NumFiles = 3:TotFiles % Iterating on the files the first two are "." and ".." so begin with index 3
    NameFile = ListFiles(NumFiles).name;
    NameFilePure = NameFile(1:(find(NameFile=='.')-1));
    checkPath = [saveDir, NameSub, '.set']; % check if the file already exists
    disp(checkPath) % display the path of the file being created
    %if ~exist(checkPath) % Do not replace the file
        PathFullFile = [PathFile, NameFile]; % Full path to the file


        %
        % Now Preprocess the data
        %
        EEG = pop_loadcnt( PathFullFile , 'dataformat', 'auto', 'memmapfile', '');
        LOADEDEEG = EEG;
        % add the location map according to the channel_name
        EEG=pop_chanedit(EEG, 'lookup','Datasets\MI3class1Limb\code\channel_dict.ced');
        
        % Extract EMG and EOG channels
        EMG = pop_select(EEG, 'channel', {'EMG1', 'EMG2'});
        EOG = pop_select(EEG, 'channel', {'HEO', 'VEO'});

        % remove HEO, M2, VEO, EMG1, EMG2 from EEG
        EEG = pop_select(EEG, 'nochannel', {'HEO', 'M2', 'VEO', 'EMG1', 'EMG2'});


        %TOBE VERIFIES WHAT CASES ARE THOSE remove the clutter before and after the task
        if strfind(NameFile, '-motorimagery_') % only for task session
            % remove the clutter before (<4s) and after (>5s)
            EEG = pop_select(EEG, 'notime', [0, EEG.urevent(1).latency/1000 - 4; EEG.urevent(end).latency/1000 + 5, EEG.xmax]);
        %             EEG = pop_select(EEG, 'notime', [EEG.urevent(end).latency/1000 + 5, EEG.xmax]);
        end
        CUTTEDEEG = EEG;

    
        
        % re-reference, CAR,common average reference
        % To be verified if referencing on CZ is the better
        EEG = pop_reref( EEG, []);
        EEGreref = EEG;

        % filter
        EEG = pop_eegfiltnew(EEG, 'locutoff', 2, 'hicutoff', 45); % Bandpass filter: 1-40 Hz
        %EEG = pop_eegfiltnew(EEG,  2, []);
        %EEG = pop_eegfiltnew(EEG, [], 45);
        EEGfiltered = EEG;

        % recsample to 90 Hz to reduce computational cost % Done on the postprocessing
        EEG = pop_resample(EEG, 90);
        EEGresampled = EEG;
        
        % remove base ???????????
        EEG = pop_rmbase(EEG, [1 EEG.times(end)]);   
        EEGbaseless = EEG;         

        % remove the EOG with AAR_fd
        EEG = pop_autobsseog( EEG, [416.32], [416.32], 'sobi', {'eigratio', [1000000]}, 'eog_fd', {'range',[2  22]});
        EEGclean = EEG;

        % reref the data
        EEG = pop_reref( EEG, []);

        % check the data
        EEG = eeg_checkset(EEG);

        %%
        % Epoching
        % Define parameters
        % According to the paper, each trial is 8s with 4s of actual MI task
        % The target appears at t=3s and stays for 4s (thus ending at t=7s)
        trial_duration = 4; % seconds (from target onset to end of trial)
        sampling_rate = EEG.srate; % e.g., 90 Hz after resampling
        trial_samples = round(trial_duration * sampling_rate);

        % For the MI-2 dataset:
        % We'll use: 0 = rest, 1 = elbow, 2 = hand
        is_rest_session = contains(NameFile, '-rest_');
        is_mi_session = contains(NameFile, '-motorimagery_');
        if is_mi_session
            % Process motor imagery session (hand/elbow)
            
            % 2. Find trial events for hand and elbow
            trial_events = [];
            for i = 1:length(EEG.event)
                % Check if event type is 1 (hand) or 2 (elbow)
                if ismember(EEG.event(i).type, {'1', '2'}) || ...
                (isnumeric(EEG.event(i).type) && ismember(EEG.event(i).type, [1, 2]))
                    trial_events = [trial_events, i];
                end
            end
            % 3. Initialize arrays
            num_trials = length(trial_events);
            num_channels = EEG.nbchan;
            session_data = zeros(num_trials, num_channels, trial_samples);
            session_label = zeros(num_trials, 1);
            
            % 4. Extract trials
            for i = 1:num_trials
                event_idx = trial_events(i);
                event_pos = EEG.event(event_idx).latency;
                
                % Get event type (1=hand, 2=elbow)
                if isnumeric(EEG.event(event_idx).type)
                    event_type = EEG.event(event_idx).type;
                else
                    event_type = str2double(EEG.event(event_idx).type);
                end

                % Map the event types to our label convention
                % According to paper: 1 = hand, 2 = elbow
                % We want: 0 = rest, 1 = elbow, 2 = hand
                if event_type == 1
                    label = 2;  % Hand
                elseif event_type == 2
                    label = 1;  % Elbow
                else
                    label = -1; % Invalid
                end

                % Extract data segment (4s from target onset)
                start_sample = round(event_pos);
                end_sample = start_sample + trial_samples - 1;
                
                % Check if trial goes beyond data bounds
                if end_sample <= size(EEG.data, 2)
                    % Extract and store trial data
                    trial_data = EEG.data(:, start_sample:end_sample);
                    session_data(i, :, :) = trial_data;
                    session_label(i) = event_type;
                else
                    fprintf('Trial %d exceeds data bounds. Skipping.\n', i);
                    session_data(i, :, :) = zeros(num_channels, trial_samples);
                    session_label(i) = 0;  % Mark as invalid
                end
            end
            % Remove invalid trials
            valid_trials = session_label > 0;
            session_data = session_data(valid_trials, :, :);
            session_label = session_label(valid_trials);
            
            % Append to all_data and all_label
            all_data = cat(1, all_data, session_data);
            all_label = cat(1, all_label, session_label);
            
        elseif is_rest_session
            % Process rest session
            
            % For rest sessions, we'll extract fixed-length segments
            % According to the paper, each rest trial lasts 4s as well
            
            % Calculate how many complete 4s trials we can extract
            total_samples = size(EEG.data, 2);
            num_complete_trials = floor(total_samples / trial_samples);
            
            % Initialize rest data
            rest_session_data = zeros(num_complete_trials, EEG.nbchan, trial_samples);
            rest_session_label = zeros(num_complete_trials, 1); % All zeros for rest
            
            % Extract each 4s segment
            for i = 1:num_complete_trials
                start_idx = (i-1) * trial_samples + 1;
                end_idx = i * trial_samples;
                
                if end_idx <= total_samples
                    segment = EEG.data(:, start_idx:end_idx);
                    rest_session_data(i, :, :) = segment;
                    % Label is already 0 for rest
                end
            end
            % Append to all_data and all_label
            all_data = cat(1, all_data, rest_session_data);
            all_label = cat(1, all_label, rest_session_label);
        end
    %end
end

% Save the combined data
subject_file = fullfile(saveDir, [NameSub '_eeg.mat']);
save(subject_file, 'all_data', 'all_label');

% Display class distribution
num_rest = sum(all_label == 0);
num_elbow = sum(all_label == 1);
num_hand = sum(all_label == 2);
fprintf('Class distribution: Rest: %d, Elbow: %d, Hand: %d\n', num_rest, num_elbow, num_hand);
