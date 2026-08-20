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
    PathSes = [loadDir, NameSub, '\']; % Path to the sessions folder
    if ~exist(PathSes) % if dosent exist create a folder
        mkdir(PathSes)
    end

    %
    % LIST ALL THE SESSIONS
    %
    ListSes=dir(fullfile(PathSes));  % List all the sessions for a given volunteer
    TotSes=size(ListSes,1); % the first two are "." and ".." 

    for NumSes = 3: TotSes % Iterate on the Sessions list, the first two are "." and ".." so begin with index 3
        NameSes = ListSes(NumSes).name;
        PathTech = [PathSes, NameSes, '\']; % Path to the techniques folder
        if ~exist(PathTech) % if dosent exist create a folder
            mkdir(PathTech)
        end


        %
        % List all the techniques
        %
        ListTech=dir(fullfile(PathTech));  % List all the techniques for a given volunteer
        TotTech=size(ListTech,1); % the first two are "." and ".." 

        %for NumTech = 3: TotTech % Iterate on the tech list, the first two are "." and ".." so begin with index 3
            %NameTech = ListTech(NumSes).name; %EEG, EMG, EOG...
            NameTech = 'EEG';
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
                checkPath = [saveDir, NameSub, '\',NameSes,'\',NameTech,'\', NameFilePure, '.set']; % check if the file already exists
                disp(checkPath) % display the path of the file being created
                %if ~exist(checkPath) % Do not replace the file
                    PathFullFile = [PathFile, NameFile]; % Full path to the file


                    %
                    % Now Preprocess the data
                    %
                    EEG = pop_loadcnt( PathFullFile , 'dataformat', 'auto', 'memmapfile', '');

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
                    EEG = pop_eegfiltnew(EEG, 'locutoff', 7, 'hicutoff', 35); % Bandpass filter: 1-40 Hz
                    %EEG = pop_eegfiltnew(EEG,  7, []);
                    %EEG = pop_eegfiltnew(EEG, [], 35);
                    
                    % recsample to 128 Hz to reduce computational cost
                    EEG = pop_resample(EEG, 128);
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

                    % save .set data
                    saveNameSet = [NameFilePure, '.set'];
                    EEG = pop_saveset( EEG, 'filename', saveNameSet ,'filepath',savePathSet);
                    % It can be saved directly as a matlab variable: 
                    % save([savePathSet, NameSub] , 'EEG');
                    
                    % Loading and saving bids parameters
                    % Copy and rename the files from the root to the individual folder
                    PathRoot = 'Datasets\MI3class1Limb';
                    filestocopy = {'task-motorimagery_electrodes.tsv', 'task-motorimagery_coordsystem.json', 'task-motorimagery_channels.tsv', 'task-motorimagery_eeg.json'};
                    for i = 1:length(filestocopy)
                        FileName = [NameSub, '_', NameSes, '_', char(filestocopy{i})];
                        copyfile([PathRoot, char(filestocopy{i})], fullfile(savePathSet, FileName));
                    end
                    %Copy the original events marker from the original dataset
                    if exist([PathRoot,NameSub,'\',NameSes,'\',NameTech,'\',NameSub,'_',NameSes,'_','task-motorimagery_events.tsv'])
                        copyfile([PathRoot,NameSub,'\',NameSes,'\',NameTech,'\',NameSub,'_',NameSes,'_','task-motorimagery_events.tsv'], savePathSet);
                        display(['Events file for ', NameSub, ' ', NameSes, ' copied'])
                    end
                    pop_eegplot(EEG, 1, 1, 1);
                    %pop_eegplot(EMG, 1, 1, 1);
                    %pop_eegplot(EOG, 1, 1, 1);
                %end
                
            end
        %end
    end
end