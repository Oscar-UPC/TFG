%%
clear all;
close all;
clc;
%%
addpath(('/autofs/vast/neuromod/personal_folders/asif_jamil/Projects/fieldtrip'))
addpath('/autofs/vast/neuromod/personal_folders/asif_jamil/Projects/misc_scripts');
addpath('/autofs/vast/neuromod/tDCS_EEG/scripts')
addpath /homes/7/aj123/Projects/scripts/
ft_defaults % just initializes fieldtrip

%% convert each project to BIDS
BIDS_DIR = '/autofs/vast/neuromod/tDCS_EEG/BIDS';
raw_dir = '/autofs/vast/neuromod/tDCS_EEG/Projects';
cd(raw_dir)
projects = dir([raw_dir '/*_*']);
disp({projects.name}');
%% Main Project loop
for p=[18:18]

    %11%size(projects,1)

    mkdir_if_not_exist([BIDS_DIR '/' projects(p).name])
    studyName=projects(p).name;
    % case by case
    if(contains(projects(p).name,'ADHD_attention'))
        labels={'Fp1','Fp2','F3','Fz','F4','P3','P4','Oz'}; % as per the documentation in Neuromod folder
        Fs = 500; % TODO: CHECK/CONFIRM
        mkdir_if_not_exist([BIDS_DIR '/' projects(p).name '/'])


        dataPath = ['/autofs/vast/neuromod/tDCS_EEG/Projects/01_ADHD_attention/EEG/ADD/'];
        cd(dataPath);
        subjFiles=dir('*ADD*');

        subjFiles = {subjFiles([subjFiles.isdir]).name};
        subjFiles = subjFiles(~ismember(subjFiles ,{'.','..'}))';
        for subj=1:size(subjFiles,1)%:-1:1 % ALWAYS GO BACKWARDS IN CASE THERE WERE RE-RUNS

            cd(subjFiles{subj});
            easyFiles=dir(['*.easy']);
            if(isempty(easyFiles))
                cd ..
                continue;
            end
            for eF=1:size(easyFiles,1)
                activeFile=easyFiles(eF).name

                if(contains(activeFile,'stim'))
                    warning('just a stim file, ignore')
                    continue;
                end
                % get the task name
                taskName = getTaskName(activeFile);

                % parse file name
                suStr=strsplit(activeFile,'_');

                % use the folder name
                folderStr=strsplit(subjFiles{subj},'_');
                visitNumber=str2num(folderStr{4});
                su =folderStr{3}; % suNumber
                suNumber=str2num(su);
                suDate = suStr{1};
                suGroup='';

                if(contains(activeFile,'pre'))
                    tptNumber=1;
                elseif(contains(activeFile,'post'))
                    tptNumber=2;
                end

                % sanity check
                if(isempty(visitNumber) | isempty(tptNumber))
                    error('missing session number of tpt number')
                end


                if(~exist([BIDS_DIR '/' projects(p).name '/sub-' num2str(suNumber) '/ses-' num2str(visitNumber) '/eeg/'...
                        'sub-' num2str(suNumber) '_ses-' num2str(visitNumber) '_task-' taskName '_run-' num2str(tptNumber) '_eeg.eeg'],'file'))

                    [data codes] = convert_easy(activeFile,labels);

                    [events, eventsTable] = process_codes(codes,studyName,taskName, Fs,suNumber,suGroup,visitNumber,tptNumber);
                    bids_root=[BIDS_DIR '/' projects(p).name '/'];
                    status = convert_bids(eventsTable, data, bids_root, taskName, suNumber,suGroup,visitNumber,tptNumber,suDate);

                    clear data;
                end
            end % easyFiles

            cd ..
        end % subj



    end % end ADD project
    if(contains(projects(p).name,'ADHD_impulsivity'))
        labels={'Fp1','Fp2','F3','Fz','F4','P3','P4','Oz'}; % as per the documentation in Neuromod folder
        Fs = 500; % TODO: CHECK/CONFIRM
        mkdir_if_not_exist([BIDS_DIR '/' projects(p).name '/'])


        dataPath = ['/autofs/vast/neuromod/tDCS_EEG/Projects/02_ADHD_impulsivity/EEG/'];
        cd(dataPath);
        subjFiles=dir('*ADHDi*');

        subjFiles = {subjFiles([subjFiles.isdir]).name};
        subjFiles = subjFiles(~ismember(subjFiles ,{'.','..'}))';
        for subj=1:size(subjFiles,1)%:-1:1 % ALWAYS GO BACKWARDS IN CASE THERE WERE RE-RUNS

            cd(subjFiles{subj});
            easyFiles=dir(['*.easy']);
            if(isempty(easyFiles))
                cd ..
                continue;
            end
            for eF=1:size(easyFiles,1)
                activeFile=easyFiles(eF).name

                if(contains(activeFile,'stim'))
                    warning('just a stim file, ignore')
                    continue;
                end
                % get the task name
                taskName = getTaskName(activeFile);

                % parse file name
                suStr=strsplit(activeFile,'_');

                % use the folder name
                folderStr=strsplit(subjFiles{subj},'_');
                visitNumber=str2num(folderStr{4});
                su =folderStr{3}; % suNumber
                suNumber=str2num(su);
                suDate = suStr{1};
                suGroup='';



                if(contains(activeFile,'pre'))
                    tptNumber=1;
                elseif(contains(activeFile,'post'))
                    tptNumber=2;
                end

                % sanity check
                if(isempty(visitNumber) | isempty(tptNumber))
                    error('missing session number of tpt number')
                end


                if(~exist([BIDS_DIR '/' projects(p).name '/sub-' num2str(suNumber) '/ses-' num2str(visitNumber) '/eeg/'...
                        'sub-' num2str(suNumber) '_ses-' num2str(visitNumber) '_task-' taskName '_run-' num2str(tptNumber) '_eeg.eeg'],'file'))

                    [data codes] = convert_easy(activeFile,labels);

                    [events, eventsTable] = process_codes(codes,studyName,taskName, Fs,suNumber,suGroup,visitNumber,tptNumber);
                    bids_root=[BIDS_DIR '/' projects(p).name '/'];
                    status = convert_bids(eventsTable, data, bids_root, taskName, suNumber,suGroup,visitNumber,tptNumber,suDate);
                    %                 error('s');

                    clear data;
                end
            end % easyFiles

            cd ..
        end % subj



    end % end ADD project

    if(contains(projects(p).name,'03_HC_HCa_FPN_attention_2018'))
        labels={'Fp1','Fp2','F3','Fz','F4','P3','P4','Oz'}; % as per the documentation in Neuromod folder
        Fs = 500; % TODO: CHECK/CONFIRM
        mkdir_if_not_exist([BIDS_DIR '/' projects(p).name '/'])


        dataPath = ['/autofs/vast/neuromod/tDCS_EEG/Projects/03_HC_HCa_FPN_attention_2018/EEG/'];
        cd(dataPath);
        subjFiles=dir('*.easy');

        for subj=1:size(subjFiles,1)%:-1:1 % ALWAYS GO BACKWARDS IN CASE THERE WERE RE-RUNS


            activeFile=subjFiles(subj).name

            if(contains(activeFile,'stim'))
                warning('just a stim file, ignore')
                continue;
            elseif(contains(activeFile,'test'))
                warning('just a test file, ignore')
                continue;
            end
            % get the task name
            taskName = getTaskName(activeFile);

            % parse file name
            suStr=strsplit(activeFile,'_');

            if(strcmp(suStr{3},'TransDx'))
                su =suStr{4}; % suNumber
                suNumber=str2num(su);
                suDate = suStr{1};
                visitNumber=(suStr{5});
                visitNumber=strrep(visitNumber,'V','');
                visitNumber=strrep(visitNumber,'notcompleted','');
                visitNumber=str2num(visitNumber);
            else

                su =suStr{5}; % suNumber
                suNumber=str2num(su);
                suDate = suStr{1};
                visitNumber=(suStr{6});
                visitNumber=strrep(visitNumber,'V','');
                visitNumber=strrep(visitNumber,'notcompleted','');
                visitNumber=str2num(visitNumber);
            end
            if(contains(activeFile,'pre'))
                tptNumber=1;
            elseif(contains(activeFile,'post'))
                tptNumber=2;
            end

            suGroup='';

            % sanity check
            if(isempty(visitNumber) | isempty(tptNumber))
                error('missing session number of tpt number')
            end


            if(~exist([BIDS_DIR '/' projects(p).name '/sub-' num2str(suNumber) '/ses-' num2str(visitNumber) '/eeg/'...
                    'sub-' num2str(suNumber) '_ses-' num2str(visitNumber) '_task-' taskName '_run-' num2str(tptNumber) '_eeg.eeg'],'file'))

                try
                    [data codes] = convert_easy(activeFile,labels);
                catch
                    continue;
                end

                [events, eventsTable] = process_codes(codes,studyName,taskName, Fs,suNumber,suGroup,visitNumber,tptNumber);
                bids_root=[BIDS_DIR '/' projects(p).name '/'];
                status = convert_bids(eventsTable, data, bids_root, taskName, suNumber,suGroup,visitNumber,tptNumber,suDate);
                %                 error('s');

                clear data;
            end

        end % subj



    end % end ADD project

    if(contains(projects(p).name,'HC_impulsivity'))
        labels={'Fp1','Fp2','F3','Fz','F4','P3','P4','Oz'}; % as per the documentation in Neuromod folder
        Fs = 500; % TODO: CHECK/CONFIRM
        mkdir_if_not_exist([BIDS_DIR '/' projects(p).name '/'])


        dataPath = ['/autofs/vast/neuromod/tDCS_EEG/Projects/04_HC_impulsivity_2017/EEG/'];
        cd(dataPath);
        subjFiles=dir('*.easy');

        for subj=size(subjFiles,1):-1:1 % ALWAYS GO BACKWARDS IN CASE THERE WERE RE-RUNS


            activeFile=subjFiles(subj).name

            if(contains(activeFile,'stim'))
                warning('just a stim file, ignore')
                continue;
            elseif(contains(activeFile,'test'))
                warning('just a test file, ignore')
                continue;
            end
            % get the task name
            taskName = getTaskName(activeFile);

            % parse file name
            suStr=strsplit(activeFile,'_');


            su =suStr{4}; % suNumber
            suNumber=str2num(su);
            suDate = suStr{1};
            visitNumber=str2num(suStr{5});

            if(contains(activeFile,'pre'))
                tptNumber=1;
            elseif(contains(activeFile,'pos'))
                tptNumber=2;
            end
            suGroup='';
            % sanity check
            if(isempty(visitNumber) | isempty(tptNumber))
                error('missing session number of tpt number')
            end


            if(~exist([BIDS_DIR '/' projects(p).name '/sub-' num2str(suNumber) '/ses-' num2str(visitNumber) '/eeg/'...
                    'sub-' num2str(suNumber) '_ses-' num2str(visitNumber) '_task-' taskName '_run-' num2str(tptNumber) '_eeg.eeg'],'file'))

                try
                    [data codes] = convert_easy(activeFile,labels);
                catch
                    continue;
                end

                [events, eventsTable] = process_codes(codes,studyName,taskName, Fs,suNumber,suGroup,visitNumber,tptNumber);
                bids_root=[BIDS_DIR '/' projects(p).name '/'];
                status = convert_bids(eventsTable, data, bids_root, taskName, suNumber,suGroup,visitNumber,tptNumber,suDate);
                %                 error('s');

                clear data;
            end

        end % subj



    end % end ADD project

    if(contains(projects(p).name,'HC_HD2'))
        labels={'Fp1','Fp2','F3','F4','Fz','P3','P4','Oz'}; % as per Laura's email on HD2 dataset
        Fs = 500; % TODO: CHECK/CONFIRM
        mkdir_if_not_exist([BIDS_DIR '/' projects(p).name '/'])

        for tasks={'flanker','nback','resting_state','msit_iaps'}


            dataPath = ['/autofs/vast/neuromod/tDCS_EEG/Projects/05_HC_HD2_Bifrontal_attention_2019/EEG/' tasks{1}];
            cd(dataPath);
            load('/autofs/vast/neuromod/tDCS_EEG/Projects/05_HC_HD2_Bifrontal_attention_2019/subjLevels.mat')
            load('/autofs/vast/neuromod/tDCS_EEG/Projects/05_HC_HD2_Bifrontal_attention_2019/analysis/randomization.mat')
            easyFiles=dir('*.easy');
            for subj=1:size(easyFiles,1)%:-1:1 % ALWAYS GO BACKWARDS IN CASE THERE WERE RE-RUNS


                activeFile=easyFiles(subj).name
                if(strcmp(activeFile,'20190531104631_flanker_pre_HD2_HCa_10_02_1.easy') || ...
                        strcmp(activeFile,'20190605095454_Flanker_post_HD2_HCa_10_03_2.easy'))
                    continue
                end
                su = activeFile([1:end-5]);
                su=strrep(su,'-','_');
                suDate = su(1:14);

                % cut everything before "HD_HCa"
                suStr=extractAfter(su,'HD');
                % take the subj number out of the filename
                suStr=strsplit(suStr,'_');
                suNumber = str2num(suStr{3});
                % get the visit number
                visitNumber = str2num(suStr{4});

                suGroup='';


                if(contains(activeFile,'pre'))
                    tptNumber=1;
                elseif(contains(activeFile,'post'))
                    tptNumber=2;
                end

                %
                %                 if(suNumber==10)
                %                     continue
                %                 end

                %                 suDetails=subjLevels(suNumber,:);
                %                 lvl=suDetails(2);
                %                 del=suDetails(3);
                %                 ons=suDetails(4);

                % get the task name
                taskName = getTaskName(activeFile);
                if(~exist([BIDS_DIR '/' projects(p).name '/1sub-' num2str(suNumber) '/ses-' num2str(visitNumber) '/eeg/'...
                        'sub-' num2str(suNumber) '_ses-' num2str(visitNumber) '_task-' taskName '_run-' num2str(tptNumber) '_eeg.eeg'],'file'))

                    [data codes] = convert_easy(activeFile,labels);

                    [events, eventsTable] = process_codes(codes,studyName,taskName, Fs,suNumber,suGroup,visitNumber,tptNumber);
                    bids_root=[BIDS_DIR '/' projects(p).name '/'];
                    status = convert_bids(eventsTable, data, bids_root, taskName, suNumber,suGroup,visitNumber,tptNumber,suDate);
                    %                 error('s');

                    clear data;
                end


            end % subj
        end % task


    end % HD2 project

    if(contains(projects(p).name,'SUD'))
        % SUD
        labels = { 'P8';'T8';'CP6';'FC6';'F8';  'F4';...
            'C4';   'P4';   'AF4';   'Fp2';  'Fp1';  'AF3';  'Fz';   'FC2'; ...
            'Cz';    'CP2';  'PO3';  'O1';   'Oz';   'O2';   'PO4';  'Pz';...
            'CP1';   'FC1';  'P3';   'C3';   'F3';   'F7';   'FC5';  'CP5';  'T7';   'P7'; };
        Fs = 500; % TODO: CHECK/CONFIRM
        dataPath = ['/mnt/aj123/EEG/0_Projects/8_SUD_TobaccoCannabis/EEG'];
        finalPath = ['/mnt/aj123/EEG/0_Projects/8_SUD_TobaccoCannabis/analysis/SST'];
        cd(dataPath)

        load([dataPath '/randomization_c.mat']);
        load([dataPath '/randomization_t.mat']);
        load([finalPath '/metadata.mat'])
        SUD_root = '/mnt/aj123/EEG/0_Projects/8_SUD_TobaccoCannabis';
        cd(SUD_root);
        cd("EEG/")
        for pop=1:2

            if(pop==1)
                population='cannabis';
                easyFiles=dir('*anna*.easy');
            else
                population='tobacco';
                easyFiles=dir('*bacco*.easy');
            end

            mkdir_if_not_exist([BIDS_DIR '/' projects(p).name '/' population])

            for subj=size(easyFiles,1):-1:1


                activeFile=easyFiles(subj).name
                su = activeFile([1:end-5]);
                suDate = su(1:14);
                % take the subj number out of the filename
                suStr=strsplit(activeFile,'_');
                suNumber = str2num(suStr{5});
                % get the visit number
                visitNumber = str2num(suStr{6}(2));

                % get the task name
                taskName = getTaskName(activeFile);
                suGroup=population;

                % get Pre/Post?
                if(contains(activeFile,'pre'))
                    tpt = 'pre';tptNumber=1;
                elseif(contains(activeFile,'post'))
                    tpt = 'post';tptNumber=2;
                else
                    error('no tpt found')
                end

                if(~exist([BIDS_DIR '/' projects(p).name '/' population '/sub-' num2str(suNumber) '/ses-' num2str(visitNumber) '/eeg/'...
                        'sub-' num2str(suNumber) '_ses-' num2str(visitNumber) '_task-' taskName '_run-' num2str(tptNumber) '_eeg.eeg'],'file'))

                    [data codes] = convert_easy(activeFile,labels);

                    [events, eventsTable] =process_codes(codes,studyName,taskName, Fs, suNumber,suGroup,visitNumber,tptNumber);
                    bids_root=[BIDS_DIR '/' projects(p).name '/' population '/'];
                    status = convert_bids(eventsTable, data, bids_root, taskName, suNumber,suGroup,visitNumber,tptNumber,suDate);


                    clear data;
                    clear rawdata;
                end



            end
        end

    end

    if(contains(projects(p).name,'FFOR'))

        FFOR_root='/autofs/vast/neuromod/tDCS_EEG/Projects/11_FFOR'; %'/mnt/aj123/EEG/0_Projects/11_FFOR';
        cd(FFOR_root);

        suDirs=dir('FFOR_*');
        for subj=1:size(suDirs,1)
            cd([FFOR_root '/' suDirs(subj).name '/EEG/'])
            eegFiles=dir('*.eeg');
            for ef=1:size(eegFiles,1)
                activeFile=eegFiles(ef).name
                su = activeFile([1:end-4]);

                % take the subj number out of the filename
                suStr=strsplit(activeFile,'_');
                % set subj number
                suNumber = str2num(suStr{2});


                % set the session number
                if(contains(activeFile,'ay1'))
                    tpt = 'Day1';visitNumber=1;
                elseif(contains(activeFile,'ay2'))
                    tpt = 'Day2';visitNumber=2;
                elseif(contains(activeFile,'SST'))
                    tpt='Day1',visitNumber=1;
                else
                    error('no day info found')
                end

                % set the task name
                taskName = getTaskName(activeFile);

                % set a group name
                suGroup='';

                % set timepoint number
                tptNumber=1;

                % find out if there are 2 parts to the recording (in case
                % of interruption)
                % in this case, the pattern should be "Block2_2" for eg.
                appendFlag=0;
                if(contains(activeFile,'_1_'))
                    taskName=[taskName '_1'];
                    appendFlag=1;
                end
                if(contains(activeFile,'_2_'))
                    taskName=[taskName '_2'];
                    appendFlag=1;
                end

                if(~exist([BIDS_DIR '/' projects(p).name '/sub-' num2str(suNumber) '/ses-' num2str(visitNumber) '/eeg/'...
                        'sub-' num2str(suNumber) '_ses-' num2str(visitNumber) '_task-' getTaskName(activeFile) '_run-' num2str(1) '_eeg.eeg'],'file'))

                    cfg=[];
                    cfg.headerfile     = [su '.vhdr'];
                    cfg.datafile = [su '.eeg'];
                    hdr        = ft_read_header(cfg.headerfile);
                    Fs = hdr.Fs;

                    % set date of recordings by the timestamp in the
                    % vmrk file
                    vmrkFile= [su '.vmrk'];
                    [s o]=system(['awk "NR==12{ print; exit }" "' vmrkFile '"']);
                    o=strsplit(o,',');suDate=o{5};
                    suDate=datetime(suDate,'InputFormat','yyyyMMddHHmmssSSSSSS','Format','yyyyMMddhhmmss');

                    suDate=char(suDate);
                    if(length(suDate)~=14)
                        error('check date!')
                    end
                    orighdr=hdr;


                    event      = ft_read_event(cfg.headerfile);
                    markers  = (find(strcmp({event.type},'Stimulus')));

                    data = ft_preprocessing(cfg);
                    codes={event(markers).value};
                    for c=1:length(codes)
                        codes(c)=strrep(codes(c),'S','');
                        codes(c)=strrep(codes(c),' ','');
                    end
                    codes=cellfun(@str2num,codes);
                    triggers=[codes' cell2mat({event(markers).sample})'];


                    % for sub 202 to sub 204, need to fix eeg markers!
                    if(suNumber>=202 && suNumber<=204 && contains(activeFile,'Block'))

                        % figured out the general rule:
                        % the "real" trigger is the one which has a
                        % specific delay in the range of 200msec between
                        % its onset and a subsequent trigger code  OR
                        % if its a REAL SHOCK (trig 3), its in the range of
                        % 500 msec
                        % this can be calculated by taking the diff of the
                        % onset time of all triggers and finding the
                        % triggers which are around 195-205/495-505 msec preceding
                        % a trigger code of anything else

                        % calculate latency diffs of triggers
                        triggers(:,3)=[diff(triggers(:,2));0];
                        rmList=[];
                        for it=1:size(triggers,1)
                            if(~ismember(triggers(it,3),[195:205 495:505]))
                                rmList(end+1)=it;
                            end
                        end
                        triggers(rmList,:)=[];

                    end

                    % for sub 202, 203, 204, 205, and 206 Block2 triggers were not
                    % labelled correctly to differentiate CS+ with CS+
                    % w/shock
                    if(suNumber>=202 && suNumber<=207 && contains(activeFile,'Block2'))
                        triggers=[codes' cell2mat({event(markers).sample})'];
                        %                         error('s');

                        % go case by case
                        if(suNumber==202)

                            % figured out the general rule:

                            % remove all 5s categorically first
                            triggers(triggers(:,1)==5,:)=[];

                            % now, find any "4" codes, and remove the "1" before it,
                            % if exists

                            % find any "2" codes and remove "1" and "3"
                            % preceding, and "3" and "1" immediately after
                            rmList=[];
                            for it=1:size(triggers,1)

                                if(triggers(it,1)==4 && triggers(it-1,1)==1)
                                    rmList(end+1)=it-1;
                                end
                                if(triggers(it,1)==2 && triggers(it-1,1)==3 && triggers(it-2,1)==1)
                                    rmList(end+1)=it-1;
                                    rmList(end+1)=it-2;
                                end
                                if(triggers(it,1)==2 && triggers(it+1,1)==3 && triggers(it+2,1)==1)
                                    rmList(end+1)=it+1;
                                    rmList(end+1)=it+2;
                                end


                            end
                            % remove last trig too
                            rmList(end+1)=size(triggers,1);
                            triggers(rmList,:)=[];
                        end
                        if(suNumber==203 || suNumber==204)
                            %                             error('s')
                            triggers(:,3)=[diff(triggers(:,2));0];

                            % calculate latency diffs of triggers
                            rmList=[];
                            for it=1:size(triggers,1)
                                if(~ismember(triggers(it,3),[195:205 495:505]))
                                    rmList(end+1)=it;
                                end
                            end
                            triggers(rmList,:)=[];

                            % at this point, all "3" trials are good
                            % but need to fix 2/4s

                            % first, all trials preceding 3 should be 1
                            % by rule
                            chList=[];
                            for it=1:size(triggers,1)
                                if(triggers(it,1)==3 && triggers(it-1,1)~=1)
                                    chList(end+1)=it-1;
                                end
                            end
                            triggers(chList,1)=1;

                            % remaining trials need to be swapped
                            % between 2/4
                            triggers(triggers(:,1)==2)=44;
                            triggers(triggers(:,1)==4)=2;
                            triggers(triggers(:,1)==44)=4;
                        end

                        if(suNumber==205 || suNumber==206 || suNumber==207)

                            %                             error('s')
                            % for this subject, everything is ok with
                            % triggers, but need to manually recode 2->1s
                            chList=[];
                            for it=1:size(triggers,1)
                                if(triggers(it,1)==3 && triggers(it-1,1)~=1)
                                    chList(end+1)=it-1;
                                end
                            end
                            triggers(chList,1)=1;

                        end

                        triggers(triggers(:,1)==8,:)=[];


                    end



                    [events, eventsTableTSV] = process_task(studyName,getTaskName(activeFile),triggers,Fs,suNumber,suGroup,visitNumber,tptNumber);

                    bids_root=[BIDS_DIR '/' projects(p).name '/'];
                    status = convert_bids(eventsTableTSV, data, bids_root, taskName, suNumber,suGroup,visitNumber,tptNumber,suDate);
                    %                 error('s');

                    clear data;
                else
                    % if the final dataset is there, no need to append
                    % anything
                    appendFlag=0;
                end
                if(~exist([BIDS_DIR '/' projects(p).name '/sub-' num2str(suNumber) '/ses-' num2str(visitNumber) '/eeg/'...
                        'sub-' num2str(suNumber) '_ses-' num2str(visitNumber) '_task-' getTaskName(activeFile) '_run-' num2str(1) '_electrodes.tsv'],'file'))
                    % post-conversion: copy full header information
                    % over from bv2bids command
                    hdr_path=[su '.vhdr'];
                    taskName = getTaskName(activeFile);
                    % ensure the marker file is the same name as
                    % the base name
                    cmd=sprintf(['sed -i "s/MarkerFile=.*/MarkerFile=%s/g" ' hdr_path],...
                        [hdr_path(1:end-4) 'vmrk']);
                    [s o]=system(cmd)
                    cmd=sprintf(['sed -i "s/DataFile=.*/DataFile=%s/g" ' hdr_path],...
                        [hdr_path(1:end-4) 'eeg']);
                    [ s o]= system(cmd)

                    bv2bids_exec='/autofs/homes/007/aj123/.dotnet/bvtools/BVTools-master/out/bin/FileFormats.BrainVisionToBidsConverterCLI/Release/net7.0/linux-x64/publish/BV2BIDS';
                    % build command
                    cmd=sprintf([bv2bids_exec ' -hdr %s -tsk %s -ses %s -sub %s -dst %s'], ...
                        hdr_path, taskName, num2str(visitNumber), num2str(suNumber), '/scratch/bv2bids');

                    [s o]= system(cmd)
                    % copy over the electrodes tsv file
                    copyfile(['/scratch/bv2bids/sub-' num2str(suNumber) '/ses-' num2str(visitNumber) '/eeg/sub-' num2str(suNumber) '_ses-' num2str(visitNumber) '_task-' taskName '_electrodes.tsv'],...
                        [BIDS_DIR '/' projects(p).name '/sub-' num2str(suNumber) '/ses-' num2str(visitNumber) '/eeg/'...
                        'sub-' num2str(suNumber) '_ses-' num2str(visitNumber) '_task-' taskName '_run-' num2str(tptNumber) '_electrodes.tsv'  ])
                    % copy over the header vhdr file again (has
                    % more info!
                    copyfile(['/scratch/bv2bids/sub-' num2str(suNumber) '/ses-' num2str(visitNumber) '/eeg/sub-' num2str(suNumber) '_ses-' num2str(visitNumber) '_task-' taskName '_eeg.vhdr'],...
                        [BIDS_DIR '/' projects(p).name '/sub-' num2str(suNumber) '/ses-' num2str(visitNumber) '/eeg/'...
                        'sub-' num2str(suNumber) '_ses-' num2str(visitNumber) '_task-' taskName '_run-' num2str(tptNumber) '_eeg.vhdr'  ])
                    % copy over the new vmrk
                    copyfile(['/scratch/bv2bids/sub-' num2str(suNumber) '/ses-' num2str(visitNumber) '/eeg/sub-' num2str(suNumber) '_ses-' num2str(visitNumber) '_task-' taskName '_eeg.vmrk'],...
                        [BIDS_DIR '/' projects(p).name '/sub-' num2str(suNumber) '/ses-' num2str(visitNumber) '/eeg/'...
                        'sub-' num2str(suNumber) '_ses-' num2str(visitNumber) '_task-' taskName '_run-' num2str(tptNumber) '_eeg.vmrk'  ])
                    % copy over coordsystem (just in case)
                    copyfile(['/scratch/bv2bids/sub-' num2str(suNumber) '/ses-' num2str(visitNumber) '/eeg/sub-' num2str(suNumber) '_ses-' num2str(visitNumber) '_task-' taskName '_coordsystem.json'],...
                        [BIDS_DIR '/' projects(p).name '/sub-' num2str(suNumber) '/ses-' num2str(visitNumber) '/eeg/'...
                        'sub-' num2str(suNumber) '_ses-' num2str(visitNumber) '_task-' taskName '_run-' num2str(tptNumber) '_coordsystem.json'  ])
                    % full data!
                    % disp('eeg cp..')
                    %   copyfile(['/scratch/bv2bids/sub-' num2str(suNumber) '/ses-' num2str(visitNumber) '/eeg/sub-' num2str(suNumber) '_ses-' num2str(visitNumber) '_task-' taskName '_eeg.eeg'],...
                    %     [BIDS_DIR '/' projects(p).name '/sub-' num2str(suNumber) '/ses-' num2str(visitNumber) '/eeg/'...
                    %      'sub-' num2str(suNumber) '_ses-' num2str(visitNumber) '_task-' taskName '_run-' num2str(tptNumber) '_eeg.eeg'  ])
                    %
                    copyfile(activeFile,...
                        [BIDS_DIR '/' projects(p).name '/sub-' num2str(suNumber) '/ses-' num2str(visitNumber) '/eeg/'...
                        'sub-' num2str(suNumber) '_ses-' num2str(visitNumber) '_task-' taskName '_run-' num2str(tptNumber) '_eeg.eeg'  ])


                    system('rm -rf /scratch/bv2bids')
                    disp('BV2BIDS Files copied!')
                end

                % if any files need to be appended together, do the
                % stitching, but only after both parts are ready
                if(appendFlag)
                    taskName=getTaskName(activeFile);
                    part1=[BIDS_DIR '/' projects(p).name '/sub-' num2str(suNumber) '/ses-' num2str(visitNumber) '/eeg/'...
                        'sub-' num2str(suNumber) '_ses-' num2str(visitNumber) '_task-' taskName '_1_run-' num2str(1) '_eeg.eeg'];
                    part2=[BIDS_DIR '/' projects(p).name '/sub-' num2str(suNumber) '/ses-' num2str(visitNumber) '/eeg/'...
                        'sub-' num2str(suNumber) '_ses-' num2str(visitNumber) '_task-' taskName '_2_run-' num2str(1) '_eeg.eeg'];
                    if(exist(part1,'file') && exist(part2,'file'))
                        %                         error('s');
                        warning('beginning file concatenation process')
                        hdr1 = ft_read_header([part1(1:end-3) 'vhdr' ]);
                        dat1 = ft_read_data(part1);
                        evt1 = ft_read_event([part1]);

                        hdr2 = ft_read_header([part2(1:end-3) 'vhdr' ]);
                        dat2 = ft_read_data(part2);
                        evt2 = ft_read_event(part2);

                        hdr = hdr1;               % the headers are assumed to be the same wrt samping rate and channel names
                        dat = cat(2, dat1, dat2);  % concatenate the data along the 2nd dimension

                        cfg=[];
                        data=[];
                        data.label     = hdr1.label;
                        data.trial{1} = dat;
                        data.fsample=hdr1.Fs;
                        data.time{1} = (1:size(dat, 2)) / hdr1.Fs;
                        ftdata=ft_preprocessing(cfg,data);

                        Fs = hdr.Fs;
                        % shift the sample of the events or triggers in the second block
                        for i=1:length(evt2)
                            evt2(i).sample = evt2(i).sample + hdr1.nSamples;
                        end

                        evt = [evt1, evt2]; % concatenate the events
                        % convert to an eventsTableTSV
                        eventsTSV=struct2table(evt);
                        % should be following order:
                        % onset duration sample trial_type value
                        eventsTSV=[eventsTSV(:,6) eventsTSV(:,5) eventsTSV(:,3) eventsTSV(:,1) eventsTSV(:,2)];
                        eventsTSV=renamevars(eventsTSV,["type","timestamp"],["trial_type","onset"]);
                        disp('exporting appended dataset....')
                        bids_root=[BIDS_DIR '/' projects(p).name '/'];
                        status = convert_bids(eventsTSV, ftdata, bids_root, getTaskName(activeFile), suNumber,suGroup,visitNumber,tptNumber,suDate);
                        clear ftdata; clear data;

                    end
                end



            end
        end
    end

    if(contains(projects(p).name,'PASC'))

        PASC_root='/autofs/vast/neuromod/tDCS_EEG/Projects/12_PASC'; %'/mnt/aj123/EEG/0_Projects/11_FFOR';
        cd(PASC_root);

        suDirs=dir('PASC_*');

        for subj=1:size(suDirs,1)
            cd([PASC_root '/' suDirs(subj).name '/'])
            subfolders=dir('*');subfolders([1:2])=[];
            for sub=1:size(subfolders,1)
                cd(subfolders(sub).name)
                eegFiles=dir('*.eeg');
                for ef=1:size(eegFiles,1)
                    activeFile=eegFiles(ef).name
                    su = activeFile([1:end-4]);

                    % take the subj number out of the filename
                    suStr=strsplit(activeFile,'_');
                    % set subj number
                    suNumber = str2num(suStr{3});

                    % set the session/visit number
                    ses = suStr{4};ses=strrep(ses,'V','');visitNumber=str2num(ses);

                    % set the task name
                    taskName = getTaskName(activeFile);

                    % set a group name
                    suGroup='';

                    % set timepoint/run number
                    tptNumber=1;



                    if(~exist([BIDS_DIR '/' projects(p).name '/sub-' num2str(suNumber) '/ses-' num2str(visitNumber) '/eeg/'...
                            'sub-' num2str(suNumber) '_ses-' num2str(visitNumber) '_task-' taskName '_run-' num2str(tptNumber) '_eeg.eeg'],'file'))

                        cfg=[];
                        cfg.headerfile     = [su '.vhdr'];
                        cfg.datafile = [su '.eeg'];
                        hdr        = ft_read_header(cfg.headerfile);
                        Fs = hdr.Fs;

                        % set date of recordings by the timestamp in the
                        % vmrk file
                        vmrkFile= [su '.vmrk'];
                        [s o]=system(['awk "NR==12{ print; exit }" "' vmrkFile '"']);
                        o=strsplit(o,',');suDate=o{5};
                        suDate=datetime(suDate,'InputFormat','yyyyMMddHHmmssSSSSSS','Format','yyyyMMddhhmmss');

                        suDate=char(suDate);
                        if(length(suDate)~=14)
                            error('check date!')
                        end
                        orighdr=hdr;
                        event      = ft_read_event(cfg.headerfile);
                        markers  = (find(strcmp({event.type},'Stimulus')));

                        data = ft_preprocessing(cfg);
                        codes={event(markers).value};
                        for c=1:length(codes)
                            codes(c)=strrep(codes(c),'S','');
                            codes(c)=strrep(codes(c),' ','');
                        end
                        codes=cellfun(@str2num,codes);
                        triggers=[codes' cell2mat({event(markers).sample})'];

                        %                         try
                        [events, eventsTableTSV] = process_task(studyName, taskName,triggers,Fs,suNumber,suGroup,visitNumber,tptNumber);
                        %                         catch
                        %                             warning('no triggers found for this task, skipping for now');
                        %                             continue;
                        %                         end
                        bids_root=[BIDS_DIR '/' projects(p).name '/'];
                        status = convert_bids(eventsTableTSV, data, bids_root, taskName, suNumber,suGroup,visitNumber,tptNumber,suDate);
                        %                 error('s');

                        clear data;
                    end %   if check

                    if(~exist([BIDS_DIR '/' projects(p).name '/sub-' num2str(suNumber) '/ses-' num2str(visitNumber) '/eeg/'...
                            'sub-' num2str(suNumber) '_ses-' num2str(visitNumber) '_task-' taskName '_run-' num2str(tptNumber) '_electrodes.tsv'],'file'))
                        % post-conversion: copy full header information
                        % over from bv2bids command
                        hdr_path=[su '.vhdr'];
                        % ensure the marker file is the same name as
                        % the base name
                        cmd=sprintf(['sed -i "s/MarkerFile=.*/MarkerFile=%s/g" ' hdr_path],...
                            [hdr_path(1:end-4) 'vmrk']);
                        [s o]=system(cmd)
                        cmd=sprintf(['sed -i "s/DataFile=.*/DataFile=%s/g" ' hdr_path],...
                            [hdr_path(1:end-4) 'eeg']);
                        [ s o]= system(cmd)

                        bv2bids_exec='/autofs/homes/007/aj123/.dotnet/bvtools/BVTools-master/out/bin/FileFormats.BrainVisionToBidsConverterCLI/Release/net7.0/linux-x64/publish/BV2BIDS';
                        % build command
                        cmd=sprintf([bv2bids_exec ' -hdr %s -tsk %s -ses %s -sub %s -dst %s'], ...
                            hdr_path, taskName, num2str(visitNumber), num2str(suNumber), '/scratch/bv2bids');

                        [s o]= system(cmd)
                        % copy over the electrodes tsv file
                        copyfile(['/scratch/bv2bids/sub-' num2str(suNumber) '/ses-' num2str(visitNumber) '/eeg/sub-' num2str(suNumber) '_ses-' num2str(visitNumber) '_task-' taskName '_electrodes.tsv'],...
                            [BIDS_DIR '/' projects(p).name '/sub-' num2str(suNumber) '/ses-' num2str(visitNumber) '/eeg/'...
                            'sub-' num2str(suNumber) '_ses-' num2str(visitNumber) '_task-' taskName '_run-' num2str(tptNumber) '_electrodes.tsv'  ])
                        % copy over the header vhdr file again (has
                        % more info!
                        copyfile(['/scratch/bv2bids/sub-' num2str(suNumber) '/ses-' num2str(visitNumber) '/eeg/sub-' num2str(suNumber) '_ses-' num2str(visitNumber) '_task-' taskName '_eeg.vhdr'],...
                            [BIDS_DIR '/' projects(p).name '/sub-' num2str(suNumber) '/ses-' num2str(visitNumber) '/eeg/'...
                            'sub-' num2str(suNumber) '_ses-' num2str(visitNumber) '_task-' taskName '_run-' num2str(tptNumber) '_eeg.vhdr'  ])
                        % copy over the new vmrk
                        copyfile(['/scratch/bv2bids/sub-' num2str(suNumber) '/ses-' num2str(visitNumber) '/eeg/sub-' num2str(suNumber) '_ses-' num2str(visitNumber) '_task-' taskName '_eeg.vmrk'],...
                            [BIDS_DIR '/' projects(p).name '/sub-' num2str(suNumber) '/ses-' num2str(visitNumber) '/eeg/'...
                            'sub-' num2str(suNumber) '_ses-' num2str(visitNumber) '_task-' taskName '_run-' num2str(tptNumber) '_eeg.vmrk'  ])
                        % copy over coordsystem (just in case)
                        copyfile(['/scratch/bv2bids/sub-' num2str(suNumber) '/ses-' num2str(visitNumber) '/eeg/sub-' num2str(suNumber) '_ses-' num2str(visitNumber) '_task-' taskName '_coordsystem.json'],...
                            [BIDS_DIR '/' projects(p).name '/sub-' num2str(suNumber) '/ses-' num2str(visitNumber) '/eeg/'...
                            'sub-' num2str(suNumber) '_ses-' num2str(visitNumber) '_task-' taskName '_run-' num2str(tptNumber) '_coordsystem.json'  ])
                        % full data!

                        copyfile(['/scratch/bv2bids/sub-' num2str(suNumber) '/ses-' num2str(visitNumber) '/eeg/sub-' num2str(suNumber) '_ses-' num2str(visitNumber) '_task-' taskName '_eeg.eeg'],...
                            [BIDS_DIR '/' projects(p).name '/sub-' num2str(suNumber) '/ses-' num2str(visitNumber) '/eeg/'...
                            'sub-' num2str(suNumber) '_ses-' num2str(visitNumber) '_task-' taskName '_run-' num2str(tptNumber) '_eeg.eeg'  ])

                        system('rm -rf /scratch/bv2bids')
                        disp('BV2BIDS Files copied!')
                    end

                end %   eegfile
                cd ..
            end %   subfolder
        end % subj
    end % project

    if(contains(projects(p).name,'HAT'))

        HAT_root='/autofs/vast/neuromod/tDCS_EEG/Projects/13_HAT'; %'/mnt/aj123/EEG/0_Projects/11_FFOR';
        cd(HAT_root);

        suDirs=dir('HAT_*');

        for subj=1:size(suDirs,1)
            cd([HAT_root '/' suDirs(subj).name '/'])
            subfolders=dir('*');subfolders([1:2])=[];
            for sub=1:size(subfolders,1)
                cd(subfolders(sub).name)
                eegFiles=dir('*.eeg');
                for ef=1:size(eegFiles,1)
                    activeFile=eegFiles(ef).name
                    su = activeFile([1:end-4]);

                    % take the subj number out of the filename
                    suStr=strsplit(activeFile,'_');
                    % set subj number
                    suNumber = str2num(suStr{2});

                    % set the session/visit number
                    ses = suStr{3};ses=strrep(ses,'V','');visitNumber=str2num(ses);

                    % set the task name
                    taskName = getTaskName(activeFile);

                    % set a group name
                    suGroup='';

                    % set timepoint/run number
                    tptNumber=1;



                    if(~exist([BIDS_DIR '/' projects(p).name '/sub-' num2str(suNumber) '/ses-' num2str(visitNumber) '/eeg/'...
                            'sub-' num2str(suNumber) '_ses-' num2str(visitNumber) '_task-' taskName '_run-' num2str(tptNumber) '_eeg.eeg'],'file'))

                        cfg=[];
                        cfg.headerfile     = [su '.vhdr'];
                        cfg.datafile = [su '.eeg'];
                        hdr        = ft_read_header(cfg.headerfile,'headerformat','brainvision_vhdr');

                        Fs = hdr.Fs;

                        % set date of recordings by the timestamp in the
                        % vmrk file
                        vmrkFile= [su '.vmrk'];
                        [s o]=system(['awk "NR==12{ print; exit }" "' vmrkFile '"']);
                        o=strsplit(o,',');suDate=o{5};
                        suDate=datetime(suDate,'InputFormat','yyyyMMddHHmmssSSSSSS','Format','yyyyMMddhhmmss');

                        suDate=char(suDate);
                        if(length(suDate)~=14)
                            error('check date!')
                        end
                        orighdr=hdr;
                        event      = ft_read_event(cfg.headerfile);
                        markers  = (find(strcmp({event.type},'Stimulus')));

                        data = ft_preprocessing(cfg);
                        codes={event(markers).value};
                        for c=1:length(codes)
                            codes(c)=strrep(codes(c),'S','');
                            codes(c)=strrep(codes(c),' ','');
                        end
                        codes=cellfun(@str2num,codes);
                        triggers=[codes' cell2mat({event(markers).sample})'];

                        %                         try
                        [events, eventsTableTSV] = process_task(studyName,taskName,triggers,Fs,suNumber,suGroup,visitNumber,tptNumber);
                        %                         catch
                        %                             warning('no triggers found for this task, skipping for now');
                        %                             continue;
                        %                         end
                        % embed type of amplifier (active vs passive)
                        bids_root=[BIDS_DIR '/' projects(p).name '/'];
                        status = convert_bids(eventsTableTSV, data, bids_root, taskName, suNumber,suGroup,visitNumber,tptNumber,suDate);
                        %                 error('s');


                        clear data;
                    end %   if check
                    if(~exist([BIDS_DIR '/' projects(p).name '/sub-' num2str(suNumber) '/ses-' num2str(visitNumber) '/eeg/'...
                            'sub-' num2str(suNumber) '_ses-' num2str(visitNumber) '_task-' taskName '_run-' num2str(tptNumber) '_electrodes.tsv'],'file'))
                        % post-conversion: copy full header information
                        % over from bv2bids command
                        hdr_path=[su '.vhdr'];
                        % ensure the marker file is the same name as
                        % the base name
                        cmd=sprintf(['sed -i "s/MarkerFile=.*/MarkerFile=%s/g" ' hdr_path],...
                            [hdr_path(1:end-4) 'vmrk']);
                        [s o]=system(cmd)
                        cmd=sprintf(['sed -i "s/DataFile=.*/DataFile=%s/g" ' hdr_path],...
                            [hdr_path(1:end-4) 'eeg']);
                        [ s o]= system(cmd)

                        bv2bids_exec='/autofs/homes/007/aj123/.dotnet/bvtools/BVTools-master/out/bin/FileFormats.BrainVisionToBidsConverterCLI/Release/net7.0/linux-x64/publish/BV2BIDS';
                        % build command
                        cmd=sprintf([bv2bids_exec ' -hdr %s -tsk %s -ses %s -sub %s -dst %s'], ...
                            hdr_path, taskName, num2str(visitNumber), num2str(suNumber), '/scratch/bv2bids');

                        [s o]= system(cmd)
                        % copy over the electrodes tsv file
                        copyfile(['/scratch/bv2bids/sub-' num2str(suNumber) '/ses-' num2str(visitNumber) '/eeg/sub-' num2str(suNumber) '_ses-' num2str(visitNumber) '_task-' taskName '_electrodes.tsv'],...
                            [BIDS_DIR '/' projects(p).name '/sub-' num2str(suNumber) '/ses-' num2str(visitNumber) '/eeg/'...
                            'sub-' num2str(suNumber) '_ses-' num2str(visitNumber) '_task-' taskName '_run-' num2str(tptNumber) '_electrodes.tsv'  ])
                        % copy over the header vhdr file again (has
                        % more info!
                        copyfile(['/scratch/bv2bids/sub-' num2str(suNumber) '/ses-' num2str(visitNumber) '/eeg/sub-' num2str(suNumber) '_ses-' num2str(visitNumber) '_task-' taskName '_eeg.vhdr'],...
                            [BIDS_DIR '/' projects(p).name '/sub-' num2str(suNumber) '/ses-' num2str(visitNumber) '/eeg/'...
                            'sub-' num2str(suNumber) '_ses-' num2str(visitNumber) '_task-' taskName '_run-' num2str(tptNumber) '_eeg.vhdr'  ])
                        % copy over the new vmrk
                        copyfile(['/scratch/bv2bids/sub-' num2str(suNumber) '/ses-' num2str(visitNumber) '/eeg/sub-' num2str(suNumber) '_ses-' num2str(visitNumber) '_task-' taskName '_eeg.vmrk'],...
                            [BIDS_DIR '/' projects(p).name '/sub-' num2str(suNumber) '/ses-' num2str(visitNumber) '/eeg/'...
                            'sub-' num2str(suNumber) '_ses-' num2str(visitNumber) '_task-' taskName '_run-' num2str(tptNumber) '_eeg.vmrk'  ])
                        % copy over coordsystem (just in case)
                        copyfile(['/scratch/bv2bids/sub-' num2str(suNumber) '/ses-' num2str(visitNumber) '/eeg/sub-' num2str(suNumber) '_ses-' num2str(visitNumber) '_task-' taskName '_coordsystem.json'],...
                            [BIDS_DIR '/' projects(p).name '/sub-' num2str(suNumber) '/ses-' num2str(visitNumber) '/eeg/'...
                            'sub-' num2str(suNumber) '_ses-' num2str(visitNumber) '_task-' taskName '_run-' num2str(tptNumber) '_coordsystem.json'  ])
                        % full data!

                        copyfile(['/scratch/bv2bids/sub-' num2str(suNumber) '/ses-' num2str(visitNumber) '/eeg/sub-' num2str(suNumber) '_ses-' num2str(visitNumber) '_task-' taskName '_eeg.eeg'],...
                            [BIDS_DIR '/' projects(p).name '/sub-' num2str(suNumber) '/ses-' num2str(visitNumber) '/eeg/'...
                            'sub-' num2str(suNumber) '_ses-' num2str(visitNumber) '_task-' taskName '_run-' num2str(tptNumber) '_eeg.eeg'  ])


                        system('rm -rf /scratch/bv2bids')
                        disp('BV2BIDS Files copied!')

                    end
                end %   eegfile
                cd ..
            end %   subfolder
        end % subj
    end % project

    if(contains(projects(p).name,'DSRF'))

        labels = { 'P8'; 'T8'; 'CP6';'FC6';'F8';  'F4';...
            'C4';   'P4';   'AF4';   'Fp2';  'Fp1';  'AF3';  'Fz';   'FC2'; ...
            'Cz';    'CP2';  'PO3';  'O1';   'Oz';   'O2';   'PO4';  'Pz';...
            'CP1';   'FC1';  'P3';   'C3';   'F3';   'F7';   'FC5';  'CP5';  'T7';   'P7'; };
        Fs = 500; % sampling rate

        dataPath = ['/autofs/vast/neuromod/tDCS_EEG/Projects/14_DSRF'];

        cd(dataPath)

        subjFolders=dir('1*');
        for subj=1:size(subjFolders,1)

            cd(subjFolders(subj).name);

            su=subjFolders(subj).name; % the subj id

            sessionFolders=dir('1*');
            for sess=1:size(sessionFolders,1)
                cd(sessionFolders(sess).name);


                if(contains(sessionFolders(sess).name,'PRE'))
                    tpt = 'pre';visitNumber=1;
                elseif(contains(sessionFolders(sess).name,'POST'))
                    tpt = 'post';visitNumber=2;
                elseif(contains(sessionFolders(sess).name,'FU'))
                    tpt= 'fu';visitNumber=3;
                else
                    error('no tpt identified')
                end

                easyFiles=dir('*.easy');

                for file=1:size(easyFiles,1)

                    activeFile=easyFiles(file).name


                    suDate = activeFile(1:14);

                    tptNumber= 1;

                    % get the task name
                    taskName = getTaskName(activeFile);

                    % set a group name
                    suGroup='';

                    if(~exist([BIDS_DIR '/' projects(p).name '/sub-' su '/ses-' num2str(visitNumber) '/eeg/'...
                            'sub-' su '_ses-' num2str(visitNumber) '_task-' taskName '_run-' num2str(tptNumber) '_eeg.eeg'],'file'))

                        [data codes] = convert_easy(activeFile,labels);

                        [events, eventsTable] =process_codes(codes,studyName,taskName, Fs, su, suGroup, visitNumber,tptNumber);
                        bids_root=[BIDS_DIR '/' projects(p).name '/'];
                        status = convert_bids(eventsTable, data, bids_root, taskName, su, suGroup, visitNumber,tptNumber,suDate);


                        clear data;
                        clear rawdata;
                    end

                end

                cd ..
            end


            cd ..
        end


    end

    % ENGINE DATA SET
    if(contains(projects(p).name,'ENGINE'))

       addpath '/autofs/vast/neuromod/tDCS_EEG/Projects/16_ENGINE'      
       ENGINE_root = ['/autofs/vast/neuromod/tDCS_EEG/Projects/16_ENGINE'];
       cd(ENGINE_root)
        
        for pop=1:2
            if(pop==1)      % BIDS conversion for Healthy Control
                population='HC';
                cd([ENGINE_root '/' population ]);
                recordingFolders=dir('*');                   
                mkdir_if_not_exist([BIDS_DIR '/' projects(p).name '/' population])

                for subj=size(recordingFolders,1):-1:3
                
                    cd([ENGINE_root '/' population ]);
                    recordingFolders=dir('*');
                    if ~isfolder(recordingFolders(subj).name)   % We make sure it only goes
                        continue;                               % through the folders, not
                    end                                         % other files.
                    cd(recordingFolders(subj).name);
                    cd('NKT')
                    cd('EEG2100')
                    Path=[ENGINE_root '/' population '/' recordingFolders(subj).name '/NKT/EEG2100/'];
                    eegFiles=dir('*.EEG');      % Recording the eeg file in a variable

                    FileName=eegFiles(1).name;  % Recording file name
                
                    DateReading = eegFiles().date;      % Read date from the data file
                    DateVector = datevec(DateReading);  % Make it a vector
                    year = num2str(DateVector(1), '%04d');  % Hour with 4 digits
                    month = num2str(DateVector(2), '%02d'); % Month with 2 digits
                    day = num2str(DateVector(3), '%02d');   % Day with 2 digits
                    hour = num2str(DateVector(4), '%02d');  % Hour with 2 digits
                    minute = num2str(DateVector(5), '%02d');% Minute with 2 digits
                    second = num2str(DateVector(6), '%02d');% Seconds with 2 digits
                        
                    % Build the string in the 'yyyymmddHHMMSS' format
                    suDate = [year month day hour minute second];


                    % take the subj number out of the filename
                    
                    su = recordingFolders(subj).name;
                    suNumber = str2num(su); 
                    
                    % get the visit number
                    visitNumber = 1;

                    % get the task name
                    taskName = 'RestingStateEyesClosed';%getTaskName(activeFile);
                    suGroup=population;

                    tptNumber=1;
                
                    if(~exist([BIDS_DIR '/' projects(p).name '/' population '/sub-' num2str(suNumber) '/ses-' num2str(visitNumber) '/eeg/'...
                         'sub-' num2str(suNumber) '_ses-' num2str(visitNumber) '_task-' taskName '_run-' num2str(tptNumber) '_eeg.eeg'],'file'))

                        % Convert discovery files to be readable 
                        [data] = convert_discovery(Path,FileName);
                        eventsTable=table;

                        % Convert to BIDS
                        bids_root=[BIDS_DIR '/' projects(p).name '/' population '/'];
                        status = convert_bids(eventsTable, data, bids_root, taskName, suNumber,suGroup,visitNumber,tptNumber,suDate);

                         clear data;
                         clear rawdata;
                    end
                end
           
            else
                population='MDD';           % BIDS conversion for Major Depressive Disorder
                cd([ENGINE_root '/' population ]);
                recordingFolders=dir('*');                           
                mkdir_if_not_exist([BIDS_DIR '/' projects(p).name '/' population])

                for subj=size(recordingFolders,1):-1:3 % Go through all the subject folders
                    cd([ENGINE_root '/' population ]);                     
                    cd(recordingFolders(subj).name);
                    recordingSessions=dir('*');
                    for sess=size(recordingSessions,1):-1:3 % Go through all the sessions folders
                        cd([ENGINE_root '/' population ]);
                        cd(recordingFolders(subj).name);
                        cd(recordingSessions(sess).name);
                        if ~isfolder('NKT')                 % If the folder doesn't contain an 'NKT'
                        cd('..');                           % folder, go back to the previos path
                            continue;                       % and go back to the beggining of the
                        end                                 % loop.
                        cd('NKT')
                        cd('EEG2100')
                        Path=[ENGINE_root '/' population '/' recordingFolders(subj).name '/' recordingSessions(sess).name '/NKT/EEG2100/'];
                        eegFiles=dir('*.EEG');
                        FileName=eegFiles(1).name;          % Recording the EEG file in a variable

                
                        su = recordingFolders(subj).name;   % Recording the EEG file name in a variable
                        
                        DateReading = eegFiles().date;      % Read date from the data file
                        DateVector = datevec(DateReading);  % Make it a vector
                        year = num2str(DateVector(1), '%04d');  % Hour with 4 digits
                        month = num2str(DateVector(2), '%02d'); % Month with 2 digits
                        day = num2str(DateVector(3), '%02d');   % Day with 2 digits
                        hour = num2str(DateVector(4), '%02d');  % Hour with 2 digits
                        minute = num2str(DateVector(5), '%02d');% Minute with 2 digits
                        second = num2str(DateVector(6), '%02d');% Seconds with 2 digits

                        % Build the string in the 'yyyymmddHHMMSS' format
                        suDate = [year month day hour minute second];

                        % take the subj number out of the filename
                        suNumber = str2num(su);

                        % get the visit number
                        visitNumber = sess-2; % Substract 2 because recordingSessions has size 5 despite having 3 folders

                        % get the task name
                        taskName = 'RestingStateEyesClosed';%getTaskName(activeFile);
                        suGroup=population;

                        tptNumber=1;
                
                        
                        if(~exist([BIDS_DIR '/' projects(p).name '/' population '/sub-' num2str(suNumber) '/ses-' num2str(visitNumber) '/eeg/'...
                                'sub-' num2str(suNumber) '_ses-' num2str(visitNumber) '_task-' taskName '_run-' num2str(tptNumber) '_eeg.eeg'],'file'))

                            % Convert discovery files to be readable 
                            [data] = convert_discovery(Path,FileName);
                            eventsTable=table;
                            
                            % Convert to BIDS
                            bids_root=[BIDS_DIR '/' projects(p).name '/' population '/'];
                            status = convert_bids(eventsTable, data, bids_root, taskName, suNumber,suGroup,visitNumber,tptNumber,suDate);

                            clear data;
                            clear rawdata;
                        end
                    end
                end 
            end % Population cases
        end % Population loop (pop = 1:2)
    end % ENGINE

    if(contains(projects(p).name,'BLOSSOM'))

       BLOSSOM_root = ['/autofs/vast/neuromod/tDCS_EEG/Projects/18_BLOSSOM'];
       cd(BLOSSOM_root);
       BLOSSOM_data = [BLOSSOM_root '/data/eeg'];
       cd(BLOSSOM_data);
       recordingData = dir('*.e');

       for subj = 1:size(recordingData,1)
            % Convert .e data

            data = convert_nicolet(recordingData(1).name);
       
            eventsTable=table; % Maybe modify

            % Obtain the subject number
            splitname = strsplit(recordingData(subj).name, '_');
            suNumber = splitname{1};

            taskName = 'RestingStateEyesClosed'; % Task name
            tptNumber = 1; % Time points

            % Visit number
            visitdate = splitname{2};
            splitvisitdate = strsplit(visitdate,'-');
            visitNumber = splitvisitdate{1}(1);
            
            % Visit data
            Year = splitvisitdate{4}(1:4);  
            Month = splitvisitdate{2};
            Day = splitvisitdate{3};
            suDate = [Year Month Day '000000'];

            suGroup = ''; % Maybe modify



                if(~exist([BIDS_DIR '/' projects(p).name '/sub-' num2str(suNumber) '/ses-' num2str(visitNumber) '/eeg/'...
                        'sub-' num2str(suNumber) '_ses-' num2str(visitNumber) '_task-' taskName '_run-' num2str(tptNumber) '_eeg.eeg'],'file'))

                    bids_root=[BIDS_DIR '/' projects(p).name '/'];
                    status = convert_bids(eventsTable, data, bids_root, taskName, suNumber,suGroup,visitNumber,tptNumber,suDate);


                    clear data;
                    clear rawdata;
                end
       end
       


    end


end % all projects
%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% SUBFUNCTIONS - do not edit unless necessary
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [data] = convert_nicolet(FileName)

    % using high-level functions (recommended)
    cfg            = [];
    cfg.dataset    = FileName;
    cfg.continuous = 'yes';
    cfg.channel    = 'all';
    data           = ft_preprocessing(cfg);
end

function [data] = convert_discovery(Path,FileName)


    % This function will provide you all the information needed
    FileInfo = NK_FileInfo([Path FileName])
    
    % The output should be like this 
    % 
    % DeviceType: 'EEG-1100C V01.00'
    %         StartDate: {'07:11:17'}
    %         StartTime: {'15:39:09'}
    %      SamplingRate: 200
    %         TotalTime: 1932
    %             NumCh: 44
    %       HeaderBytes: 6611
    %     ElectrodeCode: {[0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 42 43 44 45 76 77 78]}
    %             ChMap: {44×1 cell}
    %          Comments: [1×1 struct]
    
    % Please write FileInfo.ChMap in commend line
    
    Ch = [1:size(FileInfo.ChMap,1)];
    
    
    %%
    
    % Open the data file for reading 
    Fid = fopen([Path FileName]);
    
    % skipe the header to read the data
    fseek(Fid,FileInfo.HeaderBytes,'bof');
    
    % Read the entire of data
    Temp = -(fread(Fid,[FileInfo.NumCh inf],'uint16')'-32768)*3200/32768;
    
    % close the data file
    fclose(Fid);
    
    data = Temp(:,Ch); % dimensions: samples x chan
    %%
    
    time = [1:length(data)]/FileInfo.SamplingRate;
    
    labels=FileInfo.ChMap(1:19);
    
    dt = data(:,[1:19]);
    
    time = time';
    
    
    t = mat2cell(time,size(time,1),size(time,2));
    l = mat2cell(dt',size(dt',1),size(dt',2));
    
    % Fieldtrip conversion
    cfg = [];
    rawdata.label   = labels; % cell-array containing strings, Nchan X 1
    rawdata.fsample = FileInfo.SamplingRate; % sampling frequency in Hz, single number
    rawdata.trial   =  mat2cell(dt',size(dt',1),size(dt',2)); % cell-array containing a data matrix for each trial (1 X Ntrial), each data matrix is    Nchan X Nsamples
    rawdata.time    =t; % cell-array containing a time axis for each trial (1 X Ntrial), each time axis is a 1 X Nsamples vector
    
    data = ft_preprocessing(cfg,rawdata);

  
end



function [data, codes] = convert_easy(activeFile,labels)
d = load(activeFile); % import the .easy file into matlab
if(isempty(d))
    error('empty file, skipping')
    return;
end
% Next we define the time axis using the last column in the data (in ms
% Unix time):
time=d(:,end); % time stamp is in the last column - in ms Unix time
time=time-time(1); % set clock to zero in first sample
time=time/1000; % change time units to seconds
% place the new time back into data for trial setup later
d(:,end) = time;
time = time';

% convert to mV (?)because Nic records in microvolts
if(length(labels)<9)
    dt = d(:,1:8)/1e3; % 8 channels
    codes = d(:,[9]); % codes are in col 9 for 8 channel
else
    codes = d(:,[36,end]); % codes are in col 36 for 32 channel
    dt = d(:,1:32)/1e3; % 32 channels

end

t = mat2cell(time,size(time,1),size(time,2));
l = mat2cell(dt',size(dt',1),size(dt',2));

% Fieldtrip conversion
cfg = [];
rawdata.label   = labels; % cell-array containing strings, Nchan X 1
rawdata.fsample = 500; % sampling frequency in Hz, single number
rawdata.trial   =  mat2cell(dt',size(dt',1),size(dt',2)); % cell-array containing a data matrix for each trial (1 X Ntrial), each data matrix is    Nchan X Nsamples
rawdata.time    =t; % cell-array containing a time axis for each trial (1 X Ntrial), each time axis is a 1 X Nsamples vector

data = ft_preprocessing(cfg,rawdata);
end

function taskName = getTaskName(activeFile)

    if(contains(activeFile,'open','IgnoreCase',true) | ...
            contains(activeFile,'EO','IgnoreCase',true))
        taskName='RestingStateEyesOpen';
    elseif(contains(activeFile,'EC','IgnoreCase',true) | ...
            contains(activeFile,'close','IgnoreCase',true))
        taskName='RestingStateEyesClosed';

    elseif(contains(activeFile,'SS','IgnoreCase',true))
        taskName='StopSignalTask';
    elseif(contains(activeFile,'lanker','IgnoreCase',true)| ...
            contains(activeFile,'_ft','IgnoreCase',true))
        taskName='FlankerTask';
    elseif(contains(activeFile,'_DD','IgnoreCase',true))
        taskName='DelayDiscountingTask';
    elseif(contains(activeFile,'back','IgnoreCase',true)| ...
            contains(activeFile,'_nb','IgnoreCase',true))
        taskName='NbackTask';
    elseif(contains(activeFile,'msit','IgnoreCase',true)| ...
            contains(activeFile,'MSIT','IgnoreCase',true))
        taskName='MsitIapsTask';
    elseif(contains(activeFile,'ig','IgnoreCase',true))
        taskName='IowaGamblingTask';
        % FFOR specific
        % block_names={'D1: Habituation','D1: Acquisition',
        % 'D1: Extinction','D2: Recall','D2: Reinstatement'};

        elseif(contains(activeFile,'lock1','IgnoreCase',true) | ...
                contains(activeFile,'B1','IgnoreCase',true))
           taskName='FFORDay1Habituation';
        elseif(contains(activeFile,'lock2','IgnoreCase',true)| ...
                contains(activeFile,'B2','IgnoreCase',true))
           taskName='FFORDay1Acquisition';
        elseif(contains(activeFile,'lock3','IgnoreCase',true)| ...
                contains(activeFile,'B3','IgnoreCase',true))
           taskName='FFORDay1Extinction';
        elseif(contains(activeFile,'lock4','IgnoreCase',true)| ...
                contains(activeFile,'B4','IgnoreCase',true))
           taskName='FFORDay2Recall';
        elseif(contains(activeFile,'lock5','IgnoreCase',true)| ...
                contains(activeFile,'B5','IgnoreCase',true))
           taskName='FFORDay2Reinstatement';
     
    else
        disp(activeFile);
        error('no matching task found')
    end
    
end

function [events, eventsTableTSV] = process_codes(codes,studyName,taskName,Fs, suNumber,suGroup,visitNumber,tptNumber)
    triggers=[];idx=1;
    for c=1:length(codes)
        if(codes(c,1)~=0)
            triggers(idx,1)=codes(c);
            triggers(idx,2)=c;
            idx=idx+1;
        end
    end
    if(~isempty(triggers))
        [events, eventsTableTSV]=process_task(studyName,taskName,triggers,Fs,suNumber,suGroup,visitNumber,tptNumber);
        % Setup the trials of interest
%         cfg = [];
%         alltrls = events; % ALL trials
%         cfg.trl = alltrls;
%         data = ft_redefinetrial(cfg,data);
    else
        disp('no triggers found')
        events = [];
        eventsTable=[];
        eventsTableTSV=[];

    end
end
    
function [events, eventsTableTSV]= process_task(studyName,taskName,triggers,Fs,suNumber,suGroup,visitNumber,tptNumber)

    events=[];
    eventsTable= [];
    eventsTableTSV = [];
    % set up triggers depending on task
    switch taskName
        case 'StopSignalTask'
            % code meanings:
            % 1 and 2 are responses
            % 3 is A, 4 is Z, 5 is stop signal
    
            % output meannigs:
            % 100 - correct A
            % 200 correct x
            % 300 - correct stop
            % 150 - incorrect A
            % 250 - incorrect X/incorrect stop
            %
            % remove everything that's not 1 2 3 4 5
            triggers(find(~ismember(triggers(:,1),[1:5])),:)=[];
            % preprocess the trigger codes and epoch trials
            % trigger code meanings
            % A trials (code 3) followed by 1 is correct
            %  X trials (code 4) followed by 2 is correct
            % Stop trials (5) followed by OR preceded by NO RESPONSE is correct
            % all correct A trials = 100
            % all correct X trials = 200
            % all correct stop trials = 300
            % all incorrect A trials = 150
            % all incorrect X trials = 250
            % all incorrect stop trials = 350
            triggers(end+1,1)=0; % pad with zero just in case we have a final trial of 5 (stop)
            for i=1:size(triggers,1)-1
    
                if(triggers(i,1)>2) % 1 and 2 are responses!
                    if(triggers(i,1)==3 & triggers(i+1,1)~=5 )
                        if(triggers(i+1,1)==1)
                            triggers(i,1)=100; % correct A
                        else
                            triggers(i,1)=150; % incorr A
                        end
                        triggers(i,3) = triggers(i+1,2)-triggers(i,2);
                    elseif(triggers(i,1)==4 & triggers(i+1,1)~=5 )
                        if(triggers(i+1,1)==2) % correct X
                            triggers(i,1)=200;
                        else
                            triggers(i,1)=250; % incorr X
                        end
                        triggers(i,3) = triggers(i+1,2)-triggers(i,2);
                    elseif(triggers(i,1)==5 & ismember(triggers(i-1,1),[3 4]))
                        if(triggers(i+1,1)==3)
                            triggers(i,1)=300; % correct st.
                        elseif(triggers(i+1,1)==4)
                            triggers(i,1)=300; % also correct
                        elseif(triggers(i+1,1)==0)
                            triggers(i,1)=300; % also correct
                        else
                            triggers(i,1)=250; % incorr stop.
                            triggers(i,3) = triggers(i+1,2)-triggers(i,2);
                        end
                    end
                end
            end
    
            % add latency diffs
    
            % add ITI latency diffs ?!?
            try
                 triggers(:,4)= triggers(:,3)/Fs;
            catch
                error('no conditions identified');
                eventsTableTSV=table;
                return ;
            end
            %             triggers(:,4)=[0; diff(triggers(:,2))];
            %             triggers(:,4)=triggers(:,4)/Fs; % in sec;
    
            triggers(99>triggers(:,1),:)=[];
            idx=1;
            for i=1:length(triggers)
                if(triggers(i,1)>99)
                    events(idx,1) = triggers(i,2) - (0.5*Fs) ; % beg tpt back track 150ms from trigger
                    events(idx,2) = triggers(i,2) + (1 * Fs); % % end time, 1 sec after trigger
                    events(idx,3) = (-0.5 * Fs); % offset
                    events(idx,4) = triggers(i,1); % trial code
                    events(idx,5) = triggers(i,3); % RT
                    idx=idx+1;
                end
            end
    
    
            eventsTable=array2table(events,'VariableNames',{'begsample','endsample','offset','code','RT'});
            % convert to events.tsv struct for BIDS
            % see https://bids-specification.readthedocs.io/en/stable/04-modality-specific-files/05-task-events.html
            % first col- onset (in seconds)
            % duration (number or n/a)
            % sample
            % trial_type (go/no-go/etc)
            % response_time (in sec)
            % value (trigger code)
            % stim file (if there's an external file URI)
            eventsTSV={};
            eventsTSV(:,1)=num2cell((eventsTable.begsample+(eventsTable.offset*-1))/Fs); % sec
            eventsTSV(:,2)=num2cell(zeros(size(eventsTSV,1),1)); % duration
            eventsTSV(:,3)=num2cell(eventsTable.begsample+(eventsTable.offset*-1)); % sample
            eventsTSV(:,4)=num2cell(eventsTable.code); % trigger code
            % eventsTSV(:,5)=num2cell(eventsTable.code); % value
            eventsTSV(:,5)=num2cell(eventsTable.RT);
            eventsTableTSV=cell2table(eventsTSV,'VariableNames',{'onset','duration','sample','trial_type','value'});
    
        case 'FlankerTask' 

            % 1 2 and 3 are responses (left, right and space)
            % 11, 21, 12, 22 are stimuli  onsets
            triggers2=triggers;
            for i=1:length(triggers)
    
                if(i==length(triggers)) continue; end
    
                if(triggers(i,1)>=11)
                    if(triggers(i,1)==11)
                        if(triggers(i+1,1)==1)
                            triggers(i,1)=100; % correct cong
                        else
                            triggers(i,1)=150; % incorr cong
                        end
                    elseif(triggers(i,1)==12)
                        if(triggers(i+1,1)==1) % correct incong
                            triggers(i,1)=200;
                        else
                            triggers(i,1)=250; % incorr incong
                        end
                    elseif(triggers(i,1)==21)
                        if(triggers(i+1,1)==2)
                            triggers(i,1)=200; % correct incong.
                        else
                            triggers(i,1)=250; % incorr incong.
                        end
                    elseif(triggers(i,1)==22)
                        if(triggers(i+1,1)==2)
                            triggers(i,1)=100; % corr cong
                        else
                            triggers(i,1)=150; % incor cong.
                        end
                    end
                end
    
            end
            % add RT's
            triggers(:,3)=[0;diff(triggers(:,2))]/Fs;
            % bring RT up one row to perserve it in trialinfo
            triggers(:,3)=[triggers([2:end],3);0];
            % add ITI
            triggers(find(triggers(:,1)<99),:)=[];
            triggers(:,4)=[0;diff(triggers(:,2))]/Fs;
            triggers(:,4)=[triggers([2:end],4);0];
            % do an adjustment to starting sample based on ITI
            triggers(:,5)=triggers(:,2);
    
            %                             triggers(:,6)=table2array(CSV(:,5))/10000;
    
            % sanity chekc
            if(isempty(triggers))
                warning('no triggers found, check~');
                return;
            end

            % remove first trial (contains zero RT)
            triggers(1,:)=[];
    
            idx=1;
            for i=1:size(triggers,1)
                if(triggers(i,1)>99)
                    events(idx,1) = triggers(i,2) - (1*Fs) ; % beg tpt back track 1sec from trigger
                    events(idx,2) = triggers(i,2) + (1 * Fs); % % end time, 1 sec after trigger
                    events(idx,3) = (-1 * Fs); % offset
                    events(idx,4) = triggers(i,1); % trial code
                    events(idx,5) = triggers(i,3); % RT
                    events(idx,6) = triggers(i,4); %ITI
                    idx=idx+1;
                end
    
            end
            eventsTable=array2table(events,'VariableNames',{'begsample','endsample','offset','code','RT','ITI'});
            % convert to events.tsv struct for BIDS
            % see https://bids-specification.readthedocs.io/en/stable/04-modality-specific-files/05-task-events.html
            % first col- onset (in seconds)
            % duration (number or n/a)
            % sample
            % trial_type (go/no-go/etc)
            % response_time (in sec)
            % value (trigger code)
            % stim file (if there's an external file URI)
            eventsTSV={};
            eventsTSV(:,1)=num2cell((eventsTable.begsample+(eventsTable.offset*-1))/Fs); % sec
            eventsTSV(:,2)=num2cell(zeros(size(eventsTSV,1),1)); % duration
            eventsTSV(:,3)=num2cell(eventsTable.begsample+(eventsTable.offset*-1)); % sample
            eventsTSV(:,4)=num2cell(eventsTable.code); % trigger code
            eventsTSV(:,5)=num2cell(eventsTable.code); % value
            % eventsTSV(:,5)=num2cell(eventsTable.RT);
            eventsTableTSV=cell2table(eventsTSV,'VariableNames',{'onset','duration','sample','trial_type','value'});

        case 'DelayDiscountingTask'
            % 99 = start of block
            % 3 = stimulus (value)
            % 1 = left option
            % 2 = right option (always 100 in X days)
    
            nns=find(triggers([1:10],1)==99);
            if(~isempty(nns))
                triggers(1:nns(end),:)=[];
            end
            DD_path=['/mnt/aj123/EEG/0_Projects/8_SUD_TobaccoCannabis/analysis/DD/preprocessedLogFiles/'...
                suGroup '_sub-' num2str(suNumber) '_ses-' num2str(visitNumber) '_tpt-' num2str(tptNumber) '_DD_log.mat'];
            if(~exist(DD_path,'file'))
                disp(DD_path);
                warning('no behav file found for DD task, skipping');
                                eventsTableTSV=table;

                return;
            end
            load(DD_path); % output is finalTable
    
            for i=1:length(triggers)
    
                if(i==length(triggers)) continue; end
    
                if(triggers(i,1)<99)
                    if(triggers(i,1)==3)
                        if(triggers(i+1,1)==1)
                            triggers(i,1)=100; % immediate value
                        else
                            triggers(i,1)=200; % delayed value
                        end
    
                    end
    
                end
            end
            % add RT's
            triggers(:,3)=[0;diff(triggers(:,2))]/Fs;
            % bring RT up one row to perserve it in trialinfo
            triggers(:,3)=[triggers([2:end],3);0];
            triggers(find(triggers(:,1)<100),:)=[];
            % add values of the rewards from log file
            % quick sanity check
            chk=[triggers(:,1) finalTable(:,5)*100];
            if(sum(abs(chk(:,1)-chk(:,2)))<=100) disp('good');
            else
                error('mismatch between eeg and pres.');
            end
            % do an adjustment to starting sample based on ITI
            triggers(:,[4:3+size(finalTable,2)])=finalTable;
            idx=1;
            for i=1:size(triggers,1)
                if(triggers(i,1)>99)
                    events(idx,1) = triggers(i,2) - (1*Fs) ; % beg tpt back track 150ms from trigger
                    events(idx,2) = triggers(i,2) + (8 * Fs); % % end time, 8 sec after trigger
                    events(idx,3) = (-1 * Fs); % offset
                    events(idx,4) = triggers(i,1); % trial code
                    events(idx,5) = triggers(i,3); % RT
                    events(idx,6) = triggers(i,4); % immediate reward
                    events(idx,7) = triggers(i,6); % late reward
                    events(idx,8) = triggers(i,7); % late delay
                    events(idx,9) = triggers(i,9)/10000; % more accurate RT
    
                    idx=idx+1;
                end
    
            end
            eventsTable=array2table(events,'VariableNames',{'begsample','endsample','offset','code','RT','IR','LR','LD','aRT'});
            % convert to events.tsv struct for BIDS
            % see https://bids-specification.readthedocs.io/en/stable/04-modality-specific-files/05-task-events.html
            % first col- onset (in seconds)
            % duration (number or n/a)
            % sample
            % trial_type (go/no-go/etc)
            % response_time (in sec)
            % value (trigger code)
            % stim file (if there's an external file URI)
            eventsTSV={};
            eventsTSV(:,1)=num2cell((eventsTable.begsample+(eventsTable.offset*-1))/Fs); % sec
            eventsTSV(:,2)=num2cell(zeros(size(eventsTSV,1),1)); % duration
            eventsTSV(:,3)=num2cell(eventsTable.begsample+(eventsTable.offset*-1)); % sample
            eventsTSV(:,4)=num2cell(eventsTable.code); % trigger code
            eventsTSV(:,5)=num2cell(eventsTable.code); % value
            % eventsTSV(:,5)=num2cell(eventsTable.RT);
            eventsTableTSV=cell2table(eventsTSV,'VariableNames',{'onset','duration','sample','trial_type','value'});
    
        case 'NbackTask'
            % preprocess the trigger codes
    
            % trigger code meanings
            % 1 2 are responses
            % all other numbers represent the ASCII code for keyboard letters
            % in order to get actual results, you have to rely on behav. log file
            % from Presentation
%             expFolder='/autofs/vast/neuromod/tDCS_EEG/Projects/05_HC_HD2_attention/Behav/nback';
            expFolder=['/autofs/vast/neuromod/tDCS_EEG/BIDS/' studyName '/sub-' num2str(suNumber) '/ses-' num2str(visitNumber) '/beh'];
 
            fileOut = [expFolder '/sub-' num2str(suNumber) '_ses-' num2str(visitNumber) '_task-NbackTask_run-' num2str(tptNumber) '_events.tsv'];
    
            if(exist([fileOut],'file'))
                finalTable=ft_read_tsv(fileOut);
                finalTable=table2array(finalTable);
                finalTable2=finalTable;
                triggers2=triggers;
                if(strcmp(fileOut,'/autofs/vast/neuromod/tDCS_EEG/BIDS/05_HC_HD2_attention/sub-15/ses-1/beh/sub-15_ses-1_task-NbackTask_run-2_events.tsv'))
                    finalTable([1:5],:)=[];
                elseif(strcmp(fileOut,'/autofs/vast/neuromod/tDCS_EEG/BIDS/01_ADHD_attention/sub-9/ses-1/beh/sub-9_ses-1_task-NbackTask_run-1_events.tsv'))
                    finalTable([1:36],:)=[];
                elseif(strcmp(fileOut,'/autofs/vast/neuromod/tDCS_EEG/BIDS/01_ADHD_attention/sub-9/ses-1/beh/sub-9_ses-1_task-NbackTask_run-2_events.tsv'))
                    finalTable([1:9],:)=[];
                elseif(strcmp(fileOut,'/autofs/vast/neuromod/tDCS_EEG/BIDS/01_ADHD_attention/sub-15/ses-2/beh/sub-15_ses-2_task-NbackTask_run-2_events.tsv'))
                    finalTable([1:3],:)=[];
                elseif(strcmp(fileOut,'/autofs/vast/neuromod/tDCS_EEG/BIDS/03_HC_HCa_FPN_attention_2018/sub-17/ses-3/beh/sub-17_ses-3_task-NbackTask_run-2_events.tsv'))
                    finalTable([1:15],:)=[];    
                elseif(strcmp(fileOut,'/autofs/vast/neuromod/tDCS_EEG/BIDS/03_HC_HCa_FPN_attention_2018/sub-14/ses-3/beh/sub-14_ses-3_task-NbackTask_run-2_events.tsv'))
                    finalTable([1:6],:)=[];    
                elseif(strcmp(fileOut,'/autofs/vast/neuromod/tDCS_EEG/BIDS/03_HC_HCa_FPN_attention_2018/sub-14/ses-3/beh/sub-14_ses-3_task-NbackTask_run-1_events.tsv'))
                    finalTable([1:3],:)=[];
                 elseif(strcmp(fileOut,'/autofs/vast/neuromod/tDCS_EEG/BIDS/03_HC_HCa_FPN_attention_2018/sub-4/ses-3/beh/sub-4_ses-3_task-NbackTask_run-1_events.tsv'))
                    finalTable([1:4],:)=[];   
                 elseif(strcmp(fileOut,'/autofs/vast/neuromod/tDCS_EEG/BIDS/05_HC_HD2_Bifrontal_attention_2019/sub-15/ses-1/beh/sub-15_ses-1_task-NbackTask_run-2_events.tsv'))
                    finalTable([1:5],:)=[];   
                
                elseif(strcmp(fileOut,'/autofs/vast/neuromod/tDCS_EEG/BIDS/03_HC_HCa_FPN_attention_2018/sub-3/ses-1/beh/sub-3_ses-1_task-NbackTask_run-1_events.tsv'))
                    return;
                end
%                 load(fileOut); % finalTable is output
                % strip out rows with 3 in response col
                finalTable(:,8)=[diff(finalTable(:,1));0];
                finalTable(find(finalTable(:,4)==3),:)=[];
                triggers(find(triggers(:,1)==3),:)=[];
    
                    % remove double responses from behav file
                    rmList=[];
                    for tr=1:size(finalTable,1)
                        if(finalTable(tr,4)<3 & tr<size(finalTable,1))
                            if(finalTable(tr+1,4)<3)
                                rmList(end+1)=tr+1;
                            end
                        end
                        
                    end
                    finalTable(rmList,:)=[];

                    % remove double responses from eeg file (should rarely
                    % happ1n)
                    rmList=[];
                    for tr=1:size(triggers,1)
                        if(triggers(tr,1)<3 & tr<size(triggers,1))
                            if(triggers(tr+1,1)<3)
                                rmList(end+1)=tr+1;
                            end
                        end
                        
                    end
                    triggers(rmList,:)=[];


                if(size(finalTable,1)>240)
                    if(finalTable(end,2)<3 & finalTable(end-1,2)<3)
                        % there's a double resp causing the lag
                        finalTable=finalTable([1:240],:);
                    else
                        % trim top values to preserve bottom sync
                        finalTable=finalTable([(size(finalTable,1)-240)+1:end],:);
                        %                 finalTable=finalTable([1:240],:);
                    end
                end
                if(size(triggers,1)>240)
    
                    % trim top values to preserve bottom sync
                    triggers=triggers([(size(triggers,1)-240)+1:end],:);
                    %                 triggers=triggers([1:240],:);
                end
                if(size(finalTable,1)==size(triggers,1))
                    triggers=[triggers finalTable];
                else
                    % try to match the two using trial codes
                    [lag]=finddelay(triggers(:,1),finalTable(:,2))
                    triggers=[triggers([1+-lag:end],:)];
                    if(size(triggers,1)<size(finalTable,1))
                        finalTable=finalTable([1:size(triggers,1)],:);
                    else
                        triggers=triggers([1:size(finalTable,1)],:);
                    end
                    if(size(finalTable,1)==size(triggers,1))
                        triggers=[triggers finalTable];
                    else
                        error('mismatch in number of rows! check')
                    end
                end
                if(sum(triggers(:,6)-triggers(:,1)))
                    chk=[triggers(:,6) triggers(:,1) triggers(:,6)-triggers(:,1)];
                    error('mismatch in trigger values..check! you can use finaltable2 and triggers2 to quickly revert.')
                end
    
                diff1=diff(triggers(:,2))/Fs;
                diff2=diff(triggers(:,3)/10000);
                jitter=diff1-diff2;
                disp('Average Jitter:');
                disp(nanmean(jitter));
    
                % Correct Hits  = 100 or 200
                % Incorrect = -100 or -200
                % Miss = -200 or -400
    
                for i=1:length(triggers)
    
                    if(i==length(triggers))
                        continue;
                    end
    
                    if(triggers(i,1)>3)
                        if(triggers(i,7)==1) % correct hit
    
                            triggers(i,7)=100*triggers(i,8); %
    
                        elseif(triggers(i,7)==-1)   % incorrect
    
                            triggers(i,7)=-100*triggers(i,8);
    
                        elseif(triggers(i,7)==-2)   % miss
    
                            triggers(i,7)=-200*triggers(i,8);
    
                        end
                    end
    
                end
                % add RT diffs
                triggers(:,9)=[0; diff2];
                triggers(:,9)=[triggers([2:end],9); 0];
                idx=1;
                for i=1:length(triggers)
                    if(triggers(i,1)>3)
                        events(idx,1) = triggers(i,2) - (0.5*Fs) ; % beg tpt back track 500ms from trigger
                        events(idx,2) = triggers(i,2) + (1.4 * Fs); % % end time, 1 sec after trigger
                        events(idx,3) = (-0.5 * Fs); % offset
                        events(idx,4) = triggers(i,6); % trial code
                        events(idx,5) = triggers(i,9); % RT
                        events(idx,6) = triggers(i,8); % dprime response
                        events(idx,7) = triggers(i,5); % trial number

                        idx=idx+1;
    
                    end
                end
    
                eventsTable=array2table(events,'VariableNames',{'begsample','endsample','offset','code','RT','dPrimeResponse','TrialNumber'});
                % convert to events.tsv struct for BIDS
                % see https://bids-specification.readthedocs.io/en/stable/04-modality-specific-files/05-task-events.html
                % first col- onset (in seconds)
                % duration (number or n/a)
                % sample
                % trial_type (go/no-go/etc)
                % response_time (in sec)
                % value (trigger code)
                % stim file (if there's an external file URI)
                eventsTSV={};
                eventsTSV(:,1)=num2cell((eventsTable.begsample+(eventsTable.offset*-1))/Fs); % sec
                eventsTSV(:,2)=num2cell(zeros(size(eventsTSV,1),1)); % duration
                eventsTSV(:,3)=num2cell(eventsTable.begsample+(eventsTable.offset*-1)); % sample
                eventsTSV(:,4)=num2cell(eventsTable.code); % trigger code
                eventsTSV(:,5)=num2cell(eventsTable.code); % value
                eventsTSV(:,6)=num2cell(eventsTable.RT);
                eventsTSV(:,7)=num2cell(eventsTable.TrialNumber);
                eventsTSV(:,8)=num2cell(eventsTable.dPrimeResponse);
                eventsTableTSV=cell2table(eventsTSV,'VariableNames',{'onset','duration','sample','trial_type','value','response_time','trial_number','response_value'});
    
            else
                disp(fileOut)
                warning('no behav file found!')
                                eventsTableTSV=table;

                return;
    
            end
    
        case 'MsitIapsTask'
            % preprocess the trigger codes
    
            % trigger code meanings
            % there's an emotional valence picture (pos neg or netural)
            % theres a 3 digit number after
            % subj has to identify the number that is incong.
            % the stimulus is coded- first 3 numbers are the task, last 2
            % are the emotional valence

            % last two digits = 91 (neg) 92 (post) 93 (neutral)
            % you have to pad the first three digits if theres leading
            % zeros
    
            % first three digits can distinguish btwn interference vs non
            % interference
            interferenceTrials=[212
                                332
                                311
                                112
                                232
                                313
                                211
                                322
                                221
                                131
                                331
                                233]; 
            nonInterferenceTrials=[020, 003, 100];


            %  2 3 or 4 are responses for mismatch of 1 2 or 3
            %  (respectively)
    
            % final output table:
            % 100 - correct in positive valence - no interference
            % 200 - correct in negative valence- no interference
            % 300 - correct in neutral valence- no interference

            % 1000 - correct in positive valence - w/ interference
            % 2000 - correct in negative valence- w/ interference
            % 3000 - correct in neutral valence- w/ interference

            % 150 - incorr pos. - no int
            % 250 - incorr neg. - no int
            % 350 - incorr neut. - no int

            % 1050 - incorr pos. - no int
            % 2050 - incorr neg. - no int
            % 3050 - incorr neut. - no int

            % 0 - no response
    
            for i=1:size(triggers,1)-1
                if(triggers(i,1)>15)
                    stimulus_code = num2str(triggers(i,1));
                    valence=stimulus_code(end-1:end);
                     
                    if(strcmp(valence,'91')) % neg
                        val=200;
                    elseif(strcmp(valence,'92')) % pos
                        val=100;
                    elseif(strcmp(valence,'93')) % neutral
                        val=300;
                    end
                    if(length(stimulus_code)==3)
                        stringMatch=['00' stimulus_code(1)];
                    elseif(length(stimulus_code)==4)
                        stringMatch=['0' stimulus_code(1:2)];
                    else
                        try
                            stringMatch=stimulus_code(1:3);
                        catch
%                             error('check');
                            return;
                        end
                    end
                    % determine interference vs non-int
                    if(ismember(str2num(stringMatch),interferenceTrials))
                        val=val*10;
                    end

                    % calculate the correct response required
                    chars=unique(stringMatch);
                    for c=1:2
                        char=chars(c);
                        if(sum(char==stringMatch)==1)
                            mm=str2num(char);
                        end
                    end
                    if(triggers(i+1,1)<5)
                        if(triggers(i+1,1)-1==mm)
                            finalVal=val;
                        else
                            finalVal=val+50;
                        end
                    else
                        finalVal=0; % no response
                    end
                    triggers(i,3)=finalVal;
    
                end
    
            end
            % add RT's
            triggers(:,4)=[0;diff(triggers(:,2))]/Fs;
            % bring RT up one row to perserve it in trialinfo
            triggers(:,4)=[triggers([2:end],4);0];
    
            % remove first trial (contains zero RT)
            triggers(1,:)=[];
    
            idx=1;
            for i=1:size(triggers,1)
                if(triggers(i,1)>99)
                    events(idx,1) = triggers(i,2) - (1*Fs) ; % beg tpt back track 150ms from trigger
                    events(idx,2) = triggers(i,2) + (1 * Fs); % % end time, 1 sec after trigger
                    events(idx,3) = (-1 * Fs); % offset
                    events(idx,4) = triggers(i,3); % response code
                    events(idx,5) = triggers(i,1); % orig code
                    events(idx,6) = triggers(i,4); % RT
                    idx=idx+1;
                end
    
            end

            if(isempty(events))
                eventsTableTSV=table;
                events=[];
                return;
            end
    
            eventsTable=array2table(events,'VariableNames',{'begsample','endsample','offset','code','rawCode','RT'});
            % convert to events.tsv struct for BIDS
            % see https://bids-specification.readthedocs.io/en/stable/04-modality-specific-files/05-task-events.html
            % first col- onset (in seconds)
            % duration (number or n/a)
            % sample
            % trial_type (go/no-go/etc)
            % response_time (in sec)
            % value (trigger code)
            % stim file (if there's an external file URI)
            eventsTSV={};
            eventsTSV(:,1)=num2cell((eventsTable.begsample+(eventsTable.offset*-1))/Fs); % sec
            eventsTSV(:,2)=num2cell(zeros(size(eventsTSV,1),1)); % duration
            eventsTSV(:,3)=num2cell(eventsTable.begsample+(eventsTable.offset*-1)); % sample
            eventsTSV(:,4)=num2cell(eventsTable.code); % trigger code
            % eventsTSV(:,5)=num2cell(eventsTable.code); % value
            eventsTSV(:,5)=num2cell(eventsTable.RT);
            eventsTableTSV=cell2table(eventsTSV,'VariableNames',{'onset','duration','sample','trial_type','value'});
    
        case { 'FFORDay1Habituation' , 'FFORDay1Acquisition' , 'FFORDay1Extinction' , 'FFORDay2Recall' , 'FFORDay2Reinstatement'}
    
            % trial codes:
            % 1 : CS+ with real shock (US) (block 2 only)
            % 2 : CS+ with no shock
            % 3 : shock, should come ~7.8sec after code 1, co-terminating with prev. trial)
            % 4 : CS- with no shock (around 10x)
            % 8 : start/end of block
    
            % total trials expected:
            % Block 1: 10 (5 CS+, 5CS-), no Shocks (e.g. codes 2 and 4)
            % Block 2: 20 (19?) (7 (6?) CS+ with shock (code 1), 3 CS+ no shock, and 10 CS-)
            % Block 3: 24 (12 CS+ no shock, 12 CS-)
            % Block 4: 24 (12 CS+ no shock, 12 CS-)
            % Block 5: 24 (3 shocks + 12 CS+ no shock, 12 CS-)
            events=[];idx=1;
    
            for i=1:size(triggers,1)
                if(ismember(triggers(i,1),[1 2 3 4]))
                    events(idx,1)= (triggers(i,2))-(1*Fs);
                    events(idx,2)=events(idx,1)+(1*Fs)+(1*Fs); % 2 sec total?
                    events(idx,3)=(-1*Fs); % offset
                    events(idx,4)= triggers(i,1);
                    idx=idx+1;
    
                end
            end
            
            if(~length(events))
                eventsTable=table;
                eventsTableTSV={};
            else
                eventsTable=array2table(events,'VariableNames',{'begsample','endsample','offset','code'});
                % convert to events.tsv struct for BIDS
                % see https://bids-specification.readthedocs.io/en/stable/04-modality-specific-files/05-task-events.html
                % first col- onset (in seconds)
                % duration (number or n/a)
                % sample
                % trial_type (go/no-go/etc)
                % response_time (in sec)
                % value (trigger code)
                % stim file (if there's an external file URI)
                eventsTSV={};
                eventsTSV(:,1)=num2cell((eventsTable.begsample+(eventsTable.offset*-1))/Fs); % sec
                eventsTSV(:,2)=num2cell(zeros(size(eventsTSV,1),1)); % duration
                eventsTSV(:,3)=num2cell(eventsTable.begsample+(eventsTable.offset*-1)); % sample
                eventsTSV(:,4)=num2cell(eventsTable.code); % trigger code
                eventsTSV(:,5)=num2cell(eventsTable.code); % value
                % eventsTSV(:,5)=num2cell(eventsTable.RT);
                eventsTableTSV=cell2table(eventsTSV,'VariableNames',{'onset','duration','sample','trial_type','value'});
            end
    
    end
end

function [status] = convert_bids(eventsTable, data, bids_root, taskName, suNumber, suGroup, visitNumber,tptNumber,suDate)

hdr=ft_fetch_header(data);

cfg = [];
cfg.method    = 'convert';
cfg.datatype  = 'eeg';


% specify the output directory
cfg.bidsroot  = bids_root;
cfg.sub       = num2str(suNumber);% '.' num2str(stage)]; % this will be the unique ID
cfg.ses         = num2str(visitNumber);
cfg.run         = tptNumber;

cfg.task = taskName;
if(~isempty(eventsTable))
    cfg.events=eventsTable;
end

% this goes in channels.tsv
cfg.channels.name               = hdr.label;
cfg.channels.type               = repmat({'EEG'}, hdr.nChans, 1);  % Type of channel
cfg.channels.units              = repmat({'uV'}, hdr.nChans, 1);% Physical unit of the data values recorded by this channel in SI
cfg.channels.sampling_frequency = repmat(hdr.Fs, hdr.nChans, 1); % Sampling rate of the channel in Hz.

try
    cfg.electrodes.impedance = hdr.orig.impedances;
catch
end

cfg.sessions.acq_time = datestr(datenum(suDate,'yyyymmddHHMMSS'),'yyyy-mm-ddThh:MM:SS'); % according to RFC3339

% specify some general information that will be added to the eeg.json file
cfg.InstitutionName             = 'MGH';
cfg.InstitutionalDepartmentName = 'Dept Psychiatry';
cfg.InstitutionAddress          = '';

% provide the mnemonic and long description of the task
cfg.TaskName       = taskName;
cfg.TaskDescription = '';

% these are EEG specific
cfg.eeg.PowerLineFrequency = 60;
cfg.eeg.EEGReference       = '';

status=data2bids(cfg,data);

end
