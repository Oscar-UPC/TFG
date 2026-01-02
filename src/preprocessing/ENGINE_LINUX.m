%% 

clear all;
close all;
clc;

%%
addpath(('/autofs/vast/neuromod/asif/Projects/fieldtrip'))
addpath(('/autofs/space/tonetti_001/users/asif/Projects/eeglab/eeglab2024.2'))
addpath('/autofs/vast/neuromod/asif/Projects/misc_scripts');
addpath /homes/7/aj123/Projects/scripts/
addpath /autofs/vast/neuromod/tDCS_EEG/scripts
addpath('/homes/7/aj123/asif_neuromod/Projects/zapline-plus-main')
addpath('/homes/7/aj123/asif_neuromod/Projects/misc_scripts/restingIAF')
addpath('/autofs/vast/neuromod/asif/Projects/misc_scripts/cbrewer/cbrewer/cbrewer/')


ft_defaults % just initializes fieldtrip
ft_info off;
ft_warning off;

cfg=[];
cfg.layout='EEG1005.lay';
layout=ft_prepare_layout(cfg);
lay.lay=layout;

load('jet_light.mat');


eeglab; close all;


%% TD BRAIN PiPeLiNe INIT : Define the study here in accordance with the name in the BIDS directory

STUDY='ENGINE';
 

% do not modify these paths
base_path=['/autofs/vast/neuromod/tDCS_EEG/Projects/16_ENGINE'];
analysis_path=['/autofs/vast/neuromod/tDCS_EEG/Projects/16_ENGINE/analysis'];
root_path=['/autofs/vast/neuromod/tDCS_EEG/BIDS/16_ENGINE' ];
qcPath=['/autofs/vast/neuromod/tDCS_EEG/Projects/16_ENGINE/QC/' ];
%% define parameters related to all the conditions
conditions=struct;
% 1 Resting State
conditions(1).task='EC';
conditions(1).block_names={
    'RestingStateEyesClosed'
    };
conditions(1).triggers=[];
conditions(1).triggerLabels={};

 
%% choose what to analyze in accordance with the condition number above
% 1= resting state 

tskIdx=1

task=conditions(tskIdx).task; triggers=conditions(tskIdx).triggers;
  
block_names=conditions(tskIdx).block_names;
triggerLabels=conditions(tskIdx).triggerLabels;


%% Convert + QC raw data
%
cd(root_path)


for dataset=1:2

    % go into HC folder or MDD folder
    if(dataset==1)
        population='HC';
    else
        population='MDD';
    end
    root_path=['/autofs/vast/neuromod/tDCS_EEG/BIDS/16_ENGINE' ];

    cd([root_path '/' population])
    suList=dir('*sub*');
    
    for subj=1:size(suList,1)
        
        actSu=suList(subj).name;
        cd([root_path '/' population '/' actSu '/']);
        sessions= dir('*ses-*');
        for sess=1:size(sessions,1)
            cd([root_path '/' population '/' actSu '/' sessions(sess).name '/eeg']);
            blocks=dir(['*.eeg']);
            if(isempty(blocks))
                continue;
            end 
                
           
            for block=1:size(blocks,1)  % eo, ec
                cd([root_path '/' population '/' actSu '/' sessions(sess).name '/eeg']);
                if(contains(blocks(block).name,'EyesClosed'))
                    actTask= 'EC';
                elseif(contains(blocks(block).name,'EyesOpen'))
                    actTask='EO';
                else
                    error('check name, no eo/ec found');
                end
                disp(blocks(block).name(1:end-4));
                if(~exist([analysis_path '/' population '/' actSu '/ses_' num2str(sess) '_' actTask '_data_raw.mat']))
                    
                     
                    fn=blocks(block).name(1:end-4);
                
                    cfg=[];
                    cfg.headerfile     = [fn '.vhdr'];
                    cfg.datafile = [fn '.eeg'];
                    hdr        = ft_read_header(cfg.headerfile);
                    orighdr=hdr;
                 
                    data=ft_preprocessing(cfg);
                    
         
                
                   
                    %% QC the data
                    gcf=qa_computeSpectrum(data,01);
                    suptitle([ actSu ' - sess-' num2str(sess) ' - ' actTask ])
                    %% save snap
                    mkdir_if_not_exist([qcPath '/' population '/' actSu]);
                    fN=append(qcPath, population, "/", actSu, "/spectrogram_", num2str(sess), actTask, "_", num2str(block), ".png");
                    saveas(gcf,fN); close(gcf);
        
                    %% save raw data
                    mkdir_if_not_exist([analysis_path '/' population '/' actSu]);
                    save([analysis_path '/' population '/' actSu '/ses_' num2str(sess) '_' actTask '_data_raw.mat'],'data','-v7.3')
                else
                    disp('already done')
                end % chk
                cd ..
            end % blk
        end % sess
    end % subj
end
%% [ ALL DATASETS ] Preprocess pipeline
% 1. Bandpass
% 2. Bad channel removal
% 3. Re-reference
% 4. ICA?

cd(analysis_path)

for dataset=1:2
    % go into HC folder or MDD folder
    if(dataset==1)
        population='HC';
    else
        population='MDD';
    end
    cd([analysis_path '/' population])
    suList=dir('*sub*');
    for subj=1:size(suList,1)
        cd([analysis_path '/' population])
        actSu=suList(subj).name
        cd(actSu);
        sessions= dir(['*ses_*' 'EC' '*raw.mat']);
        for sess=1:size(sessions,1)
            fn=dir(['*' sessions(sess).name '*']);
            fn=fn.name

            if(~exist([analysis_path '/' population '/' actSu '/' fn(1:end-7) 'clean.mat']))

                load(fn);



                %% filter/preprocess
                cfg=[];
                % Fitering options
                cfg.bpfilter        = 'yes';
                cfg.bpfreq          = [0.3 50];
                cfg.demean='yes';
                cfg.detrend='yes';
                cfg.channel = ft_channelselection({'all','-EKG','-M1','-M2','-CB1','-CB2','-HEOG','-VEOG','-VPVA','-VNVB','-HPHL','-HNHR','-Erbs','-OrbOcc','-Mass'},data.label);

                % origChannel=cfg.channel;
                data_clean = ft_preprocessing(cfg,data);



                %% automated ampl. threshold removal in 100msec windows
                if(0) % REPLACED BY ASR BELOW
                        
                    % https://www.sciencedirect.com/science/article/pii/S1388245720305320
    
    
                    % go 100msec at a time, and remove any epochs that exceed a
                    % 150 uV difference
                    % error('s')
                    % find out how many samples equates to 100msec
                    windowSize=fix(data_clean.fsample/10);
                    rm=[];
                    actTrl=(data_clean.trial{1,1});
                    for ww=1:windowSize:(size(actTrl,2)/windowSize)-1
                        actWin=actTrl(:,[ww:ww+windowSize]);
                        actWinDiff=diff(actWin,1,2);
                        if(find(abs(actWinDiff)>150))
                            rm(end+1)=ww;
                        end
                    end
                    if(rm)
                        disp(rm)
                        % error('check');
                        for ww=length(rm):-1:1
                            actTrl(:,[ww+windowSize])=[];
                        end
                        data_clean.trial{1,1}=actTrl;
                        % fix time and sample axis
                        data_clean.sampleinfo=[1 size(actTrl,2)];
                        data_clean.time{1,1}=[0:(1/data_clean.fsample):(data_clean.sampleinfo(2)/data_clean.fsample)];
                        data_clean.time{1,1}=data_clean.time{1,1}(1:data_clean.sampleinfo(2));
                    else
                        % continue;
                    end
                end
                %% ASR time and channel cleaning
                if(01 )
                    % trim 60 sec from beginning and end of data (flat
                    % lines)

                    cfg=[];
                    cfg.latency=[60 data_clean.time{1,1}(end)-60];
                    data_clean=ft_selectdata(cfg,data_clean);
            %             cfg=[];
            %             % reref options
            %             cfg.reref = 'yes';
            %             cfg.refmethod = 'avg';
            %             cfg.refchannel ='all';
            %             data_clean_ref = ft_preprocessing(cfg,data_clean);
                       % clean_channel plugin
                        data_clean_ref = data_clean; % since already re-ref'd
             
                       % use cleanData EEGlab plugin to ID bad chans
                       Signal=struct;Signal.data=data_clean_ref.trial{1,1};Signal.srate=data_clean.fsample;
                       Signal.nbchan=length(data_clean_ref.label);
                       % make a chanlocs
                       cfg=[];
                       cfg.elec=ft_read_sens('/autofs/vast/neuromod/asif/Projects/fieldtrip/template/electrode/standard_1005.elc');
                        % replace with correct order and prune unused
                        A= cfg.elec.label;
                        B= data_clean.label;
                        [tf, idx] = ismember(B,A);
                        cfg.elec.label = cfg.elec.label(idx);
                        cfg.elec.elecpos=cfg.elec.elecpos(idx,:);
                        
                        Signal.chanlocs = struct('labels', cfg.elec.label, 'X', mat2cell(cfg.elec.elecpos(:,1), Signal.nbchan,1), 'Y', mat2cell(cfg.elec.elecpos(:,2), Signal.nbchan,1),'Z', mat2cell(cfg.elec.elecpos(:,3), Signal.nbchan,1));
                        Signal.chanlocs = convertlocs(Signal.chanlocs, 'cart2all');
                        Signal.etc=struct;
                        Signal.event=[];
                        Signal.urevent=[];
                        Signal.xmin=0;
                        Signal.filepath=pwd;
                        Signal.filename=actSu;
             
                       % Set parameters for ASR
                        % 'ChannelCriterion' is used to remove bad channels, default is 0.8 (can be adjusted)
                        % 'LineNoiseCriterion' deals with line noise, default is 4
                        % 'BurstCriterion' is used to remove large artifacts, default is 20
                        [EEG_clean,~,~,exclChans] = clean_artifacts(Signal, ...
                            'ChannelCriterion', 0.9, 'LineNoiseCriterion', 4, ...
                            'BurstCriterion', 25);
                         
                        % vis_artifacts(EEG_clean,Signal);
                        asrChans=(data_clean.label(exclChans));
                        disp(data_clean.label(exclChans));

                        % [signal badChans2] = clean_channels(Signal,[],[],[],0.7);
                        %replace into data_clean
                        data_clean.trial{1,1}=EEG_clean.data;
                        data_clean.time{1,1}= [0:(1/data_clean.fsample):size(EEG_clean.data,2)/data_clean.fsample];
                        data_clean.time{1,1}(end)=[];
                        data_clean.sampleinfo=[1 size(EEG_clean.data,2)];

                        data_clean.label={EEG_clean.chanlocs.labels}';
                   
                    if(length(asrChans))
                        cfgmr               = [];
                        cfgmr.layout		=  lay.lay;
                        cfgmr.method        = 'distance';
                        cfgmr.neighbourdist = 0.2;
                        neighbours = ft_prepare_neighbours(cfgmr);
                        cfg             = [];
                        cfg.method      = 'spline';
                        cfg.layout      = lay.lay;
                        cfg.elec=ft_read_sens('/autofs/vast/neuromod/asif/Projects/fieldtrip/template/electrode/standard_1005.elc');
                        cfg.badchannel =asrChans;
                        cfg.neighbours  = neighbours;
                        data_clean   = ft_channelrepair(cfg, data_clean); % make sure to use NON-REF data
                    end
                end
                %%  automated EEGLAB ICA based cleaning
                if(0 )
                    % merge layout
                    % Wronski dataset (300000 series) is EGI cap
                    if(contains(pwd,'sub-300000'))
                        EGI_layout=load('GSN-HydroCel-65_1.0.mat')
                        layout=EGI_layout.lay;
                    else
                        A= lay.lay.label;
                        B= data_clean.label;
                        layout=lay.lay;
                        [tf, idx] = ismember(lower(B),lower(A));
                        [tf2, idx2] = ismember(lower(A),lower(B));
                        layout.label = layout.label(idx);
                        data_clean.label=A(idx);
                        layout.pos=layout.pos(idx,:);
                        layout.width=layout.width(idx,:);
                        layout.height=layout.height(idx,:);
                        layout.pos = layout.pos*1;
                    end

                    % ICA Pipeline

                    try
                        data_clean=ft2eeglab_ICA_clean(data_clean,layout,0,1);
                    catch
                        try
                            % try using fastICA instead?
                            warning('run-ICA failed, going to FAST-ICA instead')
                            data_clean=ft2eeglab_ICA_clean(data_clean,layout,01,1);
                        catch
                            warning('cant do ICA, skipping for now!');
                        end
                    end

                    %                 data_clean=ft_ICA_clean(data_clean,layout,1);


                end



                %% additional line noise removal via zapline
                if(0)
                    % low freq line noises
                    % do this PER channel
                    for chan=1:size(data_clean.label,1)

                        [zappedData, zaplineConfig, analyticsResults, plothandles] = clean_data_with_zapline_plus(data_clean.trial{1,1}(chan,:),data_clean.fsample,'minfreq',0.5,'maxfreq',8);
                        if(length(analyticsResults.NremoveFinal))
                            for pp=1:size(plothandles)
                                figure(plothandles(pp));
                                suptitle([ actSu ' - block-' num2str(sess) ' - ' task ' -  ' data_clean.label{chan} ])
                                %% save snap
                                mkdir_if_not_exist([qcPath '/' actSu '/linenoise']);
                                fN=append(qcPath, "/", actSu, "/linenoise/lineNoise_", task, "_", num2str(sess), "_", data_clean.label{chan}, "_", num2str(analyticsResults.noisePeaks(pp)), ".png");
                                saveas(plothandles(pp),fN); close(plothandles(pp));
                            end
                        end
                        %                % high freq line noises
                        %                [zappedData, zaplineConfig, analyticsResults, plothandles] = clean_data_with_zapline_plus(zappedData,data_clean.fsample,'minfreq',15,'maxfreq',80);
                        %                 if(length(analyticsResults.NremoveFinal))
                        %                     for pp=1:size(plothandles)
                        %                         figure(plothandles(pp));
                        %                         suptitle([ actSu ' - block-' num2str(sess) ' - ' task ])
                        %                         %% save snap
                        %                         mkdir_if_not_exist([qcPath '/' actSu]);
                        %                         fN=append(qcPath, "/", actSu, "/lineNoise_", task, "_", num2str(sess), "_", num2str(analyticsResults.noisePeaks(pp)), ".png");
                        %                         saveas(plothandles(pp),fN); close(plothandles(pp));
                        %                     end
                        %                 end
                        close all; clear plothandles; clear analyticsResults;

                        data_clean.trial{1,1}(chan,:)=zappedData;
                    end

                    %                error('s')
                end
                %% overall cleaning
                if(0)
                    cfg=[];
                    % reref options
                    cfg.reref = 'yes';
                    cfg.refmethod = 'avg';
                    cfg.refchannel ='all';
                    data_clean_ref = ft_preprocessing(cfg,data_clean);

                    % use cleanData EEGlab plugin to ID bad chans
                    Signal=struct;Signal.data=data_clean_ref.trial{1,1};Signal.srate=data_clean.fsample;
                    Signal.nbchan=length(data_clean_ref.label);
                    % make a chanlocs
                    if(contains(pwd,'sub-300000')) % EGI has the chanlocs embedded in hdr.orig.chanlocs
                        Signal.chanlocs=data_clean.hdr.orig.chanlocs;
                    elseif(contains(pwd,'sub-02')) % MODMA 128 EGI cap
                        load('/autofs/space/tonetti_001/users/asif/Projects/MODMA/EEG_128channels_ERP_lanzhou_2015/chan_info_egi_128.mat');
                        Signal.chanlocs = chanlocs;

                    else
                        cfg=[];
                        cfg.elec=ft_read_sens('/autofs/vast/neuromod/asif/Projects/fieldtrip/template/electrode/standard_1005.elc');
                        % replace with correct order and prune unused
                        A= cfg.elec.label;
                        B= data_clean.label;
                        [tf, idx] = ismember(lower(B),lower(A));
                        cfg.elec.label = cfg.elec.label(idx);
                        cfg.elec.elecpos=cfg.elec.elecpos(idx,:);

                        Signal.chanlocs = struct('labels', cfg.elec.label, 'X', mat2cell(cfg.elec.elecpos(:,1), Signal.nbchan,1), 'Y', mat2cell(cfg.elec.elecpos(:,2), Signal.nbchan,1),'Z', mat2cell(cfg.elec.elecpos(:,3), Signal.nbchan,1));
                        Signal.chanlocs = convertlocs(Signal.chanlocs, 'cart2all');
                    end

                    Signal.etc=struct;

                    [signal badChans2] = clean_channels(Signal,[],[],[],0.7);
                    disp(data_clean_ref.label(badChans2));

                    if(length(find(badChans2)))

                        if(contains(pwd,'sub-300000')) % EGI has the chanlocs embedded in hdr.orig.chanlocs
                            Signal.chanlocs=data_clean.hdr.orig.chanlocs;
                            EGI_layout=load('GSN-HydroCel-65_1.0.mat')
                            % load neighbours and layout file
                            layout=EGI_layout.lay;
                            cfgmr               = [];
                            cfgmr.layout		=  layout;
                            cfgmr.method        = 'distance';
                            cfgmr.neighbourdist = .15;
                            neighbours = ft_prepare_neighbours(cfgmr);
                            load('/autofs/space/tonetti_001/users/asif/Projects/TDBRAIN-dataset/elife_dataset/Wronski/eeg/chanpos/neighbours.mat'); % output is neighbours
                            cfg=[];
                            cfg.method='spline';
                            cfg.elec=ft_read_sens('/autofs/vast/neuromod/asif/Projects/fieldtrip/template/electrode/GSN-HydroCel-65_1.0.sfp');
                            cfg.neighbours=neighbours;
                            cfg.senstype = 'eeg';

                        elseif(contains(pwd,'sub-02')) % MODMA 128 EGI cap
                            cfgmr=[];
                            cfg=[];
                            cfg.elec=ft_read_sens(chanlocs);
                            cfgmr.layout=ft_prepare_layout(cfg);
                            cfgmr.method        = 'distance';
                            cfgmr.neighbourdist = .2;

                            neighbours = ft_prepare_neighbours(cfgmr);

                            cfg=[];
                            cfg.method='spline';
                            cfg.elec=ft_read_sens('/autofs/space/tonetti_001/users/asif/Projects/MODMA/MODMA_EGI_cap_chanlocs.sfp');

                            cfg.neighbours=neighbours;
                            cfg.senstype = 'eeg';

                        else

                            cfgmr               = [];
                            cfgmr.layout		=  lay.lay;

                            cfgmr.method        = 'distance';
                            cfgmr.neighbourdist = .15;

                            neighbours = ft_prepare_neighbours(cfgmr);


                            cfg             = [];
                            cfg.method      = 'spline';
                            cfg.layout      = lay.lay;
                            cfg.elec=ft_read_sens('/autofs/vast/neuromod/asif/Projects/fieldtrip/template/electrode/standard_1005.elc');
                            cfg.badchannel = ft_channelselection(data_clean_ref.label(badChans2), data_clean.label);
                            cfg.neighbours  = neighbours;
                        end

                        data_clean   = ft_channelrepair(cfg, data_clean); % make sure to use NON-REF data
                    end
                end

                if(0)
                    cfg = []
                    cfg.method   = 'summary';
                    cfg.keeptrial = 'yes';
                    cfg.keepchannel = 'repair';
                    cfg.neighbours = neighbours;
                    cfg.preproc.bpfilter='yes';cfg.preproc.bpfreq=[1 45];
                    data_clean = ft_rejectvisual(cfg,data_clean);
                end


                %% check to make sure no NaNs in data
                actTrl=(data_clean.trial{1,1});
                if(sum(isnan(actTrl(:))))
                    error('nan value in data_clean, aborting');
                end
                
                %% Correct the labels
                label_flag=0;
                for i=1:size(data_clean.label,1)
                    if strcmp(data_clean.label{i}, 'T3')
                        label_flag = 1;
                        data_clean.label{i} = 'T7';
                    end
                    if strcmp(data_clean.label{i}, 'T4')
                        data_clean.label{i} = 'T8';

                    end
                    if strcmp(data_clean.label{i}, 'T5')
                        data_clean.label{i} = 'P7';

                    end
                    if strcmp(data_clean.label{i}, 'T6')
                        data_clean.label{i} = 'P8';

                    end
                end
                if(label_flag)
                    disp('corrected labels.');

                end
                %% visual inspect

                if(0)
                    cfg  = [];
                    %                 cfg.viewmode   = 'butterfly';
                    cfg.preproc.bpfilter='no';
                    cfg.preproc.bpfreq=[1 22];
                    cfg.continuous='yes';
                    cfg.blocksize=10;
                    ft_databrowser(cfg, data_clean);
                    drawnow

                end

                %% spectog and snapshot again
                if(1)
                    gcf=qa_computeSpectrum(data_clean,1);
                    suptitle([ actSu ' - block-' num2str(sess) ' - ' task ])
                    %% save snap
                    mkdir_if_not_exist([qcPath '/' population '/' actSu]);
                    fN=append(qcPath, population, "/", actSu, "/cleaned_spectrogram_", task, "_", num2str(sess), ".png");
                    saveas(gcf,fN); close(gcf);
                end
                %% save it
                save([analysis_path '/' population '/' actSu '/'  fn(1:end-7) 'clean.mat'],'data_clean','-v7.3')

           

            end

        end
    end
end
%% EPOCH/Trial generation - RESTING_STATE ONLY
% 1. Import events
% 2. Cut trials by event struct
cd(analysis_path)
 
 for dataset=1:2  
    % go into HC folder or MDD folder
    if(dataset==1)
        population='HC';
    else
        population='MDD';
    end
    cd([analysis_path '/' population])
    suList=dir('*sub*');
    for subj=1:size(suList,1)
        cd([analysis_path '/' population])

        actSu=suList(subj).name
        cd(actSu);
        sessions= dir(['*ses_*' task '*clean.mat']);
        for sess=1:size(sessions,1)
            fn=dir(['*' sessions(sess).name '*']);
            fn=fn.name
    
            if(~exist([analysis_path '/' population '/' actSu '/' fn(1:end-9) 'epoched.mat'],'file'))
                %% epoch
                 load(fn);
                 
                 
                   % cut into 4 sec segments, 50% OL
                   % split the data into 6 sec chunks (as orig)
                    cfg=[];
                    cfg.length = 4.00;
                    cfg.overlap = 0.5;

    
                 data_epoched=ft_redefinetrial(cfg,data_clean);
                 
                %% save data
                mkdir_if_not_exist([analysis_path '/' population '/' actSu]);
                save([analysis_path '/' population '/' actSu '/' fn(1:end-9) 'epoched.mat'],'data_epoched','-v7.3')
            else
                % warning('already processed, skipping..')
            end
    
        end
    end % completion check
 end 
 disp('done epoch')
%% FEATURE EXTRACTION (Trial pow, LFA, FC analysis, ...)

DO_Z = 0 

if(DO_Z)
    zfx='_Z';
else
    zfx='_nonZ';
end

cd(analysis_path)
 
 for dataset=[ 1 2 ]  
    % go into HC folder or MDD folder
    if(dataset==1)
        population='HC';
    else
        population='MDD';
    end
    cd([analysis_path '/' population])
    suList=dir('*sub*');
    for subj=1:size(suList,1)
        cd([analysis_path '/' population])

        actSu=suList(subj).name
        % actSu=suList{subj}
        cd(actSu);
        sessions= dir(['*ses_*' task '*data_epoched.mat']);
        for sess=1:size(sessions,1)
            tic;
            fn=dir(['*' sessions(sess).name '*']);
            fn=fn.name;
            %% FOOOF EXP
            if(~exist([analysis_path '/' population '/' actSu '/' fn(1:end-11) 'FOOOF_1_45' zfx '.mat'],'file'))
                 load(fn);
                    disp('fooof...')

                % re-reference
    
                if(01)
                    cfg=[];
                    % reref options
                    cfg.reref = 'yes';
                    cfg.refmethod = 'avg';
                    cfg.refchannel = 'all';
                    data_epoched = ft_preprocessing(cfg,data_epoched);
                end
                  % z trans
                if(DO_Z)
                    cfg=[];
                    cfg.method='perchannel';
                    data_epoched=ft_channelnormalise(cfg,data_epoched);
                    
                end
                fooof={};
                full_fooof_mat=compute_fooof(data_epoched,1,[1 45]); % trial x params x chan
                % avg trials
                full_fooof_mat=squeeze(nanmean(full_fooof_mat,1));
                % 2nd dimension is exponent, 1st is offset
                full_fooof_mat_exp=squeeze(full_fooof_mat(2,:));
                full_fooof_mat_offset=squeeze(full_fooof_mat(1,:));
                fooof{1}=full_fooof_mat_exp;
                fooof{2}=full_fooof_mat_offset;
                % SAVE
                save([analysis_path '/' population '/' actSu '/' fn(1:end-11) 'FOOOF_1_45' zfx '.mat'],'fooof','-v7.3')
            else
                % load([analysis_path '/' population '/' actSu '/' fn(1:end-11) 'FOOOF_1_45' zfx '.mat']);
                % full_fooof_mat_exp=fooof{1};
                % full_fooof_mat_offset=fooof{2};
            end
            if(~exist([analysis_path '/' population '/' actSu '/' fn(1:end-11) 'FOOOF_30_45' zfx '.mat'],'file'))
                 load(fn);
    
                % re-reference
                    disp('fooof...')

                if(01)
                    cfg=[];
                    % reref options
                    cfg.reref = 'yes';
                    cfg.refmethod = 'avg';
                    cfg.refchannel = 'all';
                    data_epoched = ft_preprocessing(cfg,data_epoched);
                end
                  % z trans
                if(DO_Z)
                    cfg=[];
                    cfg.method='perchannel';
                    data_epoched=ft_channelnormalise(cfg,data_epoched);
                    
                end
                fooof3045={};
                % recompute fooof in narrow band (30-45)
                full_fooof_mat3045=compute_fooof(data_epoched,1,[30 45]); % trial x params x chan
                % avg trials
                full_fooof_mat3045=squeeze(nanmean(full_fooof_mat3045,1));
                % 2nd dimension is exponent, 1st is offset
                full_fooof_mat_exp3045=squeeze(full_fooof_mat3045(2,:));
                full_fooof_mat_offset3045=squeeze(full_fooof_mat3045(1,:));
                fooof3045{1}=full_fooof_mat_exp3045;
                fooof3045{2}=full_fooof_mat_offset3045;
                % SAVE
                save([analysis_path '/' population '/' actSu '/' fn(1:end-11) 'FOOOF_30_45' zfx '.mat'],'fooof3045','-v7.3')
            else
                % load([analysis_path '/' population '/' actSu '/' fn(1:end-11) 'FOOOF_30_45' zfx '.mat']);
                % full_fooof_mat_exp3045=fooof3045{1};
                % full_fooof_mat_offset3045=fooof3045{2};
            end
            %% Compute deFOOOFed pow/FFT
            if(~exist([analysis_path '/' population '/' actSu '/' fn(1:end-11) 'deFOOOFed_power' zfx '.mat'],'file'))
                
                disp('defoofing')
                
                load(fn);
    
                % re-reference
    
                if(01)
                    cfg=[];
                    % reref options
                    cfg.reref = 'yes';
                    cfg.refmethod = 'avg';
                    cfg.refchannel = 'all';
                    data_epoched = ft_preprocessing(cfg,data_epoched);
                end
               
               % z trans
                if(DO_Z)
                    cfg=[];
                    cfg.method='perchannel';
                    data_epoched=ft_channelnormalise(cfg,data_epoched);
                    
                end
                %            trial removal based on std dev
                if(0)
                    % thresh based trial removal
                    trls=[1:size(data_epoched.trial,2)];
                    rm=[];
                    for tt=1:size(data_epoched.trial,2)
                        actTrl=squeeze(data_epoched.trial{tt});
                        if(nansum(abs(actTrl(:))>200))
                            rm(end+1)=tt;
                        end
                        if(nansum(isnan(actTrl(:))))
                            rm(end+1)=tt;
                        end
                    end
                    cfg=[];
                    cfg.trials=[1:size(data_epoched.trial,2)];
                    cfg.trials(rm)=[];
                    data_epoched=ft_selectdata(cfg,data_epoched);

                end
                
                % compute the fractal and original spectra
                cfg               = [];
                cfg.foilim        = [1 45];
                cfg.pad           = 'nextpow2';
                cfg.tapsmofrq     = 2;
                cfg.method        = 'mtmfft';
                cfg.output        = 'fooof_aperiodic';
                fractal = ft_freqanalysis(cfg, data_epoched);

                cfg.output        = 'pow';
                original = ft_freqanalysis(cfg, data_epoched);
                % subtract the fractal component from the power spectrum
                cfg               = [];
                cfg.parameter     = 'powspctrm';
                cfg.operation     = 'x2-x1';
                oscillatory = ft_math(cfg, fractal, original);

                % original implementation by Donoghue et al. 2020 operates through the semilog-power
                % (linear frequency, log10-power) space and transformed back into linear-linear space.
                % thus defining an alternative expression for the oscillatory component as the quotient of
                % the power spectrum and the fractal component
                cfg               = [];
                cfg.parameter     = 'powspctrm';
                cfg.operation     = 'x2./x1';  % equivalent to 10^(log10(x2)-log10(x1))
                oscillatory_alt = ft_math(cfg, fractal, original);
 
                if(0)
                    figure();
                    % display the spectra on a log-log scale
                    subplot(1,2,1); hold on;
                    % plot(log(original.freq), log(original.powspctrm),'k');
                    % plot(log(fractal.freq), log(fractal.powspctrm));
                    plot(log(fractal.freq), log(oscillatory.powspctrm));
                    xlabel('log-freq'); ylabel('log-power'); grid on;
                    legend({'original','fractal','oscillatory = spectrum-fractal'},'location','southwest');

                    subplot(1,2,2); hold on;
                    % plot(log(original.freq), log(original.powspctrm),'k');
                    % plot(log(fractal.freq), log(fractal.powspctrm));
                    plot(log(oscillatory_alt.freq), log(oscillatory_alt.powspctrm));
                    xlabel('log-freq'); ylabel('log-power'); grid on;
                    legend({'original','fractal','oscillatory = spectrum/fractal'},'location','southwest');
                    title('oscillatory = spectrum / fractal');
                end

                data_deFOOOFed=oscillatory_alt;
                save([analysis_path '/' population '/' actSu '/' fn(1:end-11) 'deFOOOFed_power' zfx '.mat'], 'data_deFOOOFed', '-v7.3');

            end

            %% LFA stats
            if(~exist([analysis_path '/' population '/' actSu '/' fn(1:end-11) 'LFA' zfx '.mat'],'file'))
                
                load(fn);
                disp('lfa...')
    
                % re-reference
    
                if(01)
                    cfg=[];
                    % reref options
                    cfg.reref = 'yes';
                    cfg.refmethod = 'avg';
                    cfg.refchannel = 'all';
                    data_epoched = ft_preprocessing(cfg,data_epoched);
                end
               % z trans
                if(DO_Z)
                    cfg=[];
                    cfg.method='perchannel';
                    data_epoched=ft_channelnormalise(cfg,data_epoched);
                    
                end

                %   trial removal based on std dev
                if(0)
                    % thresh based trial removal
                    trls=[1:size(data_epoched.trial,2)];
                    rm=[];
                    for tt=1:size(data_epoched.trial,2)
                        actTrl=squeeze(data_epoched.trial{tt});
                        if(nansum(abs(actTrl(:))>200))
                            rm(end+1)=tt;
                        end
                        if(nansum(isnan(actTrl(:))))
                            rm(end+1)=tt;
                        end
                    end
                    cfg=[];
                    cfg.trials=[1:size(data_epoched.trial,2)];
                    cfg.trials(rm)=[];
                    data_epoched=ft_selectdata(cfg,data_epoched);

                end
                
               
                %% FFT
                cfg = [];
                cfg.foi           = [0.5:0.25:45];
                cfg.method       = 'mtmfft';
                cfg.output       = 'powandcsd';
                cfg.output       = 'pow';
                % cfg.keeptapers= 'no';
                cfg.taper='hanning';
                % cfg.tapsmofrq=1;
                cfg.pad = 'nextpow2';
                cfg.keeptrials = 'yes';
                freq3 = ft_freqanalysis(cfg, data_epoched);

                if(0)

                    cfg=[];
                    cfg.layout=layout;
                    cfg.xlim=[8 12];
                    ft_topoplotTFR(cfg,freq3)

                end


                actChansFrontal={'Fz','F3','F4','E12','E6','E60'};
                        actChansLFA={'F3','E12'}; 
                        actChansRFA={'F4','E60'};

                     % for wronski 64 ch EGI, use E24 (F3) E11 (Fz) and
                     % E124
                     if(size(freq3.label,1)>126)
                        actChansFrontal={'E24','E11','E124'};
                        actChansLFA={'E24'};
                        actChansRFA={'E124'};
                     end
                    % https://www.fieldtriptoolbox.org/assets/img/template/layout/gsn-hydrocel-65.mat.png
                    actChansFrontalIdx=getChanIdx(actChansFrontal,freq3.label);
                    actChansLeftFrontalIdx=getChanIdx(actChansLFA,freq3.label);
                    actChansRightFrontalIdx=getChanIdx(actChansRFA,freq3.label);

                    % COMPUTE TIME SPENT WITH LFA (Left Frontal alpha
                    % asymm)
                    LFA_mat=[];
                    for trl=1:size(freq3.powspctrm,1)
                        actTrl=squeeze(freq3.powspctrm(trl,:,:));
                        LFA=actTrl(actChansLeftFrontalIdx,[nearest(freq3.freq,8):nearest(freq3.freq,12)]);
                        RFA=actTrl(actChansRightFrontalIdx,[nearest(freq3.freq,8):nearest(freq3.freq,12)]);

                        FAA=nanmean(log(LFA(:)))-nanmean(log(RFA(:)));
                        LFA_mat(end+1)=FAA;
                    end
                    disp(nanmean(LFA_mat))


                    % DO A DEFOOOFED VERSION ?! --> WILL NEED TO SUBSET
                    % TRIALS

                    
                    LFA_mat_osc=nan(1,size(data_epoched.trial,2));
                    LFA_mat_osc_alt=nan(1,size(data_epoched.trial,2));
                    tic
                    parfor (trl=1:size(data_epoched.trial,2))

                        data_sub=data_epoched;
                        data_sub.trial=[];data_sub.time=[];data_sub.sampleinfo=[];
                        data_sub.trial{1}=data_epoched.trial{trl};
                        data_sub.time{1}=data_epoched.time{trl};
                        data_sub.sampleinfo=data_epoched.sampleinfo(trl,:);

                         
                        cfg=[];
                       
                        cfg.length = 01.000;
                        cfg.overlap = 0.5;
                        data_sub=ft_redefinetrial(cfg,data_sub);

    
                        % compute the fractal and original spectra
                        cfg               = [];
                        cfg.foilim        = [1 45];
                        cfg.pad           = 'nextpow2';
                        cfg.tapsmofrq     = 2;
                        cfg.method        = 'mtmfft';
                        cfg.output        = 'fooof_aperiodic';
                        fractal = ft_freqanalysis(cfg, data_sub);
        
                        cfg.output        = 'pow';
                        original = ft_freqanalysis(cfg, data_sub);
                        % subtract the fractal component from the power spectrum
                        cfg               = [];
                        cfg.parameter     = 'powspctrm';
                        cfg.operation     = 'x2-x1';
                        oscillatory = ft_math(cfg, fractal, original);

                        LFA=oscillatory.powspctrm(actChansLeftFrontalIdx,[nearest(fractal.freq,8):nearest(fractal.freq,12)]);
                        RFA=oscillatory.powspctrm(actChansRightFrontalIdx,[nearest(fractal.freq,8):nearest(fractal.freq,12)]);

                        FAA=nanmean((LFA(:)))-nanmean((RFA(:)));
                        LFA_mat_osc(trl)=FAA;

        
                        % original implementation by Donoghue et al. 2020 operates through the semilog-power
                        % (linear frequency, log10-power) space and transformed back into linear-linear space.
                        % thus defining an alternative expression for the oscillatory component as the quotient of
                        % the power spectrum and the fractal component
                        cfg               = [];
                        cfg.parameter     = 'powspctrm';
                        cfg.operation     = 'x2./x1';  % equivalent to 10^(log10(x2)-log10(x1))
                        oscillatory_alt = ft_math(cfg, fractal, original);

                        LFA=oscillatory_alt.powspctrm(actChansLeftFrontalIdx,[nearest(fractal.freq,8):nearest(fractal.freq,12)]);
                        RFA=oscillatory_alt.powspctrm(actChansRightFrontalIdx,[nearest(fractal.freq,8):nearest(fractal.freq,12)]);

                        FAA=nanmean((LFA(:)))-nanmean((RFA(:)));
                        LFA_mat_osc_alt(trl)=FAA;

                       if(0)
                         figure;plot(original.freq, (original.powspctrm(3,:)))
                         hold on;plot(original.freq, (oscillatory.powspctrm(3,:)))
                        end
                        % mkdir_if_not_exist([qcPath '/' actSu]);
                        % fN=append(qcPath, "/", actSu, "/POWER_", task, "_", num2str(sess), zfx, ".png");
                        % saveas(h1,fN); close(h1);
                        %
                    end
                    toc
                    disp(nanmean(LFA_mat_osc))
                    disp(nanmean(LFA_mat_osc_alt))

                    LFA=struct;
                    LFA.original=LFA_mat;
                    LFA.oscillatory=LFA_mat_osc;
                    LFA.oscillatory_alt=LFA_mat_osc_alt;
                    % error('j')
                    % figure;plot(LFA.original);hold on;plot(LFA.oscillatory);hline(0);legend({'orig','osc'});drawnow;


                save([analysis_path '/' population '/' actSu '/' fn(1:end-11) 'LFA' zfx '.mat'], 'LFA', '-v7.3');

            end

            %% compute abs. power
            if(~exist([analysis_path '/' population '/' actSu '/' fn(1:end-11) 'absolute_power' zfx '.mat'],'file'))
                 load(fn);
    
                % re-reference
                    disp('pow...')

                if(01)
                    cfg=[];
                    % reref options
                    cfg.reref = 'yes';
                    cfg.refmethod = 'avg';
                    cfg.refchannel = 'all';
                    data_epoched = ft_preprocessing(cfg,data_epoched);
                end
               
                powerMat = computePower(data_epoched);
                save([analysis_path '/' population '/' actSu '/' fn(1:end-11) 'absolute_power' zfx '.mat'], 'powerMat', '-v7.3');
            end


            %% compute rel. power
            if(~exist([analysis_path '/' population '/' actSu '/' fn(1:end-11) 'relative_power' zfx '.mat'],'file'))           
                 load(fn);
    
                % re-reference
    
                if(01)
                    cfg=[];
                    % reref options
                    cfg.reref = 'yes';
                    cfg.refmethod = 'avg';
                    cfg.refchannel = 'all';
                    data_epoched = ft_preprocessing(cfg,data_epoched);
                end
                  % z trans
                if(DO_Z)
                    cfg=[];
                    cfg.method='perchannel';
                    data_epoched=ft_channelnormalise(cfg,data_epoched);
                    
                end
                relPowerMat = computePower(data_epoched,1);
                save([analysis_path '/'  population '/' actSu '/' fn(1:end-11) 'relative_power' zfx '.mat'], 'relPowerMat', '-v7.3');
            end

            %% CONN
            if(~exist([analysis_path '/' population '/' actSu '/' fn(1:end-11) 'CONN' zfx '.mat'],'file'))
                           
                load(fn);
                disp('conn...')

                % re-reference
    
                if(01)
                    cfg=[];
                    % reref options
                    cfg.reref = 'yes';
                    cfg.refmethod = 'avg';
                    cfg.refchannel = 'all';
                    data_epoched = ft_preprocessing(cfg,data_epoched);
                end
                  % z trans
     
                if(DO_Z)
                    cfg=[];
                    cfg.method='perchannel';
                    data_epoched=ft_channelnormalise(cfg,data_epoched);
                    
                end
            
                 
                 
                 %% TFR
                cfg=[];
                cfg.length = 1.00;
                cfg.overlap = 0;
                data_minisub = ft_redefinetrial(cfg,data_epoched);

                cfg = [];
                cfg.foi           = [1:0.25:45];
                cfg.method       = 'mtmfft';
                cfg.output       = 'powandcsd';
                cfg.keeptapers= 'no';
                cfg.taper='hanning';
                cfg.pad = 'nextpow2';
                cfg.keeptrials = 'yes';
                freq3 = ft_freqanalysis(cfg, data_minisub);
 
                % calculate conn matrices (WPLI, WPPC, iCoh, etc.)
                conn=struct;
                    cfg=[];
                    cfg.channel = 'all';
                    cfg.method = 'wpli_debiased';
                    stat_wpli = ft_connectivityanalysis(cfg, freq3);
                    conn.stat_wpli=ft_checkdata(stat_wpli,'cmbstyle','full');
                      cfg=[];
                    cfg.channel = 'all';
                    cfg.method = 'wppc';
                    stat_wppc = ft_connectivityanalysis(cfg, freq3);
                    conn.stat_wppc=ft_checkdata(stat_wppc,'cmbstyle','full');
                     
                     cfg=[];
                    cfg.channel = 'all';
                    cfg.method = 'coh';
                    cfg.complex='imag';
                    stat_coh = ft_connectivityanalysis(cfg, freq3);
                    conn.stat_imag_coh=ft_checkdata(stat_coh,'cmbstyle','full');

                      cfg=[];
                    cfg.channel = 'all';
                    cfg.method = 'coh';
                    stat_coh = ft_connectivityanalysis(cfg, freq3);
                    conn.stat_coh=ft_checkdata(stat_coh,'cmbstyle','full');

             
                %% SAVE  
                save([analysis_path '/' population '/' actSu '/' fn(1:end-11) 'CONN' zfx '.mat'],'conn','-v7.3')
            end
            %% COMPUTE MSWPE (or load prev completed)
           
            continue; % remove or comment this, if you want to run MSWPE
            if(0 & ~exist([analysis_path '/' actSu '/' fn(1:end-11) 'MSWPE' zfx '.mat'],'file'))               
                MSWPE_mat={};
                for freq=1:5
                    switch freq
                        case 1
                            LOW_CUT = 1;
                            HIGH_CUT = 4;
                        case 2
                            LOW_CUT = 5;
                            HIGH_CUT = 7;
                        case 3
                            LOW_CUT = 8;
                            HIGH_CUT = 12;
                        case 4
                            LOW_CUT = 13;
                            HIGH_CUT = 30;
                        case 5
                            LOW_CUT = 31;
                            HIGH_CUT = 45;
                    end
                    % DO BANDPASS RESOLVED MSE
                    bpMPEmat=[];
                    % do simple bandpass
                    % cfg.bpfilter='yes';
                    % cfg.bpfreq=[LOW_CUT HIGH_CUT];
                    % cfg.instabilityfix = 'reduce'
                    % disp('Problem here')
                    cfg = [];
                    cfg.hpfilter = 'yes';
                    cfg.hpfreq = LOW_CUT;
                    cfg.hpfiltertype = 'but';
                    cfg.instabilityfix = 'split';
                    data_hp = ft_preprocessing(cfg, data_epoched);
                    disp('Problem here HP')     % ERASE AFTERWARDS
                    
                    cfg = [];
                    cfg.lpfilter = 'yes';
                    cfg.lpfreq = HIGH_CUT;
                    cfg.lpfiltertype = 'but';
                    cfg.instabilityfix = 'split';
                    freq4 = ft_preprocessing(cfg, data_hp);
                    disp('Problem here LP')     % ERASE AFTERWARDS

                    % freq4=ft_preprocessing(cfg,data_epoched);
                    % WPE
                    % resample to 256
                    cfg = [];
                    cfg.resamplefs=256;
                    data_epoched=ft_resampledata(cfg,data_epoched);
                    SC=fix(256/0.5); % in order to ensure we can estimate at least .5 Hz
    
                    mpeMat=compute_msWPE(freq4,SC,0);
    
                    % Scale 1 is samplingRate/1 = 256 Hz freq
                    % scale (end) or SC is samplingRate/SC = 256/128 = 2 Hz
                    % interpolate everything in between?
                    scaleIndices=[1:SC];
                    scaleIndices=freq4.fsample./scaleIndices;
                    lowF=(nearest(scaleIndices,LOW_CUT));
                    highF=(nearest(scaleIndices,HIGH_CUT));
                    bpMPE=squeeze(nanmean(mpeMat(:,:,[highF:lowF]),3));
                    bpMPE=squeeze(nanmean(bpMPE,2));
                    
                    MSWPE_mat{freq}=bpMPE;
                end
                % SAVE MSWPE MAT
                save([analysis_path '/' actSu '/' fn(1:end-11) 'MSWPE' zfx '.mat'],'MSWPE_mat','-v7.3')
       
            end % if check
             toc;
             clear data_epoched;
        
       
        
        
        end % block
    
    end % subj
end  
disp('done feature proc.')
%% for MLSC job - exit here
exit;
%% Corect the labels - DO not use anymore, since already corrected in preproc

cd(analysis_path)
 
 for dataset=1:2  
    % go into HC folder or MDD folder
    if(dataset==1)
        population='HC';
    else
        population='MDD';
    end
    cd([analysis_path '/' population])
    suList=dir('*sub*');
    for subj=1:size(suList,1)
        cd([analysis_path '/' population])
        label_flag = 0;
        actSu=suList(subj).name
        % actSu=suList{subj}
        cd(actSu);
        sessions = dir(['*ses_*' task '*CONN.mat']);
        for sess = 1:size(sessions,1)
            fn=dir(['*' sessions(sess).name '*']);
            fn=fn.name;
            load(sessions(sess).name)
            label_flag = 0;
            for i = 1:size(conn.stat_coh.label)
                    if strcmp(conn.stat_coh.label{i}, 'T3')
                            label_flag = 1;
                            conn.stat_coh.label{i} = 'T7';
                            conn.stat_imag_coh.label{i} = 'T7';
                            conn.stat_wpli.label{i} = 'T7';
                            conn.stat_wppc.label{i} = 'T7';
                    end
                    if strcmp(conn.stat_coh.label{i}, 'T4')
                            conn.stat_coh.label{i} = 'T8';
                            conn.stat_imag_coh.label{i} = 'T8';
                            conn.stat_wpli.label{i} = 'T8';
                            conn.stat_wppc.label{i} = 'T8';
                    end
                    if strcmp(conn.stat_coh.label{i}, 'T5')
                            conn.stat_coh.label{i} = 'P7';
                            conn.stat_imag_coh.label{i} = 'P7';
                            conn.stat_wpli.label{i} = 'P7';
                            conn.stat_wppc.label{i} = 'P7';
                    end
                    if strcmp(conn.stat_coh.label{i}, 'T6')
                            conn.stat_coh.label{i} = 'P8';
                            conn.stat_imag_coh.label{i} = 'P8';
                            conn.stat_wpli.label{i} = 'P8';
                            conn.stat_wppc.label{i} = 'P8';
                    end
            end
            if label_flag == 1
                save([analysis_path '/' population '/' actSu '/' fn(1:end-8) 'CONN.mat'],'conn','-v7.3')
            end
        end
    end
 end
%% CSV output - CONNECTIVITY

DO_SPARSE = 0

SPARSE_CHAN_ROI ={'F7','Fz','F8','T7','Cz','T8','P7','Pz','P8'};

DO_Z = 0

if(DO_Z)
    zfx='_Z';
else
    zfx='_nonZ';
end

cd(analysis_path)
 

colheads={'Population','Subject','Session',...
    'Channel_1','Channel_2','FreqBand','Coherence','Imag_Coherence','WPLI','WPPC'};
masterTable=cell([1+(55*19*5)],length(colheads));
masterTable(1,:)=colheads;
idx=2;

freqLabels={'Delta','Theta','Alpha','Beta','Gamma'};

for dataset=1:2
    % go into HC folder or MDD folder
    if(dataset==1)
        population='HC';
    else
        population='MDD';
    end
    cd([analysis_path '/' population])
    suList=dir('*sub*');
    for subj=1:size(suList,1)
        cd([analysis_path '/' population])

        actSu=suList(subj).name
        % actSu=suList{subj}
        cd(actSu);
        % if(DO_SPARSE)
        %         sessions= dir(['*ses_*' task '*CONN_SPARSE' zfx '.mat']);
        % else
                sessions= dir(['*ses_*' task '*data_CONN' zfx '.mat']);
        % end
        % error('s')
        for sess=1:size(sessions,1)
            fn=dir(['*' sessions(sess).name '*']);
            if(isempty(fn))
                continue;
            end
            sessNum=extractAfter(fn.name,'ses_');
            sessNum=sessNum(1);
            % load FC file
           
            load(fn.name); % output is conn
            
            for freq=1:5
                    switch freq
                        case 1 % delta
                            LOW_CUT = 1;
                            HIGH_CUT = 4;
                        case 2 % theta
                            LOW_CUT = 5;
                            HIGH_CUT = 7;
                        case 3 % alpha
                            LOW_CUT = 8;
                            HIGH_CUT = 12;
                        case 4 % beta
                            LOW_CUT = 13;
                            HIGH_CUT = 30;
                        case 5 % gamma
                            LOW_CUT = 31;
                            HIGH_CUT = 45;
                    end


                    [starting_freq] = nearest(conn.stat_wpli.freq,LOW_CUT);
                    [ending_freq]= nearest(conn.stat_wpli.freq,HIGH_CUT);

                    coh_average_freq_window=(mean(conn.stat_coh.cohspctrm(:,:,[starting_freq:ending_freq]),3));
                    imag_coh_average_freq_window=(mean(conn.stat_imag_coh.cohspctrm(:,:,[starting_freq:ending_freq]),3));
                    wpli_average_freq_window=(mean(conn.stat_wpli.wpli_debiasedspctrm(:,:,[starting_freq:ending_freq]),3));
                    wppc_average_freq_window=(mean(conn.stat_wppc.wppcspctrm(:,:,[starting_freq:ending_freq]),3));

                    for chan1=1:size(conn.stat_coh.label,1)

                        for chan2=1:size(conn.stat_coh.label,1)
                            
                            if(chan2==chan1)
                                continue;
                            end

                            if(DO_SPARSE)
                                if(~ismember(conn.stat_coh.label(chan1),SPARSE_CHAN_ROI))
                                    continue
                                end
                                if(~ismember(conn.stat_coh.label(chan2),SPARSE_CHAN_ROI))
                                    continue
                                end
                            end


                            chan_A=conn.stat_coh.label{chan1};
                            chan_B=conn.stat_coh.label{chan2};
        
        
                            coh_value=coh_average_freq_window(chan1,chan2);
                            imag_coh_value = imag_coh_average_freq_window(chan1,chan2);
                            wpli_value = wpli_average_freq_window(chan1,chan2);
                            wppc_value = wppc_average_freq_window(chan1,chan2);

                             masterTable(idx,:)={
                                    population,...
                                    actSu,...
                                    (sessNum),...
                                    chan_A,...
                                    chan_B,...
                                    freqLabels{freq},...
                                    num2str(coh_value),...
                                    num2str(imag_coh_value),...
                                    num2str(wpli_value),...
                                    num2str(wppc_value)
                                    };
                    
                                idx=idx+1;
                     

                        end
                    end
            end
            clear conn;
        end
    end
end
if(DO_SPARSE)
    cell2csv(['/autofs/vast/neuromod/tDCS_EEG/Projects/16_ENGINE/analysis/group/master_connectivity_SPARSE_' zfx '_' date  '.csv'],masterTable);

else
    cell2csv(['/autofs/vast/neuromod/tDCS_EEG/Projects/16_ENGINE/analysis/group/master_connectivity_FULL_' zfx '_' date  '.csv'],masterTable);

end
disp('saved.')


%% CSV Output POW, fOOOF, etc

DO_Z = 0 
if(DO_Z)
    zfx='_Z';
else
    zfx='_nonZ';
end


colheads={'Subject','Population','Session',...
    'Channel','FreqBand', 'Abs_Power','Rel_Power','Osc_Power','FOOOF_1_45', ...
    'FOOOF_1_45_offset', 'FOOOF_30_45', 'FOOOF_30_35_offset'};
masterTable=cell([1+(55*19*5)],length(colheads));
masterTable(1,:)=colheads;
idx=2;
freqLabels={'Delta','Theta','Alpha','Beta','Gamma'};

cd([analysis_path])
suList=dir('*sub*');
 
for dataset=1:2
    % go into HC folder or MDD folder
    if(dataset==1)
        population='HC';
    else
        population='MDD';
    end
    cd([analysis_path '/' population])
    suList=dir('*sub*');
    for subj=1:size(suList,1)
        cd([analysis_path '/' population])

        actSu=suList(subj).name
        % actSu=suList{subj}
        cd(actSu);

            sessions= dir(['*ses_*' task '*data_epoched.mat']);
            rel_power = dir(['ses_*' task '*relative_power' zfx '.mat']);
            abs_power = dir(['ses_*' task '*absolute_power' zfx '.mat']);
            osc_power = dir(['ses_*' task '*deFOOOFed_power' zfx '.mat']);
            FOOOF_1_45 = dir(['ses_*' task '*FOOOF_1_45' zfx '.mat']);
            FOOOF_30_45 = dir(['ses_*' task '*FOOOF_30_45' zfx '.mat']);
            % MSWPE = dir(['ses_*' task '*MSWPE' zfx '.mat']);
            epoched = dir(['*ses_*' task '*data_epoched.mat']);
        
            for sess=1:size(sessions,1)

                sessNumb=extractAfter(epoched(sess).name,'ses_');
                sessNumb=str2num(sessNumb(1));

                rel_power_sess = dir(['*' rel_power(sess).name '*']);
                abs_power_sess = dir(['*' abs_power(sess).name '*']);
                FOOOF_1_45_sess = dir(['*' FOOOF_1_45(sess).name '*']);
                FOOOF_30_45_sess = dir(['*' FOOOF_30_45(sess).name '*']);
                % MSWPE_sess = dir(['*' MSWPE(sess).name '*']);
                epoched_sess = dir(['*' epoched(sess).name '*']);
                osc_power_sess = dir(['*' osc_power(sess).name '*']);
        
        
                % load FC file
               
                load(rel_power_sess.name);
                load(abs_power_sess.name);
                load(FOOOF_1_45_sess.name);
                % load(FOOOF_30_45_sess.name);
                % load(MSWPE_sess.name);
                load(epoched_sess.name);
                load(osc_power_sess.name);
        
                full_fooof_mat_exp = fooof{1};
                full_fooof_mat_offset = fooof{2};
                full_fooof_mat_exp3045=fooof3045{1};
                full_fooof_mat_offset3045=fooof3045{2};
        
                for freq=1:5
                        switch freq
                            case 1 % delta
                                LOW_CUT = 1;
                                HIGH_CUT = 4;
                            case 2 % theta
                                LOW_CUT = 5;
                                HIGH_CUT = 7;
                            case 3 % alpha
                                LOW_CUT = 8;
                                HIGH_CUT = 12;
                            case 4 % beta
                                LOW_CUT = 13;
                                HIGH_CUT = 30;
                            case 5 % gamma
                                LOW_CUT = 31;
                                HIGH_CUT = 45;
                        end
                    % bpMPE=MSWPE_mat{freq};
                    avg_osc_pow=squeeze(data_deFOOOFed.powspctrm(:,[nearest(data_deFOOOFed.freq,LOW_CUT):nearest(data_deFOOOFed.freq,HIGH_CUT)]));
        
                    for ch=1:size(data_epoched.label,1)
                
                        masterTable{idx,1}= actSu;
                        masterTable{idx,2}= population;
                        masterTable{idx,3}= num2str(sessNumb);
                        masterTable{idx,4} = data_epoched.label{ch};
                        masterTable{idx,5} = freqLabels{freq};
                        masterTable{idx,6} = powerMat(ch,freq);
                        masterTable{idx,7} = relPowerMat(ch,freq);
                        masterTable{idx,8} = nanmean(avg_osc_pow(ch,:));
                        masterTable{idx,9} = full_fooof_mat_exp(ch);
                        masterTable{idx,10} = full_fooof_mat_offset(ch);
                        masterTable{idx,11} = full_fooof_mat_exp3045(ch);
                        masterTable{idx,12} = full_fooof_mat_offset3045(ch);
                        % masterTable{idx,13} = bpMPE(ch);
                
                        idx=idx+1;
                    end %end chan
                end % end freq
            end % sess

    end % subj
end % dataset

cell2csv(['/autofs/vast/neuromod/tDCS_EEG/Projects/16_ENGINE/analysis/group/ENGINE_restingState_' date zfx '.csv'],masterTable);

disp('saved.')



%% CSV Output - LFA stats

DO_Z = 0  
if(DO_Z)
    zfx='_Z';
else
    zfx='_nonZ';
end

colheads={'Dataset','Subject','Population','Task','Session',...
    'Type','LFA_mean','LFA_std', 'LFA_duration', 'LFA_duration_prct','LFA-RFA_transitions',...
   };

 
masterLFAtable=cell([1+(55*3)],length(colheads));
masterLFAtable(1,:)=colheads;
idx=2;

cd([analysis_path])
suList=dir('*sub*');

for dataset=1:2
    % go into HC folder or MDD folder
    if(dataset==1)
        population='HC';
    else
        population='MDD';
    end
    cd([analysis_path '/' population])
    suList=dir('*sub*');
    for subj=1:size(suList,1)
        cd([analysis_path '/' population])

        actSu=suList(subj).name
        % actSu=suList{subj}
        cd(actSu);

        sessions= dir(['*ses_*' task '*data_LFA' zfx '.mat']);



        for sess=1:size(sessions,1)

            clear LFA:
            lfa_power_sess = dir(['*' sessions(sess).name '*']);

            load(lfa_power_sess.name);


            for type=1:2
                if(type==1)
                    typeLabel='Original';
                    actLFA=LFA.original;
                else
                    typeLabel='Oscillatory'
                    actLFA=LFA.oscillatory;
                end

                meanLFA=nanmean(actLFA);
                stdLFA=nanstd(actLFA);
                durLFA=length(find(actLFA>0));
                prctLFA=length(find(actLFA>0))/length(actLFA);
                transLFARFA=sum(actLFA(1:end-1) > 0 & actLFA(2:end) < 0); % transitions between positive and negative values



                % add to mastertable
                masterLFAtable(idx,:)={
                    'ENGINE',...
                    actSu,...
                    population,...
                    'EC',...
                    num2str(sess),...
                    typeLabel,...
                    num2str(meanLFA),...
                    num2str(stdLFA),...
                    num2str(durLFA),...
                    num2str(prctLFA),...
                    num2str(transLFARFA),...
                    };
                idx=idx+1;

            end


        end
    end

end
cell2csv(['/autofs/vast/neuromod/tDCS_EEG/Projects/16_ENGINE/analysis/group/ENGINE_restingState_LFA_' date zfx '.csv'],masterLFAtable);

disp('saved.')

