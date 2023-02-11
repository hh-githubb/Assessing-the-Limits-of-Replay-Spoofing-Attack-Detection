

clear; close all; clc;

addpath(genpath('utility'));
addpath(genpath('LFCC'));
addpath(genpath('CQCC_v1.0'));
addpath(genpath('voicebox'));
addpath(genpath('bosaris_toolkit'));
addpath(genpath('ASVspoof2017_train_dev2'));


pathToDatabase = fullfile('E:\asv_spoof\baseline_CM\ASVspoof2017_train_dev2','wav');

frame_length = 0.02; 
frame_hop = 0.01; 
n_MFCC = 13; 
delta_feature = '0'; 

fileID = fopen('ASVspoof2017_V2_train.trn.txt');
protocol = textscan(fileID, '%s%s%s%s%s%s%s');
fclose(fileID);


filelist = protocol{1};
labels = protocol{2};




genuineIdx = find(strcmp(labels,'genuine'));
spoofIdx = find(strcmp(labels,'spoof'));


disp('Extracting lfcc features for GENUINE training data...');
mfcc_genuineFeatureCell = cell(size(genuineIdx));
parfor i=1:length(genuineIdx)
    filePath = fullfile(pathToDatabase,'train',filelist{genuineIdx(i)});
    [x,fs] = audioread(filePath);
    [stat,delta,double_delta] = extract_lfcc(x,fs,20,512,20);
    lfcc_genuineFeatureCell{i} = [stat delta double_delta]';
end
disp('Done!');


disp('Extracting mfcc features for GENUINE training data...');
mfcc_genuineFeatureCell = cell(size(genuineIdx));
parfor i=1:length(genuineIdx)
    filePath = fullfile(pathToDatabase,'train',filelist{genuineIdx(i)});
    [x,fs] = audioread(filePath);
    
    mfcc_genuineFeatureCell{i} = melcepst(x, fs, '0dD', n_MFCC, floor(3*log(fs)) ,frame_length * fs, frame_hop * fs)';
end
disp('Done!');

disp('Extracting lfcc features for SPOOF training data...');
mfcc_spoofFeatureCell = cell(size(spoofIdx));
parfor i=1:length(spoofIdx)
    filePath = fullfile(pathToDatabase,'train',filelist{spoofIdx(i)});
    [x,fs] = audioread(filePath);
    [stat,delta,double_delta] = extract_lfcc(x,fs,20,512,20);
    lfcc_spoofFeatureCell{i} = [stat delta double_delta]';
end
disp('Done!');


disp('Extracting MFCC features for SPOOF training data...');
mfcc_spoofFeatureCell = cell(size(spoofIdx));
parfor i=1:length(spoofIdx)
    filePath = fullfile(pathToDatabase,'train',filelist{spoofIdx(i)});
    [x,fs] = audioread(filePath);
    tmp_fea = melcepst(x, fs, '0dD', n_MFCC, floor(3*log(fs)) ,frame_length * fs, frame_hop * fs)';
    mfcc_spoofFeatureCell{i} = tmp_fea
end
disp('Done!');



disp('Training lFCC GMM for GENUINE...');
[lfcc_genuineGMM.m, lfcc_genuineGMM.s, lfcc_genuineGMM.w] = vl_gmm([lfcc_genuineFeatureCell{:}], 512, 'verbose', 'MaxNumIterations', 100);
disp('Done!');

disp('Training MFCC GMM for GENUINE...');
[mfcc_genuineGMM.m, mfcc_genuineGMM.s, mfcc_genuineGMM.w] = vl_gmm([mfcc_genuineFeatureCell{:}], 512, 'verbose', 'MaxNumIterations', 100);
disp('Done!');


disp('Training MFCC GMM for SPOOF...');
[lfcc_spoofGMM.m, lfcc_spoofGMM.s, lfcc_spoofGMM.w] = vl_gmm([lfcc_spoofFeatureCell{:}], 512, 'verbose', 'MaxNumIterations', 100);
disp('Done!');

disp('Training MFCC GMM for SPOOF...');
[mfcc_spoofGMM.m, mfcc_spoofGMM.s, mfcc_spoofGMM.w] = vl_gmm([mfcc_spoofFeatureCell{:}], 512, 'verbose', 'MaxNumIterations', 100);
disp('Done!');


fileID = fopen('ASVspoof2017_V2_dev.trl.txt');
protocol = textscan(fileID, '%s%s%s%s%s%s%s');
fclose(fileID);


filelist = protocol{1};
labels = protocol{2};


scores = zeros(size(filelist));

disp('Computing scores for development trials...');
parfor i=1:length(filelist)
    filePath = fullfile(pathToDatabase,'dev',filelist{i});
    [x,fs] = audioread(filePath);
    % featrue extraction
   [stat,delta,double_delta] = extract_lfcc(x,fs,20,512,20);
        x_fea = [stat delta double_delta]';
    x_mfcc = melcepst(x, fs, '0dD', n_MFCC, floor(3*log(fs)) ,frame_length * fs, frame_hop * fs)';

    
    llk_genuine1 = mean(compute_llk(x_fea,lfcc_genuineGMM.m,lfcc_genuineGMM.s,lfcc_genuineGMM.w));
    llk_spoof1 = mean(compute_llk(x_fea,lfcc_spoofGMM.m,lfcc_spoofGMM.s,lfcc_spoofGMM.w));

    llk_genuine2 = mean(compute_llk(x_mfcc,mfcc_genuineGMM.m,mfcc_genuineGMM.s,mfcc_genuineGMM.w));
    llk_spoof2 = mean(compute_llk(x_mfcc,mfcc_spoofGMM.m,mfcc_spoofGMM.s,mfcc_spoofGMM.w));
    
    scores(i) = llk_genuine1 + llk_genuine2 - llk_spoof1 - llk_spoof2;
    
end
disp('Done!');



[Pmiss,Pfa] = rocch(scores(strcmp(labels,'genuine')),scores(strcmp(labels,'spoof')));
EER = rocch2eer(Pmiss,Pfa) * 100;
fprintf('combine EER is %.2f\n', EER);
