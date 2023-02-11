

clear; close all; clc;


addpath(genpath('utility'));
addpath(genpath('LFCC'));
addpath(genpath('CQCC_v1.0'));
addpath(genpath('voicebox'));
addpath(genpath('bosaris_toolkit'));
addpath(genpath('ASVspoof2017_train_dev2'));

 
pathToDatabase = fullfile('E:\asv_spoof\baseline_CM\ASVspoof2017_train_dev2','wav');



fileID = fopen('ASVspoof2017_V2_train+dev.trn.txt');
protocol = textscan(fileID, '%s%s%s%s%s%s%s');
fclose(fileID);


filelist = protocol{1};
labels = protocol{2};




genuineIdx = find(strcmp(labels,'genuine'));
spoofIdx = find(strcmp(labels,'spoof'));


disp('Extracting CQCC features for GENUINE training data...');
cqcc_genuineFeatureCell = cell(size(genuineIdx));
parfor i=1:length(genuineIdx)
    filePath = fullfile(pathToDatabase,'train+dev',filelist{genuineIdx(i)});
    [x,fs] = audioread(filePath);
    tmp_fea = cqcc(x, fs, 96, fs/2, fs/2^10, 16, 29, 'ZsdD');
    cqcc_genuineFeatureCell{i} = tmp_fea
end
disp('Done!');

disp('Extracting lfcc features for GENUINE training data...');
mfcc_genuineFeatureCell = cell(size(genuineIdx));
parfor i=1:length(genuineIdx)
    filePath = fullfile(pathToDatabase,'train+dev',filelist{genuineIdx(i)});
    [x,fs] = audioread(filePath);
    [stat,delta,double_delta] = extract_lfcc(x,fs,20,512,20);
    lfcc_genuineFeatureCell{i} = [stat delta double_delta]';
end
disp('Done!');


disp('Extracting CQCC features for SPOOF training data...');
cqcc_spoofFeatureCell = cell(size(spoofIdx));
parfor i=1:length(spoofIdx)
    filePath = fullfile(pathToDatabase,'train+dev',filelist{spoofIdx(i)});
    [x,fs] = audioread(filePath);
    tmp_fea = cqcc(x, fs, 96, fs/2, fs/2^10, 16, 29, 'ZsdD');
    cqcc_spoofFeatureCell{i} = tmp_fea
end
disp('Done!');

disp('Extracting lfcc features for SPOOF training data...');
mfcc_spoofFeatureCell = cell(size(spoofIdx));
parfor i=1:length(spoofIdx)
    filePath = fullfile(pathToDatabase,'train+dev',filelist{spoofIdx(i)});
    [x,fs] = audioread(filePath);
    [stat,delta,double_delta] = extract_lfcc(x,fs,20,512,20);
    lfcc_spoofFeatureCell{i} = [stat delta double_delta]';
end
disp('Done!');



disp('Training CQCC GMM for GENUINE...');
[cqcc_genuineGMM.m, cqcc_genuineGMM.s, cqcc_genuineGMM.w] = vl_gmm([cqcc_genuineFeatureCell{:}], 512, 'verbose', 'MaxNumIterations', 100);
disp('Done!');

disp('Training MFCC GMM for GENUINE...');
[lfcc_genuineGMM.m, lfcc_genuineGMM.s, lfcc_genuineGMM.w] = vl_gmm([lfcc_genuineFeatureCell{:}], 512, 'verbose', 'MaxNumIterations', 100);
disp('Done!');


disp('Training CQCC GMM for SPOOF...');
[cqcc_spoofGMM.m, cqcc_spoofGMM.s, cqcc_spoofGMM.w] = vl_gmm([cqcc_spoofFeatureCell{:}], 512, 'verbose', 'MaxNumIterations', 100);
disp('Done!');

disp('Training MFCC GMM for SPOOF...');
[lfcc_spoofGMM.m, lfcc_spoofGMM.s, lfcc_spoofGMM.w] = vl_gmm([lfcc_spoofFeatureCell{:}], 512, 'verbose', 'MaxNumIterations', 100);
disp('Done!');



fileID = fopen('ASVspoof2017_V2_eval.trl.txt');
protocol = textscan(fileID, '%s%s%s%s%s%s%s');
fclose(fileID);


filelist = protocol{1};
labels = protocol{2};


scores = zeros(size(filelist));
disp('Computing scores for development trials...');
parfor i=1:length(filelist)
    filePath = fullfile(pathToDatabase,'eval',filelist{i});
    [x,fs] = audioread(filePath);
    
    x_cqcc = cqcc(x, fs, 96, fs/2, fs/2^10, 16, 29, 'ZsdD');
    [stat,delta,double_delta] = extract_lfcc(x,fs,20,512,20);
        x_fea = [stat delta double_delta]';
    
    llk_genuine1 = mean(compute_llk(x_cqcc,cqcc_genuineGMM.m,cqcc_genuineGMM.s,cqcc_genuineGMM.w));
    llk_spoof1 = mean(compute_llk(x_cqcc,cqcc_spoofGMM.m,cqcc_spoofGMM.s,cqcc_spoofGMM.w));

    llk_genuine2 = mean(compute_llk(x_fea,lfcc_genuineGMM.m,lfcc_genuineGMM.s,lfcc_genuineGMM.w));
    llk_spoof2 = mean(compute_llk(x_fea,lfcc_spoofGMM.m,lfcc_spoofGMM.s,lfcc_spoofGMM.w));
    
    scores(i) = llk_genuine1 + llk_genuine2 - llk_spoof1 - llk_spoof2;
    cqcc_scores(i) = llk_genuine1 - llk_spoof1
    lfcc_scores(i) = llk_genuine2 - llk_spoof2
end
disp('Done!');


[Pmiss,Pfa] = rocch(cqcc_scores(strcmp(labels,'genuine')),cqcc_scores(strcmp(labels,'spoof')));
EER = rocch2eer(Pmiss,Pfa) * 100; 
fprintf('cqcc EER is %.2f\n', EER);


[Pmiss,Pfa] = rocch(lfcc_scores(strcmp(labels,'genuine')),lfcc_scores(strcmp(labels,'spoof')));
EER = rocch2eer(Pmiss,Pfa) * 100;
fprintf('mfcc EER is %.2f\n', EER);


[Pmiss,Pfa] = rocch(scores(strcmp(labels,'genuine')),scores(strcmp(labels,'spoof')));
EER = rocch2eer(Pmiss,Pfa) * 100;
fprintf('combine EER is %.2f\n', EER);
