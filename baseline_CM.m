
clear; close all; clc;


addpath(genpath('utility'));
addpath(genpath('CQCC_v1.0'));
addpath(genpath('bosaris_toolkit'));
addpath(genpath('ASVspoof2017_train_dev1'));

 
pathToDatabase = fullfile('E:\asv_spoof\baseline_CM\ASVspoof2017_train_dev1','wav');



fileID = fopen('ASVspoof2017_train1.trn.txt');
protocol = textscan(fileID, '%s%s%s%s%s%s%s');
fclose(fileID);


filelist = protocol{1};
labels = protocol{2};


genuineIdx = find(strcmp(labels,'genuine'));
spoofIdx = find(strcmp(labels,'spoof'));



disp('Extracting features for GENUINE training data...');
genuineFeatureCell = cell(size(genuineIdx));
parfor i=1:length(genuineIdx)
    filePath = fullfile(pathToDatabase,'train',filelist{genuineIdx(i)});
    [x,fs] = audioread(filePath);
    genuineFeatureCell{i} = cqcc(x, fs, 96, fs/2, fs/2^10, 16, 29, 'ZsdD');
end
disp('Done!');


disp('Extracting features for SPOOF training data...');
spoofFeatureCell = cell(size(spoofIdx));
parfor i=1:length(spoofIdx)
    filePath = fullfile(pathToDatabase,'train',filelist{spoofIdx(i)});
    [x,fs] = audioread(filePath);
    spoofFeatureCell{i} = cqcc(x, fs, 96, fs/2, fs/2^10, 16, 29, 'ZsdD');
end
disp('Done!');


disp('Training GMM for GENUINE...');
[genuineGMM.m, genuineGMM.s, genuineGMM.w] = vl_gmm([genuineFeatureCell{:}], 512, 'verbose', 'MaxNumIterations',100);
disp('Done!');


disp('Training GMM for SPOOF...');
[spoofGMM.m, spoofGMM.s, spoofGMM.w] = vl_gmm([spoofFeatureCell{:}], 512, 'verbose', 'MaxNumIterations',100);
disp('Done!');



fileID = fopen('ASVspoof2017_eval_v2_key.trl.txt');
protocol = textscan(fileID, '%s%s%s%s%s%s%s');
fclose(fileID);


filelist = protocol{1};
labels = protocol{2};


scores = zeros(size(filelist));
disp('Computing scores for evaluation trials...');
parfor i=1:length(filelist)
    filePath = fullfile(pathToDatabase,'eval',filelist{i});
    [x,fs] = audioread(filePath);
    
    x_cqcc = cqcc(x, fs, 96, fs/2, fs/2^10, 16, 29, 'ZsdD');

    
    llk_genuine = mean(compute_llk(x_cqcc,genuineGMM.m,genuineGMM.s,genuineGMM.w));
    llk_spoof = mean(compute_llk(x_cqcc,spoofGMM.m,spoofGMM.s,spoofGMM.w));
    
    scores(i) = llk_genuine - llk_spoof;
end
disp('Done!');


[Pmiss,Pfa] = rocch(scores(strcmp(labels,'genuine')),scores(strcmp(labels,'spoof')));
EER = rocch2eer(Pmiss,Pfa) * 100; 
fprintf('EER is %.2f\n', EER);
Pm=Pmiss*100
Pf=Pfa*100
plot (Pf,Pm,'b')
legend('B02')
