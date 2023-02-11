
addpath(genpath('utility'));
addpath(genpath('bosaris_toolkit'));
addpath(genpath('ASVspoof2017_train_dev2'));
addpath(genpath('voicebox'));
 
pathToDatabase = fullfile('E:\asv_spoof\baseline_CM\ASVspoof2017_train_dev2','wav');

frame_length = 0.02; 
frame_hop = 0.01; 
n_MFCC = 13; 



fileID = fopen('ASVspoof2017_V2_train+dev.trn.txt');
protocol = textscan(fileID, '%s%s%s%s%s%s%s');
fclose(fileID);


filelist = protocol{1};
labels = protocol{2};


genuineIdx = find(strcmp(labels,'genuine'));
spoofIdx = find(strcmp(labels,'spoof'));


disp('Extracting features for GENUINE training data...');
genuineFeatureCell = cell(size(genuineIdx));
parfor i=1:length(genuineIdx)
    filePath = fullfile(pathToDatabase,'train+dev',filelist{genuineIdx(i)});
    [x,fs] = audioread(filePath);
    genuineFeatureCell{i} = melcepst(x, fs, '0dD', n_MFCC, floor(3*log(fs)) ,frame_length * fs, frame_hop * fs)';
end
disp('Done!');


disp('Extracting features for SPOOF training data...');
spoofFeatureCell = cell(size(spoofIdx));
parfor i=1:length(spoofIdx)
    filePath = fullfile(pathToDatabase,'train+dev',filelist{spoofIdx(i)});
    [x,fs] = audioread(filePath);
    spoofFeatureCell{i} = melcepst(x, fs, '0dD', n_MFCC, floor(3*log(fs)) ,frame_length * fs, frame_hop * fs)';
end
disp('Done!');


disp('Training GMM for GENUINE...');
[genuineGMM.m, genuineGMM.s, genuineGMM.w] = vl_gmm([genuineFeatureCell{:}], 512, 'verbose', 'MaxNumIterations',100);
disp('Done!');


disp('Training GMM for SPOOF...');
[spoofGMM.m, spoofGMM.s, spoofGMM.w] = vl_gmm([spoofFeatureCell{:}], 512, 'verbose', 'MaxNumIterations',100);
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
   
    x_mfcc = melcepst(x, fs, '0dD', n_MFCC, floor(3*log(fs)) ,frame_length * fs, frame_hop * fs)';

    llk_genuine = mean(compute_llk(x_mfcc,genuineGMM.m,genuineGMM.s,genuineGMM.w));
    llk_spoof = mean(compute_llk(x_mfcc,spoofGMM.m,spoofGMM.s,spoofGMM.w));
    
    scores(i) = llk_genuine - llk_spoof;
end
disp('Done!');


[Pmiss,Pfa] = rocch(scores(strcmp(labels,'genuine')),scores(strcmp(labels,'spoof')));
EER = rocch2eer(Pmiss,Pfa) * 100; 
fprintf('EER is %.2f\n', EER);
