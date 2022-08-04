%% Script Information
%Jared Rifkin
%08/04/22
%HCPGrouping.m

%This script creates the groups identified in Rifkin et al., 2022.
%% GE Computing / Data Transforming
clear; close all;
addpath('') %Add Brain Connectivity Toolbox path here
load('HCP1200_SC_1065subjects.mat')

systems = ["Vis", "SomMot", "DorsAttn", "SalVenAttn", "Lim", "Cont", "Default"];
sysIdx = {[01:09 51:58]; ...
          [10:15 59:66]; ...
          [16:23 67:73]; ...
          [24:30 74:78]; ...
          [31:33 79:80]; ...
          [34:37 81:89]; ...
          [38:50 90:100]};

N = length(HCP1200_SC(1).SC); %# nodes
subs = length(HCP1200_SC); %# subjects
wSC = zeros(7,7,subs); %changes from structure to array
wSC_fine = zeros(N,N,subs);
gen = zeros(subs,1,'logical');

%Iterate through subjects to get edgewise correlations
for i = 1:subs
    for i2 = 1:7
        for i3 = 1:7
            wSC(i2,i3,i) = mean(HCP1200_SC(i).wSC(sysIdx{i2},sysIdx{i3}),'all');
        end
    end
    wSC_fine(:,:,i) = HCP1200_SC(i).wSC;
    if strcmpi(HCP1200_SC(i).sex,'F')
        gen(i) = true;
    end
end

%Split into gender
wSC_female = wSC(:,:,gen); fems = size(wSC_female,3);
wSC_male = wSC(:,:,~gen); mals = size(wSC_male,3);
wSC_female_fine = wSC_fine(:,:,gen);
wSC_male_fine = wSC_fine(:,:,~gen);
wSC_malfem = cat(3,wSC_male,wSC_female);

ID_fem = [HCP1200_SC(gen).ID];
ID_mal = [HCP1200_SC(~gen).ID];
IDS_SC = [ID_mal ID_fem];

%% Pearson Corr Distance
subs = size(wSC_malfem,3);
corDist = zeros(subs);
for i1 = 1:subs
    for i2 = 1:subs
        dummy1 = triu(wSC(:,:,i1));
        dummy2 = triu(wSC(:,:,i2));
        dummy3 = corrcoef(dummy1(:),dummy2(:));
        corDist(i1,i2) = dummy3(2);
    end
end
maxCor = max(corDist,[],'all');
corDist = corDist./max(corDist,[],'all');

corDist_fem = zeros(fems);
for i1 = 1:fems
    for i2 = 1:fems
        dummy1 = triu(wSC_female(:,:,i1));
        dummy2 = triu(wSC_female(:,:,i2));
        dummy3 = corrcoef(dummy1(:),dummy2(:));
        corDist_fem(i1,i2) = dummy3(2);
    end
end
corDist_fem = corDist_fem./maxCor;

corDist_mal = zeros(mals);
for i1 = 1:mals
    for i2 = 1:mals
        dummy1 = triu(wSC_male(:,:,i1));
        dummy2 = triu(wSC_male(:,:,i2));
        dummy3 = corrcoef(dummy1(:),dummy2(:));
        corDist_mal(i1,i2) = dummy3(2);
    end
end
corDist_mal = corDist_mal./maxCor;


%% Scaling and Modularity
close all;
gammas = 0.01:0.01:1;
for gamI = 1:length(gammas)
    gammaBig = gammas(gamI);
    [Ci_fem_cor,Q_fem_cor] = modularity_und(corDist_fem,gammaBig);
    [Ci_mal_cor,Q_mal_cor] = modularity_und(corDist_mal,gammaBig);
    if gammaBig+Q_fem_cor > 1 && gammaBig+Q_mal_cor > 1 && max(Ci_fem_cor) > 1 && max(Ci_mal_cor) > 1
        fprintf('Number of female modules: %d\n',max(Ci_fem_cor));
        fprintf('Number of male modules: %d\n',max(Ci_mal_cor));
        break
    end
end

for gamI = 1:length(gammas)
    gammaBig = gammas(gamI);
    [Ci_cor,Q_cor] = modularity_und(corDist,gammaBig);
    if gammaBig+Q_cor > 1 && max(Ci_cor) > 1
        fprintf('Number of all modules: %d\n',max(Ci_cor));
        break
    end
end
%% K-fold cross validation
randIdx_mal = randperm(length(wSC_male));
randIdx_fem = randperm(length(wSC_female));
k = 5;
Ci_mal_sub = zeros(length(wSC_male),k);
Ci_fem_sub = zeros(length(wSC_female),k);
sucRate_mal = zeros(k,1);
sucRate_fem = zeros(k,1);
gammaOut = zeros(k,1);

for kI = 1:k
    randIdxIdx_mal = 1+ceil((kI-1)*length(randIdx_mal)/k):ceil(kI*length(randIdx_mal)/k);
    randIdxIdx_fem = 1+ceil((kI-1)*length(randIdx_fem)/k):ceil(kI*length(randIdx_fem)/k);
    logIdx_mal = true(length(wSC_male),1);
    logIdx_mal(randIdx_mal(randIdxIdx_mal)) = 0;
    logIdx_fem = true(length(wSC_female),1);
    logIdx_fem(randIdx_fem(randIdxIdx_fem)) = 0;

    kFoldDist_mal = corDist_mal(logIdx_mal,logIdx_mal);
    kFoldDist_fem = corDist_fem(logIdx_fem,logIdx_fem);

    for gamI = 1:length(gammas)
        gamma = gammas(gamI);
        [Ci_fem_sub(logIdx_fem,kI),Q_fem_sub] = modularity_und(kFoldDist_fem,gamma);
        [Ci_mal_sub(logIdx_mal,kI),Q_mal_sub] = modularity_und(kFoldDist_mal,gamma);


        if gamma+Q_fem_sub > 1 && gamma+Q_mal_sub > 1 && max(Ci_fem_sub(:,kI)) > 1 && max(Ci_mal_sub(:,kI)) > 1
            fprintf('Number of female modules: %d\n',max(Ci_fem_sub(:,kI)));
            fprintf('Number of male modules: %d\n',max(Ci_mal_sub(:,kI)));
            gammaOut(kI) = gamma;
            break
        end
    end
    
    if round(mean(Ci_mal_sub(Ci_mal_cor==1&Ci_mal_sub(:,kI)~=0,kI))) == 2
        dummy = zeros(length(Ci_mal_sub),1);
        dummy(Ci_mal_sub(:,kI)==1) = 2;
        dummy(Ci_mal_sub(:,kI)==2) = 1;
        Ci_mal_sub(:,kI) = dummy;
    end

    dummy1 = mean(corDist_mal(Ci_mal_sub(:,kI)==1,randIdx_mal(randIdxIdx_mal)));
    dummy2 = mean(corDist_mal(Ci_mal_sub(:,kI)==2,randIdx_mal(randIdxIdx_mal)));
    Ci_mal_sub(randIdx_mal(randIdxIdx_mal(dummy1 > dummy2)),kI) = 1;
    Ci_mal_sub(randIdx_mal(randIdxIdx_mal(dummy1 < dummy2)),kI) = 2;
    
    sucRate_mal(kI) = sum(Ci_mal_sub(randIdx_mal(randIdxIdx_mal),kI)==Ci_mal_cor(randIdx_mal(randIdxIdx_mal)))/length(randIdxIdx_mal);

    if round(mean(Ci_fem_sub(Ci_fem_cor==1&Ci_fem_sub(:,kI)~=0,kI))) == 2
        dummy = zeros(length(Ci_fem_sub),1);
        dummy(Ci_fem_sub(:,kI)==1) = 2;
        dummy(Ci_fem_sub(:,kI)==2) = 1;
        Ci_fem_sub(:,kI) = dummy;
    end
    
    dummy1 = mean(corDist_fem(Ci_fem_sub(:,kI)==1,randIdx_fem(randIdxIdx_fem)));
    dummy2 = mean(corDist_fem(Ci_fem_sub(:,kI)==2,randIdx_fem(randIdxIdx_fem)));
    Ci_fem_sub(randIdx_fem(randIdxIdx_fem(dummy1 > dummy2)),kI) = 1;
    Ci_fem_sub(randIdx_fem(randIdxIdx_fem(dummy1 < dummy2)),kI) = 2;
    
    sucRate_fem(kI) = sum(Ci_fem_sub(randIdx_fem(randIdxIdx_fem),kI)==Ci_fem_cor(randIdx_fem(randIdxIdx_fem)))/length(randIdxIdx_fem);
end