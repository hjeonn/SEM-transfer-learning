% 200 samples per class for oversampling
maxSamples = 201; 

%% Dataset 1 (e1) oversampling
oversampledDataset1 = oversampleDatastore(dataset1, maxSamples);

disp("After Oversampling Dataset1:");
disp(countEachLabel(oversampledDataset1));



% Step 1: testset2 (20 per class)
[testSet2, dataset2_remain] = splitEachLabel(dataset2, 20, 'randomized');

%% Dataset 2 (e2) oversampling
oversampledDataset2 = oversampleDatastore(dataset2_remain, maxSamples);

disp("After Oversampling Dataset2:");
disp(countEachLabel(oversampledDataset2));

minTestSamples = min(countEachLabel(testSet2).Count);
[testSet2, ~] = splitEachLabel(testSet2, minTestSamples, 'randomized');

disp("Final Balanced Test Set:");
disp(countEachLabel(testSet2));

% data augmentation
imageAugmenter = imageDataAugmenter( ...
    'RandRotation',[-10,10], ...
    'RandXReflection',true, ...
    'RandScale', [0.8, 1.2]);



% load network (EfficientNet-B0)
net_1 = imagePretrainedNetwork("efficientnetb0", "NumClasses", numel(classNames), 'Weights', 'pretrained');
imageSize = net_1.Layers(1).InputSize;

% Resize only (TestSet2 with no augmentation)

augTrainSet1 = augmentedImageDatastore(imageSize, oversampledDataset1,DataAugmentation=imageAugmenter, ColorPreprocessing="gray2rgb");
augTrainSet2 = augmentedImageDatastore(imageSize, oversampledDataset2,DataAugmentation=imageAugmenter, ColorPreprocessing="gray2rgb");
augTestSet2  = augmentedImageDatastore(imageSize, testSet2,  ColorPreprocessing="gray2rgb");



% training options
options = trainingOptions("adam", ...
    MaxEpochs=1, ...
    MiniBatchSize=16, ...
    Plots="training-progress", ...
    Metrics="accuracy", ...
    Verbose=false);

% #of loop
numRuns = 10;
allConfMat1 = [];
allConfMat2 = [];

%analyzeNetwork(net_1)

if ~exist('confmat_images', 'dir'), mkdir('confmat_images'); end

for run = 1:numRuns
    fprintf('\n===== Run %d/%d =====\n', run, numRuns);

    % Model 1: train on dataset1 --> fintune with trainSet2
    model1 = trainnet(augTrainSet1, net_1, "crossentropy", options);
    model1 = trainnet(augTrainSet2, model1, "crossentropy", options);

    pred1 = minibatchpredict(model1, augTestSet2);
    pred1 = scores2label(pred1, classNames);
    true1 = testSet2.Labels;
    conf1 = confusionmat(true1, pred1);
    allConfMat1(:, :, run) = conf1;

    % Model 2: train with (only) trainset2
    model2 = trainnet(augTrainSet2, net_1, "crossentropy", options);
    pred2 = minibatchpredict(model2, augTestSet2);
    pred2 = scores2label(pred2, classNames);
    true2 = testSet2.Labels;
    conf2 = confusionmat(true2, pred2);
    allConfMat2(:, :, run) = conf2;

    % Model 3: train on dataset1, test on trainset2
    model3 = trainnet(augTrainSet1, net_1, "crossentropy", options);
    pred3 = minibatchpredict(model3, augTestSet2);
    pred3 = scores2label(pred3, classNames);
    true3 = testSet2.Labels;
    conf3 = confusionmat(true3, pred3);
    allConfMat3(:, :, run) = conf3;

    saveas(confusionchart(conf1, classNames), sprintf('confmat_images/confmat_model1_run%d.png', run));
    saveas(confusionchart(conf2, classNames), sprintf('confmat_images/confmat_model2_run%d.png', run));
    saveas(confusionchart(conf3, classNames), sprintf('confmat_images/confmat_model3_run%d.png', run));

end

meanConfMat1 = mean(allConfMat1, 3);
meanConfMat2 = mean(allConfMat2, 3);
meanConfMat3 = mean(allConfMat3, 3);  % Model 3도 추가했다면


% mean confusion matrix
meanConfMat1 = mean(allConfMat1, 3);
meanConfMat2 = mean(allConfMat2, 3);
meanConfMat3 = mean(allConfMat3, 3);  % Model 3

% --- Model 1 ---
TP1 = diag(meanConfMat1);
total1 = sum(meanConfMat1, 2);
FN1 = total1 - TP1;
FP1 = sum(meanConfMat1, 1)' - TP1;
acc1 = TP1 ./ total1;
recall1 = TP1 ./ (TP1 + FN1+ eps);
prec1 = TP1 ./ (TP1 + FP1+ eps);
f1_1 = 2 * (prec1 .* recall1) ./ (prec1 + recall1+ eps);

% --- Model 2 ---
TP2 = diag(meanConfMat2);
total2 = sum(meanConfMat2, 2);
FN2 = total2 - TP2;
FP2 = sum(meanConfMat2, 1)' - TP2;
acc2 = TP2 ./ total2;
recall2 = TP2 ./ (TP2 + FN2+ eps);
prec2 = TP2 ./ (TP2 + FP2+ eps);
f1_2 = 2 * (prec2 .* recall2) ./ (prec2 + recall2+ eps);

% ---  Model 3 ---
TP3 = diag(meanConfMat3);
total3 = sum(meanConfMat3, 2);
FN3 = total3 - TP3;
FP3 = sum(meanConfMat3, 1)' - TP3;
acc3 = TP3 ./ total3;
recall3 = TP3 ./ (TP3 + FN3+ eps);
prec3 = TP3 ./ (TP3 + FP3+ eps);
f1_3 = 2 * (prec3 .* recall3) ./ (prec3 + recall3+ eps);

% class-wise comparison
fprintf('\n==== Class-wise Performance Comparison ====\n');
fprintf('%-20s | Acc1  F1-1 | Acc2  F1-2 | Acc3  F1-3\n', 'Class');
fprintf(repmat('-', 1, 75)); fprintf('\n');

for i = 1:length(classNames)
    fprintf('%-20s | %.2f   %.2f | %.2f   %.2f | %.2f   %.2f\n', ...
        classNames{i}, ...
        acc1(i), f1_1(i), ...
        acc2(i), f1_2(i), ...
        acc3(i), f1_3(i));
end



fprintf('\n==== Class-wise Performance Comparison ====\n');
fprintf('%-20s | Acc1  F1-1 | Acc2  F1-2 | Acc3  F1-3\n', 'Class');
fprintf(repmat('-', 1, 75)); fprintf('\n');

for i = 1:length(classNames)
    fprintf('%-20s | %.2f   %.2f | %.2f   %.2f | %.2f   %.2f\n', ...
        classNames{i}, ...
        acc1(i), f1_1(i), ...
        acc2(i), f1_2(i), ...
        acc3(i), f1_3(i));
end

% Visualize Mean Confusion Matrix 
figure;
subplot(1,3,1)
confusionchart(round(meanConfMat1), classNames);
title('Mean Confusion Matrix - Model 1');

subplot(1,3,2)
confusionchart(round(meanConfMat2), classNames);
title('Mean Confusion Matrix - Model 2');

subplot(1,3,3)
confusionchart(round(meanConfMat3), classNames);
title('Mean Confusion Matrix - Model 3');
