function f1 = computeMacroF1(yTrue, yPred, classes)
    cm = confusionmat(yTrue, yPred);
    numClasses = numel(classes);
    f1 = 0;
    for i = 1:numClasses
        TP = cm(i, i);
        FP = sum(cm(:, i)) - TP;
        FN = sum(cm(i, :)) - TP;
        precision = TP / (TP + FP + eps);
        recall = TP / (TP + FN + eps);
        f1 = f1 + 2 * (precision * recall) / (precision + recall + eps);
    end
    f1 = f1 / numClasses;
end
