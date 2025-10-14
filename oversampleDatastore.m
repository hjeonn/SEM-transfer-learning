function oversampledDS = oversampleDatastore(inputDS, maxSamples)

    countTable = countEachLabel(inputDS);
    
    newFiles = [];
    newLabels = categorical([]);

    for i = 1:height(countTable)
        classLabel = countTable.Label(i);
        numSamples = countTable.Count(i);
        

        classFiles = inputDS.Files(inputDS.Labels == classLabel);

        while numSamples < maxSamples

            filesToAdd = min(maxSamples - numSamples, length(classFiles));
            

            addFiles = classFiles(randperm(length(classFiles), filesToAdd));
            

            newFiles = [newFiles; addFiles];
            newLabels = [newLabels; categorical(repmat(string(classLabel), length(addFiles), 1))];
            

            numSamples = numSamples + length(addFiles);
        end
    end
    

    oversampledDS = imageDatastore([inputDS.Files; newFiles], 'Labels', [inputDS.Labels; newLabels]);
end