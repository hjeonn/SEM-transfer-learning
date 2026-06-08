function imgOut = cropBottomTenth(img)
    if ischar(img)
        img = imread(img); % 파일 경로로 호출되었을 경우
    end

    % Grayscale → RGB 통일
    if size(img, 3) == 1
        img = repmat(img, [1 1 3]);
    end

    % 하단 1/10 제거
    [h, w, ~] = size(img);
    cropH = round(h / 10);
    imgOut = img(1:end - cropH, :, :);  % 상단 9/10만 유지

    % (선택) 크기 통일하고 싶다면 아래 추가
    % imgOut = imresize(imgOut, [224, 224]);
end
