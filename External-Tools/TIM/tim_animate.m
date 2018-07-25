%% Normalising an image sequence through TIM.
%%
%% Copyright Tomas Pfister 2011
%% Conditions for use: see license.txt
%%
%% Usage: tim_animate(srcdir, dstdir, length_of_normalised_sequence)

function tim_animate(imDir, outputDir, expandNum)

% Load images into a vector
%imgList = readImageList(imDir);
[imv,imSize] = loadImages(imDir);

% Create animation
aniModel = tim_getAniModel(imv);
aniData = tim_genAnimationData(aniModel, 0, 1, expandNum);

% Extract interpolated image sequence

for i = 1:expandNum
    img = reshape(aniData(:, i), imSize);
    img = imresize(img, [224, 224]);
    img = uint8(img);
    if i < 10
       imwrite(img, strcat(outputDir, '/00', num2str(i), '.jpg'));    
    else
       imwrite(img, strcat(outputDir, '/0', num2str(i), '.jpg')); 
    end
end


%% Function for loading images in a directory
function [imv,imSize]=loadImages(imDir)

skipInDirList = 2;  %skip . and .. in directory lists

% Inspect first image and prealloate for speed
imDirList = dir(imDir);
imDirListLength = length(imDirList);

imFile = imDirList(skipInDirList+1).name;
im = imread(strcat(imDir, '/', imFile));

% JUST FOR CROPPED CK ONLY, COMMENT OTHERWISE
im = imresize(im, [300, 300]); % this line for cropped only

imSize = size(im);

imv = zeros(size(im(:), 1), imDirListLength-skipInDirList);

% Load images
for i = (skipInDirList+1):imDirListLength
    imFile = imDirList(i).name;
    imPath = strcat(imDir, '/', imFile);
    disp(imPath);
    im = imread(imPath);
    
    
    % JUST FOR CROPPED CK ONLY, COMMENT OTHERWISE
    im = imresize(im, [300, 300]);    % this line for cropped only
    
    
    imv(:, i-skipInDirList) = im(:);  %vectorise image
end
