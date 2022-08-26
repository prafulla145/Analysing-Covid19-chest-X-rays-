imds = imageDatastore('Stat_ResNet18','IncludeSubfolders',true,'LabelSource','foldernames');
[imdsTrain,imdsTest] = splitEachLabel(imds,0.7);
net = resnet18;
inputSize = net.Layers(1).InputSize;
analyzeNetwork(net);
augimdsTrain = augmentedImageDatastore(inputSize(1:2),imdsTrain);
augimdsTest = augmentedImageDatastore(inputSize(1:2),imdsTest);
layer = 'pool5';
featuresTrain = activations(net,augimdsTrain,layer,'OutputAs','rows');
featuresTest = activations(net,augimdsTest,layer,'OutputAs','rows');

whos featuresTrain
YTrain = imdsTrain.Labels;
YTest = imdsTest.Labels;

classifier = fitcecoc(featuresTrain,YTrain);
YPred = predict(classifier,featuresTest);
accuracy = mean(YPred == YTest);

% [a,b]=size(featuresTest);
% [c,d]=size(featuresTrain);
% e=a+c;
% final=zeros(e,512);

% featuresTrain(:,(end+1))=YTrain;
% featuresTest(:,(end+1))=YTest;
final=combine(featuresTrain,featuresTest);
final1=final([6:7 10 1:5 8:9],:);