clear; clc; clf; close all;

load iris.data
load seeds.data
load wine.data

% IRIS
[trainInput, trainOutput, testInput, testOutput] = divideSet(iris, 1:4, 5, 3);

genOpt = genfisOptions('SubtractiveClustering');
fis = genfis(trainInput, trainOutput, genOpt);

testResult = Test(fis, testInput, testOutput)
testOptResult = TestOptimized(fis, iris, 1:4, 5, 3)

cvSet = [trainInput, trainOutput];
cvIndices = PrepareCV(cvSet, 5, 5);
cvResult = CrossValidate(5, cvSet, 1:4, 5, cvIndices, genOpt, false)
cvOptResult = CrossValidate(5, cvSet, 1:4, 5, cvIndices, genOpt, true)

% SEEDS
[trainInput, trainOutput, testInput, testOutput] = divideSet(seeds, 1:7, 8, 3);

genOpt = genfisOptions('SubtractiveClustering');
fis = genfis(trainInput, trainOutput, genOpt);

testResult = Test(fis, testInput, testOutput)
testOptResult = TestOptimized(fis, seeds, 1:7, 8, 3)

cvSet = [trainInput, trainOutput];
cvIndices = PrepareCV(cvSet, 8, 5);
cvResult = CrossValidate(5, cvSet, 1:7, 8, cvIndices, genOpt, false)
cvOptResult = CrossValidate(5, cvSet, 1:7, 8, cvIndices, genOpt, true)

% WINE
[trainInput, trainOutput, testInput, testOutput] = divideSet(wine, 2:14, 1, 3);

genOpt = genfisOptions('SubtractiveClustering',...
                    'ClusterInfluenceRange', 1);
fis = genfis(trainInput, trainOutput, genOpt);

testResult = Test(fis, testInput, testOutput)
testOptResult = TestOptimized(fis, wine, 2:14, 1, 3)

cvSet = [trainOutput, trainInput];
cvIndices = PrepareCV(cvSet, 1, 5);
cvResult = CrossValidate(5, cvSet, 2:14, 1, cvIndices, genOpt, false)
cvOptResult = CrossValidate(5, cvSet, 2:14, 1, cvIndices, genOpt, true)

function[result] = Test(fis, testInput, testOutput)
    evalOpt = evalfisOptions('OutOfRangeInputValueMessage', 'none');
    predicted = evalfis(fis, testInput, evalOpt);
    
    diff = predicted - testOutput;
    result = mean(diff.*diff);
    result = result * 100;
    result = 100 - result;

    predicted = round(predicted);
    predicted(predicted > 3) = 3;
    predicted(predicted < 1) = 1;
    figure;
    confusionchart(testOutput, predicted);
end

function[result] = TestOptimized(fis, dataset, inputIndices, outputIndex, typeCount)
    evalOpt = evalfisOptions('OutOfRangeInputValueMessage', 'none');
    [trainInput, trainOutput, testInput, testOutput] = divideSet(dataset, inputIndices, outputIndex, typeCount);

    [in, out] = getTunableSettings(fis);

    params = getTunableValues(fis, [in; out]);
    paramCount = 10;
    optVarCount = length(params);
    maxIt = 10;

    bestDesign = Optimize(paramCount, optVarCount, maxIt, params, in, testInput, testOutput, fis);
    fis = setTunableValues(fis, in, bestDesign);

    predicted = evalfis(fis, testInput, evalOpt);
    
    diff = predicted - testOutput;
    result = mean(diff.*diff);
    result = result * 100;
    result = 100 - result;

    predicted = round(predicted);
    predicted(predicted > 3) = 3;
    predicted(predicted < 1) = 1;
    figure;
    confusionchart(testOutput, predicted);
end

function[trainInput, trainOutput, testInput, testOutput] = divideSet(dataset, inputIndices, outputIndex, typeCount)
    trainData = [];
    testData = [];
    setCount = length(dataset);
    subsetCount = round(setCount/typeCount);
    for i = 1:typeCount
        typeTrainSetIndices = subsetCount*(i-1)+1 : subsetCount*(i-1) + round(0.8*subsetCount);
        typeTestSetIndices = subsetCount*(i-1) + round(0.8*subsetCount) + 1 : subsetCount*i;
        trainData = [trainData; dataset(typeTrainSetIndices, :)];
        testData = [testData; dataset(typeTestSetIndices, :)];
    end
    trainInput = trainData(:, inputIndices);
    trainOutput = trainData(:, outputIndex);
    testInput = testData(:, inputIndices);
    testOutput = testData(:, outputIndex);
end

function[indices] = PrepareCV(dataset, outputIndex, k)
    indices = crossvalind('Kfold', dataset(:, outputIndex), k);
end

function[result] = CrossValidate(k, dataset, inputIndices, outputIndex, indices, genOpt, opt)
    evalOpt = evalfisOptions('OutOfRangeInputValueMessage', 'none');
    
    allOutput = [];
    allPredicted = [];
    
    result = 0;
    for i = 1:k
        fprintf('CV Iteration: %d\n', i);
        test = (indices == i);
        train = ~test;
        trainInput = dataset(train, inputIndices);
        trainOutput = dataset(train, outputIndex);
        testInput = dataset(test, inputIndices);
        testOutput = dataset(test, outputIndex);
    
        if opt
            fis = genfis(trainInput, trainOutput, genOpt);
            [in, out] = getTunableSettings(fis);
            params = getTunableValues(fis, [in; out]);
            paramCount = 10;
            optVarCount = length(params);
            maxIt = 10;
            bestDesign = Optimize(paramCount, optVarCount, maxIt, params, in, testInput, testOutput, fis);
            fis = setTunableValues(fis, in, bestDesign);
        else
            fis = genfis(trainInput, trainOutput, genOpt);
        end
    
        predicted = round(evalfis(fis, testInput, evalOpt));
        predicted(predicted > 3) = 3;
        predicted(predicted < 1) = 1;
        
        allOutput = [allOutput; testOutput];
        allPredicted = [allPredicted; predicted];
        diff = predicted - testOutput;
        result = result + mean(diff.*diff);
    end
    result = result / k * 100;
    result = 100 - result;
    figure
    confusionchart(allOutput, allPredicted)
end

function[bestDesign] = Optimize(popSize, nVar, maxIt, pop, in, testInput, testOutput, fis)
    % Colliding Bodies Optimization - CBO
    
    CB = repmat(pop, popSize, 1) + 0.5 - rand(popSize, nVar);
    
    iter = 0;
    Inf = 1e100;
    bestCost = Inf;
    agentCost = zeros(popSize, 2);
    evalOpt = evalfisOptions('OutOfRangeInputValueMessage', 'none');
    
    while iter < maxIt
        iter = iter + 1;
    
        mass = zeros(popSize, 1);
        for e = 1:popSize
            fis = setTunableValues(fis, in, CB(e, :));
            predicted = evalfis(fis, testInput, evalOpt);
            diff = predicted - testOutput;
            agentCost(e, 1) = mean(diff.*diff);
            agentCost(e, 2) = e;
            mass(e, :) = 1 / (agentCost(e, 1));
        end
    
        agentCost = sortrows(agentCost);
        if agentCost(1, 1) < bestCost
            bestCost = agentCost(1,1);
            bestDesign = CB(agentCost(1,2), :);
        end
    
        for e = 1:popSize/2
            indexS = e;
            indexM = popSize/2 + e;
            COR = (1 - (iter/maxIt));
    
            velMb = (CB(agentCost(indexS, 2), :) - CB(agentCost(indexM, 2), :));
    
            velSa = ((1+COR) * mass(indexM, 1)) / (mass(indexS, 1) + mass(indexM, 1)) * velMb;
    
            velMa = (mass(indexM, 1) - COR * mass(indexS, 1)) / (mass(indexS, 1) + mass(indexM, 1)) * velMb;
    
            CB(agentCost(indexM, 2), :) = CB(agentCost(indexS, 2), :) + 2 * (0.5 - rand(1, nVar)) .* velMa;
            CB(agentCost(indexS, 2), :) = CB(agentCost(indexS, 2), :) + 2 * (0.5 - rand(1, nVar)) .* velSa;
        end
    end
end
